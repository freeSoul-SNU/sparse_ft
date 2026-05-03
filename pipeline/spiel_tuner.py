"""SpiEL (Sparse Fine-Tuning) integration within LMFlow."""
import logging
import os
import sys
import json
import time

import torch
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

RAPA_HOME = os.environ.get("RAPA_HOME", "/data/nksol0405/LLM/rapa")
PEFT_ROOT = os.environ.get("PEFT_DIR", os.path.join(RAPA_HOME, "peft"))
PEFT_SFT_PATH = os.path.join(PEFT_ROOT, "src")
SPARSE_FT_ROOT = os.environ.get(
    "SPARSE_FT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")),
)
SPIEL_ROOT = os.path.join(SPARSE_FT_ROOT, "methods", "spiel")
SPIEL_SFT_PATH = os.path.join(SPIEL_ROOT, "peft_sft")
SPIEL_LINEAR_SD_BUILD = os.path.join(
    SPIEL_SFT_PATH, "linear-sd", "build", "lib.linux-x86_64-cpython-310"
)
# Force the local PEFT fork and SpiEL implementation to override installed packages.
for p in [PEFT_SFT_PATH, SPIEL_ROOT, SPIEL_SFT_PATH, SPIEL_LINEAR_SD_BUILD]:
    if p not in sys.path:
        sys.path.insert(0, p)
# Remove cached peft module so fork gets loaded
for mod_name in list(sys.modules.keys()):
    if mod_name == "peft" or mod_name.startswith("peft."):
        del sys.modules[mod_name]


def train_spiel(
    model_name_or_path,
    dataset_path,
    output_dir,
    num_train_epochs=1,
    max_steps=-1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    max_seq_length=512,
    target_params=170_000_000,
    bf16=True,
    hf_token=None,
    seed=42,
    report_to="none",
    **kwargs,
):
    from peft.utils import TaskType
    from peft_sft import SftConfig, SftModel

    os.makedirs(output_dir, exist_ok=True)

    tok_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        **tok_kwargs,
    )

    for param in model.parameters():
        param.requires_grad = False

    # Compute total params in linear layers.
    total_linear = sum(
        p.numel() for n, p in model.named_parameters()
        if any(x in n for x in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        and p.ndim == 2
    )
    density = min(target_params / total_linear, 1.0) if total_linear > 0 else 0.025
    logger.info(f"[SpiEL] total_linear={total_linear:,}, density={density:.4f}")

    selection_start = time.time()
    peft_config = SftConfig(
        task_type=TaskType.CAUSAL_LM,
        density=density,
        num_tunable_weights=target_params,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        dtype="bfloat16" if bf16 else "float32",
    )
    model = SftModel(model, peft_config, adapter_name="default")
    model.print_trainable_parameters()
    logger.info(f"[SpiEL] weight_selection_seconds={time.time() - selection_start:.2f}")

    # Load dataset
    with open(dataset_path) as f:
        raw = json.load(f)
    instances = raw.get("instances", [])

    from torch.utils.data import Dataset as TorchDataset

    class TextDataset(TorchDataset):
        def __init__(self, instances, tokenizer, max_len):
            self.data = []
            for item in instances:
                text = item.get("text", "")
                enc = tokenizer(text, truncation=True, max_length=max_len,
                                padding="max_length", return_tensors="pt")
                ids = enc["input_ids"].squeeze()
                self.data.append({"input_ids": ids, "labels": ids.clone()})
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]

    train_dataset = TextDataset(instances, tokenizer, max_seq_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        bf16=bf16,
        save_strategy="no" if 0 < max_steps < 100 else "epoch",
        logging_steps=5,
        report_to=report_to,
        seed=seed,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        deepspeed=os.environ.get("DS_CONFIG", os.path.join(RAPA_HOME, "LMFlow", "configs", "rapa", "ds_zero1_sift.json")),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    last_checkpoint = get_last_checkpoint(output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Merge PEFT adapter into base model and save as full model
    merged_dir = output_dir + "_merged"
    os.makedirs(merged_dir, exist_ok=True)
    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        logger.info(f"[SpiEL] Merged model saved to {output_dir}")
    except Exception as e:
        logger.warning(f"[SpiEL] merge_and_unload failed: {e}, saving adapter instead")
        trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
