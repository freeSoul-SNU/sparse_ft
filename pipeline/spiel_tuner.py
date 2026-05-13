"""SpiEL (Sparse Fine-Tuning) integration within LMFlow."""
import logging
import os
import sys
import json
import sysconfig
import tempfile
import time

import torch
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

RAPA_HOME = os.environ.get("RAPA_HOME", "/data/nksol0405/LLM/rapa")
PEFT_ROOT = os.environ.get("PEFT_DIR", os.path.join(RAPA_HOME, "peft"))
PEFT_SFT_PATH = os.path.join(PEFT_ROOT, "src")
SPARSE_FT_ROOT = os.environ.get(
    "SPARSE_FT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
SPIEL_ROOT = os.path.join(SPARSE_FT_ROOT, "methods", "spiel")
SPIEL_SFT_PATH = os.path.join(SPIEL_ROOT, "peft_sft")
SPIEL_LINEAR_SD_DIR = os.path.join(SPIEL_SFT_PATH, "linear-sd")
SPIEL_LINEAR_SD_BUILD = os.path.join(
    SPIEL_LINEAR_SD_DIR,
    "build",
    f"lib.{sysconfig.get_platform()}-{sys.implementation.cache_tag}",
)
# Force the local PEFT fork and SpiEL implementation to override installed packages.
for p in [PEFT_SFT_PATH, SPIEL_ROOT, SPIEL_SFT_PATH, SPIEL_LINEAR_SD_DIR, SPIEL_LINEAR_SD_BUILD]:
    if p not in sys.path:
        sys.path.insert(0, p)
# Remove cached peft module so fork gets loaded
for mod_name in list(sys.modules.keys()):
    if mod_name == "peft" or mod_name.startswith("peft."):
        del sys.modules[mod_name]

try:
    from lmflow.pipeline.rapa.data_utils import build_lmflow_text_dataset
except ImportError:
    try:
        from .data_utils import build_lmflow_text_dataset
    except ImportError:
        from data_utils import build_lmflow_text_dataset


def _deepspeed_config(default_path):
    value = os.environ.get("DS_CONFIG", default_path)
    if value.lower() in {"", "0", "false", "none", "no"}:
        return None
    return value


def _external_optimizer_deepspeed_config(default_path, output_dir):
    value = _deepspeed_config(default_path)
    if value is None:
        return None
    if os.environ.get("SPIEL_STRIP_DS_OPTIMIZER", "true").lower() not in {"1", "true", "yes"}:
        return value
    with open(value) as f:
        config = json.load(f)
    # SpIEL needs its custom SftAdamW/SftSM3 optimizer because reselection
    # seeds optimizer state for newly grown sparse deltas.
    config.pop("optimizer", None)
    config.pop("scheduler", None)
    config["zero_force_ds_cpu_optimizer"] = False
    os.makedirs(output_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="ds_spiel_external_optimizer_", suffix=".json", dir=output_dir)
    with os.fdopen(fd, "w") as f:
        json.dump(config, f, indent=4)
    return path


def _csv_env(name, default):
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _dataloader_num_workers():
    return int(os.environ.get("RAPA_DATALOADER_NUM_WORKERS", "0"))


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


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
    warmup_steps=0,
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
    from peft_sft.trainer import SftTrainer

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
    delta_dtype = os.environ.get("SPIEL_DELTA_DTYPE", "float32")
    if delta_dtype not in {"auto", "bfloat16", "float16", "float32"}:
        raise ValueError(f"Unsupported SPIEL_DELTA_DTYPE={delta_dtype}")
    target_modules = _csv_env(
        "SPIEL_TARGET_MODULES",
        "q_proj,o_proj,v_proj,k_proj,gate_proj,up_proj,down_proj",
    )
    logger.info(
        "[SpiEL] selection_algorithm=%s, delta_dtype=%s, target_modules=%s",
        os.environ.get("SPIEL_SELECTION_ALGORITHM", "rigl"),
        delta_dtype,
        target_modules,
    )

    peft_config = SftConfig(
        task_type=TaskType.CAUSAL_LM,
        density=density,
        num_tunable_weights=target_params,
        target_modules=target_modules,
        dtype=delta_dtype,
        selection_algorithm=os.environ.get("SPIEL_SELECTION_ALGORITHM", "rigl"),
        reselection_steps=int(os.environ.get("SPIEL_RESELECTION_STEPS", "20")),
        selection_accumulation_steps=int(os.environ.get("SPIEL_SELECTION_ACCUMULATION_STEPS", "5")),
        reselection_rate_policy=os.environ.get("SPIEL_RESELECTION_RATE_POLICY", "linear"),
        initial_reselection_rate=float(os.environ.get("SPIEL_INITIAL_RESELECTION_RATE", "0.2")),
    )
    model = SftModel(model, peft_config, adapter_name="default")
    model.print_trainable_parameters()
    logger.info(f"[SpiEL] weight_selection_seconds={time.time() - selection_start:.2f}")

    train_dataset, data_collator, dynamic_padding = build_lmflow_text_dataset(
        dataset_path, tokenizer, max_seq_length
    )
    logger.info(
        "[SpiEL] data_processing=lmflow_text, samples=%s, dynamic_padding=%s, "
        "label_pad_token_id=-100, attention_mask=true",
        len(train_dataset),
        dynamic_padding,
    )
    dataloader_num_workers = _dataloader_num_workers()
    dataloader_pin_memory = _env_bool("RAPA_DATALOADER_PIN_MEMORY", True)
    logger.info(
        "[SpiEL] dataloader_num_workers=%s, dataloader_pin_memory=%s",
        dataloader_num_workers,
        dataloader_pin_memory,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        bf16=bf16,
        save_strategy="no" if 0 < max_steps < 100 else "epoch",
        logging_steps=5,
        report_to=report_to,
        seed=seed,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        deepspeed=_external_optimizer_deepspeed_config(
            os.path.join(RAPA_HOME, "LMFlow", "configs", "rapa", "ds_zero1_sift.json"),
            output_dir,
        ),
    )

    SpiELTrainer = SftTrainer(Trainer)
    trainer = SpiELTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        sft_config=peft_config,
    )

    if os.environ.get("RESUME_TRAINING", "false").lower() in {"1", "true", "yes"}:
        last_checkpoint = get_last_checkpoint(output_dir)
    else:
        last_checkpoint = None
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if os.environ.get("RAPA_SKIP_SAVE", "false").lower() in {"1", "true", "yes"}:
        logger.info("[SpiEL] RAPA_SKIP_SAVE=true; skipping model merge/save for profiling")
        return output_dir

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
