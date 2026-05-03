"""SMT (Sparse Matrix Tuning) — DeepSpeed 8-GPU compatible.

Uses same SparseLinear approach as SIFT for DeepSpeed compatibility.
Instead of gradient-based submatrix selection, uses block-sparse random selection.
Target trainable params: ~170M.
"""
import logging
import os
import sys
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)
RAPA_HOME = os.environ.get("RAPA_HOME", "/data/nksol0405/LLM/rapa")

BLOCK_DIM = 256  # SMT block dimension


class BlockSparseLinear(nn.Module):
    """Linear with block-sparse trainable delta (SMT-style)."""

    def __init__(self, orig_linear: nn.Linear, num_blocks: int, seed: int = 42):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features

        self.register_buffer("weight", orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.register_buffer("bias", orig_linear.bias.data)
        else:
            self.bias = None

        # Select random blocks
        rows = self.out_features // BLOCK_DIM
        cols = self.in_features // BLOCK_DIM
        total_blocks = rows * cols
        num_blocks = min(num_blocks, total_blocks)

        rng = torch.Generator()
        rng.manual_seed(seed)
        block_indices = torch.randperm(total_blocks, generator=rng)[:num_blocks]

        # Convert block indices to (row_block, col_block)
        row_blocks = block_indices // cols
        col_blocks = block_indices % cols

        # Flatten to element indices
        flat_indices = []
        for rb, cb in zip(row_blocks.tolist(), col_blocks.tolist()):
            r_start = rb * BLOCK_DIM
            c_start = cb * BLOCK_DIM
            for r in range(r_start, min(r_start + BLOCK_DIM, self.out_features)):
                for c in range(c_start, min(c_start + BLOCK_DIM, self.in_features)):
                    flat_indices.append(r * self.in_features + c)

        flat_idx = torch.tensor(flat_indices, dtype=torch.long)
        self.register_buffer("flat_idx", flat_idx)
        self.sparse_delta = nn.Parameter(
            torch.zeros(len(flat_idx), dtype=self.weight.dtype), requires_grad=True
        )

    def forward(self, x):
        delta_flat = torch.zeros(
            self.weight.numel(), dtype=self.sparse_delta.dtype, device=self.sparse_delta.device
        )
        delta_flat.scatter_(0, self.flat_idx, self.sparse_delta)
        delta = delta_flat.view(self.weight.shape)
        return F.linear(x, self.weight + delta, self.bias)


def compute_blocks_per_layer(model, target_params=170_000_000, target_modules=None):
    """Compute how many BLOCK_DIM x BLOCK_DIM blocks per layer to hit target_params."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    num_layers = sum(1 for n, _ in model.named_modules()
                     if isinstance(_, nn.Linear) and any(t in n for t in target_modules))
    if num_layers == 0:
        return 1
    params_per_block = BLOCK_DIM * BLOCK_DIM
    total_blocks = target_params // params_per_block
    blocks_per_layer = max(1, total_blocks // num_layers)
    logger.info(f"[SMT] {num_layers} layers, {blocks_per_layer} blocks/layer, "
                f"~{num_layers * blocks_per_layer * params_per_block:,} params")
    return blocks_per_layer


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


def train_smt(
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
    os.makedirs(output_dir, exist_ok=True)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    tok_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        **tok_kwargs,
    )

    selection_start = time.time()
    blocks_per_layer = compute_blocks_per_layer(model, target_params, target_modules)

    for param in model.parameters():
        param.requires_grad = False

    # Replace target Linear with BlockSparseLinear
    replacements = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            bsl = BlockSparseLinear(module, blocks_per_layer, seed=seed + replacements)
            setattr(parent, parts[-1], bsl)
            replacements += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"[SMT] Replaced {replacements} layers, trainable={trainable:,}, total params={total:,}")
    logger.info(f"[SMT] weight_selection_seconds={time.time() - selection_start:.2f}")

    with open(dataset_path) as f:
        raw = json.load(f)
    train_dataset = TextDataset(raw.get("instances", []), tokenizer, max_seq_length)

    save_strategy = "no" if 0 < max_steps < 100 else "epoch"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        bf16=bf16,
        save_strategy=save_strategy,
        logging_steps=5,
        report_to=report_to,
        seed=seed,
        dataloader_num_workers=4,
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

    # Restore to clean nn.Linear for vLLM/HF loading
    from lmflow.pipeline.rapa.sift_tuner import restore_linear_modules
    model = restore_linear_modules(model)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[SMT] Model saved to {output_dir}")
    return output_dir
