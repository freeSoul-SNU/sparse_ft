"""S2FT (Structured Sparse Fine-Tuning) — DeepSpeed 8-GPU compatible.

Selects contiguous rows of weight matrices as trainable (structured sparsity).
Uses buffer-based frozen weights + Parameter delta, same as SIFT/SMT.
Target trainable params: ~170M.
"""
import logging
import os
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


class RowSparseLinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, num_rows: int, seed: int = 42):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.register_buffer("weight", orig_linear.weight.data)
        self.bias = None
        if orig_linear.bias is not None:
            self.register_buffer("bias", orig_linear.bias.data)

        num_rows = min(num_rows, self.out_features)
        start = (self.out_features - num_rows) // 2
        end = start + num_rows
        flat_indices = []
        for r in range(start, end):
            for c in range(self.in_features):
                flat_indices.append(r * self.in_features + c)
        flat_idx = torch.tensor(flat_indices, dtype=torch.long)
        self.register_buffer("flat_idx", flat_idx)
        self.sparse_delta = nn.Parameter(torch.zeros(len(flat_idx), dtype=self.weight.dtype), requires_grad=True)

    def forward(self, x):
        delta_flat = torch.zeros(self.weight.numel(), dtype=self.sparse_delta.dtype, device=self.sparse_delta.device)
        delta_flat.scatter_(0, self.flat_idx, self.sparse_delta)
        return F.linear(x, self.weight + delta_flat.view(self.weight.shape), self.bias)


class TextDataset(TorchDataset):
    def __init__(self, instances, tokenizer, max_len):
        self.data = []
        for item in instances:
            text = item.get("text", "")
            enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze()
            self.data.append({"input_ids": ids, "labels": ids.clone()})
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def train_s2ft(
    model_name_or_path, dataset_path, output_dir, num_train_epochs=1, max_steps=-1,
    per_device_train_batch_size=1, gradient_accumulation_steps=1, learning_rate=5e-5,
    lr_scheduler_type="linear", max_seq_length=512, target_params=170_000_000,
    bf16=True, hf_token=None, seed=42, report_to="none", **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    tok_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16 if bf16 else torch.float32, **tok_kwargs)

    selection_start = time.time()
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and any(t in n for t in target_modules)]
    avg_in = sum(m.in_features for _, m in layers) / max(len(layers), 1)
    rows_per_layer = max(1, int(target_params / (len(layers) * avg_in)))
    logger.info(f"[S2FT] {len(layers)} layers, {rows_per_layer} rows/layer")

    for param in model.parameters():
        param.requires_grad = False

    for i, (name, module) in enumerate(layers):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], RowSparseLinear(module, rows_per_layer, seed=seed + i))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[S2FT] trainable={trainable:,}")
    logger.info(f"[S2FT] weight_selection_seconds={time.time() - selection_start:.2f}")

    with open(dataset_path) as f:
        raw = json.load(f)
    train_dataset = TextDataset(raw.get("instances", []), tokenizer, max_seq_length)

    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=num_train_epochs, max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate, lr_scheduler_type=lr_scheduler_type, bf16=bf16,
        save_strategy="no" if 0 < max_steps < 100 else "epoch", logging_steps=5,
        report_to=report_to, seed=seed, dataloader_num_workers=4, remove_unused_columns=False,
        deepspeed=os.environ.get("DS_CONFIG", os.path.join(RAPA_HOME, "LMFlow", "configs", "rapa", "ds_zero1_sift.json")),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer)
    trainer.train(resume_from_checkpoint=get_last_checkpoint(output_dir))

    from lmflow.pipeline.rapa.sift_tuner import restore_linear_modules
    model = restore_linear_modules(model)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[S2FT] Model saved to {output_dir}")
    return output_dir
