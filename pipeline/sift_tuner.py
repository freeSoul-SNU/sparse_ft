"""SIFT — DeepSpeed 8-GPU compatible via sparse LoRA-style adapters.

Each target Linear gets replaced with SparseLinear that:
1. Keeps original weight frozen
2. Has a small trainable sparse_delta (170M total across all layers)
3. Forward: output = F.linear(x, weight + scatter(sparse_delta), bias)
   - scatter is differentiable via autograd

This works natively with DeepSpeed ZeRO-1: only sparse_delta params are optimized.
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

SPARSE_FT_ROOT = os.environ.get(
    "SPARSE_FT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
RAPA_HOME = os.environ.get("RAPA_HOME", "/data/nksol0405/LLM/rapa")
SIFT_PATH = os.environ.get("SIFT_PATH", os.path.join(SPARSE_FT_ROOT, "methods", "sift"))
if SIFT_PATH not in sys.path:
    sys.path.insert(0, SIFT_PATH)

from sift import SIFT  # noqa: E402


def restore_linear_modules(model):
    """Replace all SparseLinear/BlockSparseLinear/RowSparseLinear back to nn.Linear for clean saving."""
    sparse_types = (SparseLinear,)
    # Import other sparse types if available
    try:
        from lmflow.pipeline.rapa.smt_tuner import BlockSparseLinear
    except ImportError:
        try:
            from smt_tuner import BlockSparseLinear
        except ImportError:
            BlockSparseLinear = None
    if BlockSparseLinear is not None:
        sparse_types = sparse_types + (BlockSparseLinear,)
    try:
        from lmflow.pipeline.rapa.s2ft_tuner import RowSparseLinear
    except ImportError:
        try:
            from s2ft_tuner import RowSparseLinear
        except ImportError:
            RowSparseLinear = None
    if RowSparseLinear is not None:
        sparse_types = sparse_types + (RowSparseLinear,)

    for name, module in list(model.named_modules()):
        if isinstance(module, sparse_types):
            # Merge delta into weight
            with torch.no_grad():
                delta_flat = torch.zeros(module.weight.numel(), dtype=module.weight.dtype, device=module.weight.device)
                delta_flat.scatter_(0, module.flat_idx, module.sparse_delta.data.to(module.weight.dtype))
                merged_weight = module.weight.data + delta_flat.view(module.weight.shape)

            # Create clean nn.Linear
            new_linear = nn.Linear(module.in_features, module.out_features,
                                   bias=module.bias is not None,
                                   device=merged_weight.device, dtype=merged_weight.dtype)
            new_linear.weight.data.copy_(merged_weight)
            if module.bias is not None:
                new_linear.bias.data.copy_(module.bias.data)

            # Replace in model
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_linear)

    return model


def compute_sparse_rate(model, target_params=170_000_000, sparse_modules=None):
    if sparse_modules is None:
        sparse_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    total = sum(p.numel() for n, p in model.named_parameters() if any(m in n for m in sparse_modules))
    rate = target_params / total if total > 0 else 0.0235
    logger.info(f"[SIFT] sparse_modules total: {total:,}, target: {target_params:,}, sparse_rate: {rate:.4f}")
    return min(rate, 1.0)


class SparseLinear(nn.Module):
    """Drop-in replacement for nn.Linear with sparse trainable delta."""

    def __init__(self, orig_linear: nn.Linear, flat_idx: torch.LongTensor):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features

        # Store frozen weight as buffer (NOT parameter) so DeepSpeed ignores it
        self.register_buffer("weight", orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.register_buffer("bias", orig_linear.bias.data)
        else:
            self.bias = None

        # Sparse trainable delta — the ONLY parameter DeepSpeed optimizes
        self.register_buffer("flat_idx", flat_idx)
        self.sparse_delta = nn.Parameter(
            torch.zeros(len(flat_idx), dtype=self.weight.dtype), requires_grad=True
        )

    def forward(self, x):
        # Build full delta: zeros everywhere except at flat_idx positions
        delta_flat = torch.zeros(
            self.weight.numel(), dtype=self.sparse_delta.dtype, device=self.sparse_delta.device
        )
        delta_flat.scatter_(0, self.flat_idx, self.sparse_delta)
        delta = delta_flat.view(self.weight.shape)
        # Forward with frozen weight + trainable delta
        return F.linear(x, self.weight + delta, self.bias)

    def merge_and_get_weight(self):
        """Merge sparse delta into weight (for saving)."""
        with torch.no_grad():
            delta_flat = torch.zeros(self.weight.numel(), dtype=self.weight.dtype, device=self.weight.device)
            delta_flat.scatter_(0, self.flat_idx, self.sparse_delta.data.to(self.weight.dtype))
            return self.weight.data + delta_flat.view(self.weight.shape)


def replace_with_sparse_linear(model, sift_obj, target_modules):
    """Replace target nn.Linear modules with SparseLinear."""
    replacements = 0
    for name in list(sift_obj.sparse_indices.keys()):
        # Navigate to the module
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1]

        if attr_name == "weight":
            # name is like "model.layers.0.self_attn.q_proj.weight"
            # parent is the Linear module, go one level up
            linear_name = parts[-2]
            grandparent = model
            for p in parts[:-2]:
                grandparent = getattr(grandparent, p)
            orig_linear = getattr(grandparent, linear_name)
            if isinstance(orig_linear, nn.Linear):
                flat_idx = sift_obj.sparse_indices[name]
                sparse_linear = SparseLinear(orig_linear, flat_idx)
                setattr(grandparent, linear_name, sparse_linear)
                replacements += 1
        else:
            # name is like "model.layers.0.self_attn.q_proj" → this IS the weight name
            # Find the parent module that has this as a weight
            # Actually, param names include ".weight" at the end
            pass

    # If no replacements with ".weight" suffix, try without
    if replacements == 0:
        for name, flat_idx in sift_obj.sparse_indices.items():
            # name like "model.layers.0.self_attn.q_proj.weight"
            parts = name.rsplit(".", 1)
            if len(parts) == 2 and parts[1] == "weight":
                mod_path = parts[0]
            else:
                mod_path = name
            # Navigate to module
            module_parts = mod_path.split(".")
            parent = model
            for p in module_parts[:-1]:
                parent = getattr(parent, p)
            mod_name = module_parts[-1]
            orig = getattr(parent, mod_name)
            if isinstance(orig, nn.Linear):
                sparse_linear = SparseLinear(orig, flat_idx)
                setattr(parent, mod_name, sparse_linear)
                replacements += 1

    logger.info(f"[SIFT] Replaced {replacements} Linear modules with SparseLinear")
    return model


class TextDataset(TorchDataset):
    def __init__(self, instances, tokenizer, max_len):
        self.data = []
        for item in instances:
            text = item.get("text", "")
            enc = tokenizer(text, truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze()
            self.data.append({"input_ids": ids, "labels": ids.clone()})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def train_sift(
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
    sparse_modules=None,
    bf16=True,
    hf_token=None,
    seed=42,
    report_to="none",
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)

    if sparse_modules is None:
        sparse_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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
    sparse_rate = compute_sparse_rate(model, target_params=target_params, sparse_modules=sparse_modules)

    dataset_tag = os.path.basename(dataset_path).replace(".json", "")
    sift_obj = SIFT(
        model=model,
        sparse_rate=sparse_rate,
        sparse_module=sparse_modules,
        grad_acc=gradient_accumulation_steps,
        model_name=model_name_or_path,
        seed=seed,
        dataset_name=dataset_tag,
    )
    sift_obj.print_trainable_parameters()

    for param in model.parameters():
        param.requires_grad = False

    # Replace target Linear with SparseLinear
    model = replace_with_sparse_linear(model, sift_obj, sparse_modules)
    selection_elapsed = time.time() - selection_start

    # Verify trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"[SIFT] After replacement: trainable={trainable:,}, total={total:,}")
    logger.info(f"[SIFT] weight_selection_seconds={selection_elapsed:.2f}")

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

    # Restore SparseLinear → nn.Linear (clean model for vLLM/HF loading)
    model = restore_linear_modules(model)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[SIFT] Model saved to {output_dir}")
    return output_dir
