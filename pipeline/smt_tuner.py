"""SMT (Sparse Matrix Tuning) with gradient-calibrated block selection.

Uses the same buffer + sparse delta approach as SIFT for DeepSpeed compatibility.
Before fine-tuning, SMT runs a short calibration pass, scores 256x256 blocks by
their accumulated weight-gradient magnitude, then trains only the selected blocks.
"""
import heapq
import logging
import os
import sys
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)
RAPA_HOME = os.environ.get("RAPA_HOME", "/home/mms/freeSoul/llm/rapa")

BLOCK_DIM = 256  # SMT block dimension


class BlockSparseLinear(nn.Module):
    """Linear with block-sparse trainable delta (SMT-style)."""

    def __init__(self, orig_linear: nn.Linear, selected_blocks):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features

        self.register_buffer("weight", orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.register_buffer("bias", orig_linear.bias.data)
        else:
            self.bias = None

        flat_indices = []
        for rb, cb in selected_blocks:
            r_start = rb * BLOCK_DIM
            c_start = cb * BLOCK_DIM
            for r in range(r_start, min(r_start + BLOCK_DIM, self.out_features)):
                for c in range(c_start, min(c_start + BLOCK_DIM, self.in_features)):
                    flat_indices.append(r * self.in_features + c)

        flat_idx = torch.tensor(flat_indices, dtype=torch.long, device=self.weight.device)
        self.register_buffer("flat_idx", flat_idx)
        self.sparse_delta = nn.Parameter(
            torch.zeros(len(flat_idx), dtype=self.weight.dtype, device=self.weight.device),
            requires_grad=True,
        )

    def forward(self, x):
        delta_flat = torch.zeros(
            self.weight.numel(), dtype=self.sparse_delta.dtype, device=self.sparse_delta.device
        )
        delta_flat.scatter_(0, self.flat_idx, self.sparse_delta)
        delta = delta_flat.view(self.weight.shape)
        return F.linear(x, self.weight + delta, self.bias)


def find_target_linear_layers(model, target_modules):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules)
    }


def compute_total_target_blocks(layers, target_params=170_000_000):
    """Compute total number of BLOCK_DIM x BLOCK_DIM blocks needed to hit target_params."""
    params_per_block = BLOCK_DIM * BLOCK_DIM
    available_blocks = 0
    for module in layers.values():
        available_blocks += (module.out_features // BLOCK_DIM) * (module.in_features // BLOCK_DIM)

    requested_blocks = max(1, target_params // params_per_block)
    total_blocks = min(requested_blocks, available_blocks)
    logger.info(
        f"[SMT] target_layers={len(layers)}, selected_blocks={total_blocks:,}, "
        f"available_blocks={available_blocks:,}, ~trainable_params={total_blocks * params_per_block:,}"
    )
    return total_blocks


def compute_blocks_per_layer(model, target_params=170_000_000, target_modules=None):
    """Backward-compatible estimate kept for old callers; SMT now selects blocks globally."""
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


def _move_batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def accumulate_gradient_block_scores(
    model,
    layers,
    train_dataset,
    calibration_steps=100,
    calibration_batch_size=1,
):
    """Score each candidate block by mean absolute gradient over calibration batches."""
    device = next(model.parameters()).device
    scores = {}
    for name, module in layers.items():
        row_blocks = module.out_features // BLOCK_DIM
        col_blocks = module.in_features // BLOCK_DIM
        if row_blocks > 0 and col_blocks > 0:
            scores[name] = torch.zeros((row_blocks, col_blocks), dtype=torch.float32)

    if not scores:
        return scores, 0

    for param in model.parameters():
        param.requires_grad = False
        param.grad = None
    for name in scores:
        layers[name].weight.requires_grad = True

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.train()

    loader = DataLoader(
        train_dataset,
        batch_size=calibration_batch_size,
        shuffle=False,
        num_workers=0,
    )

    completed_steps = 0
    for step, batch in enumerate(loader, start=1):
        if step > calibration_steps:
            break

        model.zero_grad(set_to_none=True)
        outputs = model(**_move_batch_to_device(batch, device))
        outputs.loss.backward()

        for name, module in layers.items():
            grad = module.weight.grad
            if grad is None or name not in scores:
                continue

            row_blocks, col_blocks = scores[name].shape
            trimmed = grad[: row_blocks * BLOCK_DIM, : col_blocks * BLOCK_DIM]
            block_scores = (
                trimmed.detach()
                .abs()
                .reshape(row_blocks, BLOCK_DIM, col_blocks, BLOCK_DIM)
                .mean(dim=(1, 3))
                .float()
                .cpu()
            )
            scores[name].add_(block_scores)
            module.weight.grad = None

        completed_steps = step
        if step == 1 or step % 10 == 0:
            logger.info(f"[SMT] calibration_progress={step}/{calibration_steps}")

    model.zero_grad(set_to_none=True)
    for name in scores:
        layers[name].weight.requires_grad = False

    return scores, completed_steps


def select_top_gradient_blocks(scores, total_blocks):
    """Select global top-k blocks across all target layers."""
    heap = []
    for name, score in scores.items():
        flat_scores = score.reshape(-1)
        col_blocks = score.shape[1]
        k = min(total_blocks, flat_scores.numel())
        if k <= 0:
            continue

        values, indices = torch.topk(flat_scores, k=k)
        for value, idx in zip(values.tolist(), indices.tolist()):
            item = (float(value), name, int(idx))
            if len(heap) < total_blocks:
                heapq.heappush(heap, item)
            elif value > heap[0][0]:
                heapq.heapreplace(heap, item)

    selected = {}
    for _, name, idx in heap:
        col_blocks = scores[name].shape[1]
        rb = idx // col_blocks
        cb = idx % col_blocks
        selected.setdefault(name, []).append((rb, cb))
    return selected


def replace_with_block_sparse_linear(model, selected_blocks):
    replacements = 0
    for name, blocks in selected_blocks.items():
        if not blocks:
            continue

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        module = getattr(parent, attr)
        if isinstance(module, nn.Linear):
            setattr(parent, attr, BlockSparseLinear(module, blocks))
            replacements += 1
    return replacements


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
    smt_calibration_steps=None,
    smt_calibration_batch_size=None,
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    target_modules = [
        item.strip()
        for item in os.environ.get("SMT_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj").split(",")
        if item.strip()
    ]
    if smt_calibration_steps is None:
        smt_calibration_steps = int(os.environ.get("SMT_CALIBRATION_STEPS", "100"))
    if smt_calibration_batch_size is None:
        smt_calibration_batch_size = int(os.environ.get("SMT_CALIBRATION_BATCH_SIZE", "1"))

    tok_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        **tok_kwargs,
    )

    with open(dataset_path) as f:
        raw = json.load(f)
    train_dataset = TextDataset(raw.get("instances", []), tokenizer, max_seq_length)

    selection_start = time.time()
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    target_layers = find_target_linear_layers(model, target_modules)
    total_selected_blocks = compute_total_target_blocks(target_layers, target_params)

    calibration_start = time.time()
    logger.info(
        f"[SMT] Starting gradient calibration: steps={smt_calibration_steps}, "
        f"batch_size={smt_calibration_batch_size}, target_modules={target_modules}"
    )
    gradient_scores, completed_calibration_steps = accumulate_gradient_block_scores(
        model=model,
        layers=target_layers,
        train_dataset=train_dataset,
        calibration_steps=smt_calibration_steps,
        calibration_batch_size=smt_calibration_batch_size,
    )
    calibration_seconds = time.time() - calibration_start
    logger.info(
        f"[SMT] calibration_seconds={calibration_seconds:.2f}, "
        f"completed_steps={completed_calibration_steps}"
    )
    selected_blocks = select_top_gradient_blocks(gradient_scores, total_selected_blocks)

    for param in model.parameters():
        param.requires_grad = False

    replacements = replace_with_block_sparse_linear(model, selected_blocks)
    selected_block_count = sum(len(blocks) for blocks in selected_blocks.values())

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[SMT] Replaced {replacements} layers, selected_blocks={selected_block_count:,}, "
        f"trainable={trainable:,}, total params={total:,}"
    )
    logger.info(f"[SMT] weight_selection_seconds={time.time() - selection_start:.2f}")

    with open(os.path.join(output_dir, "smt_selection_meta.json"), "w") as f:
        json.dump(
            {
                "target_modules": target_modules,
                "target_params": target_params,
                "block_dim": BLOCK_DIM,
                "requested_calibration_steps": smt_calibration_steps,
                "completed_calibration_steps": completed_calibration_steps,
                "calibration_batch_size": smt_calibration_batch_size,
                "calibration_seconds": calibration_seconds,
                "selected_blocks": selected_block_count,
                "selected_layers": len(selected_blocks),
                "trainable_params": trainable,
            },
            f,
            indent=2,
        )

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
    try:
        from lmflow.pipeline.rapa.sift_tuner import restore_linear_modules
    except ImportError:
        try:
            from .sift_tuner import restore_linear_modules
        except ImportError:
            from sift_tuner import restore_linear_modules
    model = restore_linear_modules(model)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[SMT] Model saved to {output_dir}")
    return output_dir
