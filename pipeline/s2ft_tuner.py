"""S2FT (Structured Sparse Fine-Tuning) with head/channel selection.

S2FT selects sparse coupled structures, not arbitrary random weights: attention
heads in MHA and intermediate channels in FFN.  By default this implementation
uses the paper's small-activation criterion, then updates the corresponding
dense substructures through sparse deltas.
"""
import logging
import os
import json
import re
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)
RAPA_HOME = os.environ.get("RAPA_HOME", "/data/nksol0405/LLM/rapa")


def _deepspeed_config(default_path):
    value = os.environ.get("DS_CONFIG", default_path)
    if value.lower() in {"", "0", "false", "none", "no"}:
        return None
    return value


try:
    from lmflow.pipeline.rapa.sift_tuner import SparseLinear
except ImportError:
    try:
        from .sift_tuner import SparseLinear
    except ImportError:
        from sift_tuner import SparseLinear


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


def _move_batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _layer_id(name):
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _flat_indices_for_rows(module, rows):
    rows = sorted(set(int(r) for r in rows if 0 <= int(r) < module.out_features))
    return torch.tensor(
        [r * module.in_features + c for r in rows for c in range(module.in_features)],
        dtype=torch.long,
    )


def _flat_indices_for_columns(module, cols):
    cols = sorted(set(int(c) for c in cols if 0 <= int(c) < module.in_features))
    return torch.tensor(
        [r * module.in_features + c for r in range(module.out_features) for c in cols],
        dtype=torch.long,
    )


def _replace_linear_with_indices(model, name_to_indices):
    replacements = 0
    for name, flat_idx in name_to_indices.items():
        if flat_idx.numel() == 0:
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        module = getattr(parent, parts[-1])
        if isinstance(module, nn.Linear):
            setattr(parent, parts[-1], SparseLinear(module, flat_idx))
            replacements += 1
    return replacements


def _collect_s2ft_activation_scores(model, train_dataset, calibration_steps, calibration_batch_size):
    device = next(model.parameters()).device
    attn_scores = {}
    ffn_scores = {}
    hooks = []
    num_heads = int(getattr(model.config, "num_attention_heads", 0) or 0)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    head_dim = hidden_size // num_heads if num_heads else 0

    def add_hook(name, module):
        layer = _layer_id(name)
        if layer is None:
            return

        def hook(_, __, output):
            with torch.no_grad():
                out = output.detach().float().abs()
                if name.endswith("q_proj") and num_heads and head_dim:
                    score = out.reshape(-1, num_heads, head_dim).mean(dim=(0, 2)).cpu()
                    attn_scores[layer] = attn_scores.get(layer, torch.zeros_like(score)) + score
                elif name.endswith("gate_proj") or name.endswith("up_proj"):
                    score = out.reshape(-1, out.shape[-1]).mean(dim=0).cpu()
                    ffn_scores[layer] = ffn_scores.get(layer, torch.zeros_like(score)) + score

        hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (
            name.endswith("q_proj") or name.endswith("gate_proj") or name.endswith("up_proj")
        ):
            add_hook(name, module)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.eval()
    loader = DataLoader(train_dataset, batch_size=calibration_batch_size, shuffle=False, num_workers=0)
    completed_steps = 0
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if step > calibration_steps:
                break
            model(**_move_batch_to_device(batch, device))
            completed_steps = step
            if step == 1 or step % 10 == 0:
                logger.info(f"[S2FT] activation_calibration_progress={step}/{calibration_steps}")

    for hook in hooks:
        hook.remove()
    return attn_scores, ffn_scores, completed_steps


def _choose_units(scores, count, method):
    if count <= 0 or not scores:
        return {}
    candidates = []
    for layer, score in scores.items():
        for idx, value in enumerate(score.tolist()):
            candidates.append((float(value), layer, idx))
    reverse = method in {"large_activation", "large"}
    candidates.sort(reverse=reverse, key=lambda item: item[0])
    selected = {}
    for _, layer, idx in candidates[: min(count, len(candidates))]:
        selected.setdefault(layer, []).append(idx)
    return selected


def _build_s2ft_indices(model, selected_heads, selected_channels):
    modules = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    num_heads = int(getattr(model.config, "num_attention_heads", 0) or 0)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    head_dim = hidden_size // num_heads if num_heads else 0
    name_to_indices = {}

    for name, module in modules.items():
        layer = _layer_id(name)
        if layer is None:
            continue
        if any(name.endswith(suffix) for suffix in ("q_proj", "k_proj", "v_proj")):
            rows = []
            for head in selected_heads.get(layer, []):
                rows.extend(range(head * head_dim, (head + 1) * head_dim))
            if rows:
                name_to_indices[name] = _flat_indices_for_rows(module, rows)
        elif name.endswith("o_proj"):
            cols = []
            for head in selected_heads.get(layer, []):
                cols.extend(range(head * head_dim, (head + 1) * head_dim))
            if cols:
                name_to_indices[name] = _flat_indices_for_columns(module, cols)
        elif name.endswith("gate_proj") or name.endswith("up_proj"):
            rows = selected_channels.get(layer, [])
            if rows:
                name_to_indices[name] = _flat_indices_for_rows(module, rows)
        elif name.endswith("down_proj"):
            cols = selected_channels.get(layer, [])
            if cols:
                name_to_indices[name] = _flat_indices_for_columns(module, cols)

    return name_to_indices


def train_s2ft(
    model_name_or_path, dataset_path, output_dir, num_train_epochs=1, max_steps=-1,
    per_device_train_batch_size=1, gradient_accumulation_steps=1, learning_rate=5e-5,
    lr_scheduler_type="linear", max_seq_length=512, target_params=170_000_000,
    bf16=True, hf_token=None, seed=42, report_to="none",
    s2ft_calibration_steps=None, s2ft_calibration_batch_size=None,
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    tok_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16 if bf16 else torch.float32, **tok_kwargs)

    with open(dataset_path) as f:
        raw = json.load(f)
    train_dataset = TextDataset(raw.get("instances", []), tokenizer, max_seq_length)

    selection_start = time.time()
    if s2ft_calibration_steps is None:
        s2ft_calibration_steps = int(os.environ.get("S2FT_CALIBRATION_STEPS", "100"))
    if s2ft_calibration_batch_size is None:
        s2ft_calibration_batch_size = int(os.environ.get("S2FT_CALIBRATION_BATCH_SIZE", "1"))
    selection_method = os.environ.get("S2FT_SELECTION_METHOD", "small_activation")
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    num_layers = len({layer for layer in (_layer_id(n) for n, _ in model.named_modules()) if layer is not None})
    num_heads = int(getattr(model.config, "num_attention_heads", 0) or 0)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    head_dim = hidden_size // num_heads if num_heads else 0
    intermediate_size = int(getattr(model.config, "intermediate_size", 0) or 0)
    attn_unit_params = 4 * hidden_size * head_dim if head_dim else 0
    ffn_unit_params = 3 * hidden_size
    possible_params = num_layers * num_heads * attn_unit_params + num_layers * intermediate_size * ffn_unit_params
    rate = min(target_params / possible_params, 1.0) if possible_params else 0.0
    num_selected_heads = int(num_layers * num_heads * rate)
    num_selected_channels = int(num_layers * intermediate_size * rate)
    logger.info(
        f"[S2FT] selection_method={selection_method}, heads={num_selected_heads}, "
        f"channels={num_selected_channels}, rate={rate:.6f}"
    )

    attn_scores, ffn_scores, completed_calibration_steps = _collect_s2ft_activation_scores(
        model=model,
        train_dataset=train_dataset,
        calibration_steps=s2ft_calibration_steps,
        calibration_batch_size=s2ft_calibration_batch_size,
    )
    selected_heads = _choose_units(attn_scores, num_selected_heads, selection_method)
    selected_channels = _choose_units(ffn_scores, num_selected_channels, selection_method)

    for param in model.parameters():
        param.requires_grad = False

    name_to_indices = _build_s2ft_indices(model, selected_heads, selected_channels)
    replacements = _replace_linear_with_indices(model, name_to_indices)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[S2FT] replacements={replacements}, selected_heads={sum(len(v) for v in selected_heads.values())}, "
        f"selected_channels={sum(len(v) for v in selected_channels.values())}, trainable={trainable:,}, "
        f"completed_calibration_steps={completed_calibration_steps}"
    )
    logger.info(f"[S2FT] weight_selection_seconds={time.time() - selection_start:.2f}")

    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=num_train_epochs, max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate, lr_scheduler_type=lr_scheduler_type, bf16=bf16,
        save_strategy="no" if 0 < max_steps < 100 else "epoch", logging_steps=5,
        report_to=report_to, seed=seed, dataloader_num_workers=4, remove_unused_columns=False,
        deepspeed=_deepspeed_config(os.path.join(RAPA_HOME, "LMFlow", "configs", "rapa", "ds_zero1_sift.json")),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer)
    trainer.train(resume_from_checkpoint=get_last_checkpoint(output_dir))

    if os.environ.get("RAPA_SKIP_SAVE", "false").lower() in {"1", "true", "yes"}:
        logger.info("[S2FT] RAPA_SKIP_SAVE=true; skipping model save for profiling")
        return output_dir

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
    logger.info(f"[S2FT] Model saved to {output_dir}")
    return output_dir
