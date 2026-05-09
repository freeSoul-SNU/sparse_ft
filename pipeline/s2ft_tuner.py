"""S2FT integration based on the official structured-sparsity implementation.

The active trainable structures are the S2FT projections used by the reference
code: ``v_proj``/``o_proj`` for attention and ``up_proj``/``down_proj`` for FFN.
The base weights are frozen and only each S2 layer's ``s2`` parameter is
optimized.
"""
import copy
import json
import logging
import os
import random
import re
import sys
import threading
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)
RAPA_HOME = os.environ.get("RAPA_HOME", "/data/nksol0405/LLM/rapa")
SPARSE_FT_ROOT = os.environ.get(
    "SPARSE_FT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if SPARSE_FT_ROOT not in sys.path:
    sys.path.insert(0, SPARSE_FT_ROOT)

try:
    from methods.s2ft import S2ColumnLinear, S2RowLinear
except ImportError:
    from s2ft import S2ColumnLinear, S2RowLinear


def _deepspeed_config(default_path):
    value = os.environ.get("DS_CONFIG", default_path)
    if value.lower() in {"", "0", "false", "none", "no"}:
        return None
    return value


def _dataloader_num_workers():
    return int(os.environ.get("RAPA_DATALOADER_NUM_WORKERS", "0"))


def _read_self_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _read_self_pss_mb():
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


class _PhaseMemoryMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self._stop = threading.Event()
        self.peak_rss_mb = 0
        self.peak_pss_mb = 0
        self._thread = None

    def _sample(self):
        self.peak_rss_mb = max(self.peak_rss_mb, _read_self_rss_mb())
        self.peak_pss_mb = max(self.peak_pss_mb, _read_self_pss_mb())

    def start(self):
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def stop(self):
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self


class TextDataset(TorchDataset):
    def __init__(self, instances, tokenizer, max_len):
        self.data = []
        for item in instances:
            text = item.get("text", "")
            enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze()
            self.data.append({"input_ids": ids, "labels": ids.clone()})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def _move_batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _layer_id(name):
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _get_layers(model):
    try:
        return model.model.layers
    except AttributeError as exc:
        raise ValueError("S2FT currently expects a LLaMA-style model with model.layers") from exc


def _env_float(name):
    value = os.environ.get(name)
    if value is None or value.lower() in {"", "auto", "none", "null"}:
        return None
    return float(value)


def _ratio_or_env(value, env_name):
    return float(value) if value is not None else _env_float(env_name)


def _projection_set():
    raw = os.environ.get("S2FT_TARGET_PROJECTIONS", "o,d")
    aliases = {
        "v_proj": "v",
        "o_proj": "o",
        "up_proj": "u",
        "down_proj": "d",
    }
    projections = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        projections.append(aliases.get(key, key))
    return set(projections)


def _resolve_ratios(model, target_params, v_ratio, o_ratio, u_ratio, d_ratio):
    explicit = {
        "v": _ratio_or_env(v_ratio, "S2FT_V_RATIO"),
        "o": _ratio_or_env(o_ratio, "S2FT_O_RATIO"),
        "u": _ratio_or_env(u_ratio, "S2FT_U_RATIO"),
        "d": _ratio_or_env(d_ratio, "S2FT_D_RATIO"),
    }
    if any(value is not None for value in explicit.values()):
        return {name: max(0.0, min(float(value or 0.0), 1.0)) for name, value in explicit.items()}, "explicit"

    num_layers = len(_get_layers(model))
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    num_heads = int(getattr(model.config, "num_attention_heads", 0) or 0)
    intermediate_size = int(getattr(model.config, "intermediate_size", 0) or 0)
    head_dim = hidden_size // num_heads if num_heads else 0
    per_full_ratio = {
        "v": num_layers * num_heads * hidden_size * head_dim,
        "o": num_layers * num_heads * hidden_size * head_dim,
        "u": num_layers * intermediate_size * hidden_size,
        "d": num_layers * intermediate_size * hidden_size,
    }
    preset = os.environ.get("S2FT_RATIO_PRESET", "budget").lower()
    if preset in {"author", "paper", "official"}:
        # Official LLaMA2 S2FT scripts use v=0, o=0.052, u=0, d=0.02.
        # Scale that pattern when a fixed target parameter count is requested.
        base = {"v": 0.0, "o": 0.052, "u": 0.0, "d": 0.02}
        base_total = sum(per_full_ratio[name] * ratio for name, ratio in base.items())
        scale = min(float(target_params) / base_total, 1.0 / max(base.values())) if base_total else 0.0
        ratios = {name: max(0.0, min(base[name] * scale, 1.0)) for name in per_full_ratio}
        return ratios, "author_target_params"

    enabled = _projection_set()
    denom = sum(params for name, params in per_full_ratio.items() if name in enabled)
    ratio = min(float(target_params) / denom, 1.0) if denom else 0.0
    ratios = {name: ratio if name in enabled else 0.0 for name in per_full_ratio}
    return ratios, "target_params"


def _count_selected(total, ratio):
    return max(0, min(total, int(total * ratio)))


def _split_uniform_counts(total_count, num_layers, rng):
    if num_layers <= 0:
        return []
    base = total_count // num_layers
    remainder = total_count % num_layers
    counts = [base] * num_layers
    for layer in rng.sample(range(num_layers), remainder):
        counts[layer] += 1
    return counts


def _select_random_units(num_layers, units_per_layer, ratio, rng, allocation):
    selected = {layer: [] for layer in range(num_layers)}
    count = _count_selected(num_layers * units_per_layer, ratio)
    if count <= 0:
        return selected
    if allocation == "uniform":
        for layer, layer_count in enumerate(_split_uniform_counts(count, num_layers, rng)):
            if layer_count > 0:
                selected[layer] = sorted(rng.sample(range(units_per_layer), min(layer_count, units_per_layer)))
        return selected
    for flat_idx in sorted(rng.sample(range(num_layers * units_per_layer), count)):
        selected[flat_idx // units_per_layer].append(flat_idx % units_per_layer)
    return selected


def _choose_units(scores, count, method):
    selected = {layer: [] for layer in scores}
    if count <= 0 or not scores:
        return selected
    candidates = []
    for layer, score in scores.items():
        for idx, value in enumerate(score.tolist()):
            candidates.append((float(value), layer, idx))
    candidates.sort(reverse=method in {"large_activation", "large"}, key=lambda item: item[0])
    for _, layer, idx in candidates[: min(count, len(candidates))]:
        selected.setdefault(layer, []).append(idx)
    return selected


def _choose_units_uniform(scores, num_layers, units_per_layer, ratio, method, rng):
    selected = {layer: [] for layer in range(num_layers)}
    count = _count_selected(num_layers * units_per_layer, ratio)
    if count <= 0:
        return selected
    reverse = method in {"large_activation", "large"}
    for layer, layer_count in enumerate(_split_uniform_counts(count, num_layers, rng)):
        score = scores.get(layer)
        if layer_count <= 0 or score is None:
            continue
        candidates = [(float(value), idx) for idx, value in enumerate(score.tolist())]
        candidates.sort(reverse=reverse, key=lambda item: item[0])
        selected[layer] = [idx for _, idx in candidates[: min(layer_count, len(candidates))]]
    return selected


def _collect_s2ft_activation_scores(model, train_dataset, calibration_steps, calibration_batch_size):
    device = next(model.parameters()).device
    scores = {"v": {}, "o": {}, "u": {}, "d": {}}
    hooks = []
    num_heads = int(getattr(model.config, "num_attention_heads", 0) or 0)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    head_dim = hidden_size // num_heads if num_heads else 0

    def add_hook(name, module):
        layer = _layer_id(name)
        if layer is None:
            return

        def hook(_, inputs, output):
            with torch.no_grad():
                if name.endswith("v_proj") and num_heads and head_dim:
                    out = output.detach().float().abs().reshape(-1, num_heads, head_dim)
                    scores["v"][layer] = out.mean(dim=(0, 2)).cpu()
                elif name.endswith("o_proj") and num_heads and head_dim:
                    inp = inputs[0].detach().float().abs().reshape(-1, num_heads, head_dim)
                    scores["o"][layer] = inp.mean(dim=(0, 2)).cpu()
                elif name.endswith("up_proj"):
                    out = output.detach().float().abs().reshape(-1, output.shape[-1])
                    scores["u"][layer] = out.mean(dim=0).cpu()
                elif name.endswith("down_proj"):
                    inp = inputs[0].detach().float().abs().reshape(-1, inputs[0].shape[-1])
                    scores["d"][layer] = inp.mean(dim=0).cpu()

        hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(("v_proj", "o_proj", "up_proj", "down_proj")):
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
    return scores, completed_steps


def _select_units(model, train_dataset, ratios, method, calibration_steps, calibration_batch_size, seed):
    layers = _get_layers(model)
    num_layers = len(layers)
    num_heads = int(getattr(model.config, "num_attention_heads", 0) or 0)
    intermediate_size = int(getattr(model.config, "intermediate_size", 0) or 0)
    allocation = os.environ.get("S2FT_LAYER_ALLOCATION", "uniform").lower()
    rng = random.Random(seed)

    if method in {"small_activation", "activation", "large_activation", "large"}:
        scores, completed_steps = _collect_s2ft_activation_scores(
            model=model,
            train_dataset=train_dataset,
            calibration_steps=calibration_steps,
            calibration_batch_size=calibration_batch_size,
        )
        if allocation == "uniform":
            selected = {
                "v": _choose_units_uniform(scores["v"], num_layers, num_heads, ratios["v"], method, rng),
                "o": _choose_units_uniform(scores["o"], num_layers, num_heads, ratios["o"], method, rng),
                "u": _choose_units_uniform(scores["u"], num_layers, intermediate_size, ratios["u"], method, rng),
                "d": _choose_units_uniform(scores["d"], num_layers, intermediate_size, ratios["d"], method, rng),
            }
        else:
            selected = {
                "v": _choose_units(scores["v"], _count_selected(num_layers * num_heads, ratios["v"]), method),
                "o": _choose_units(scores["o"], _count_selected(num_layers * num_heads, ratios["o"]), method),
                "u": _choose_units(scores["u"], _count_selected(num_layers * intermediate_size, ratios["u"]), method),
                "d": _choose_units(scores["d"], _count_selected(num_layers * intermediate_size, ratios["d"]), method),
            }
        return selected, completed_steps

    selected = {
        "v": _select_random_units(num_layers, num_heads, ratios["v"], rng, allocation),
        "o": _select_random_units(num_layers, num_heads, ratios["o"], rng, allocation),
        "u": _select_random_units(num_layers, intermediate_size, ratios["u"], rng, allocation),
        "d": _select_random_units(num_layers, intermediate_size, ratios["d"], rng, allocation),
    }
    return selected, 0


def _replace_with_s2_column(module, start, end):
    checkpoint = copy.deepcopy(module.state_dict())
    replacement = S2ColumnLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        start=start,
        end=end,
        device=next(module.parameters()).device,
        dtype=next(module.parameters()).dtype,
    )
    replacement.load_state_dict(checkpoint, strict=False)
    return replacement


def _replace_with_s2_row(module, start, end):
    checkpoint = copy.deepcopy(module.state_dict())
    replacement = S2RowLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        start=start,
        end=end,
        device=next(module.parameters()).device,
        dtype=next(module.parameters()).dtype,
    )
    replacement.load_state_dict(checkpoint, strict=False)
    return replacement


def _reorder_rows_by_units(linear, order, unit_size=1):
    weight = linear.weight.data
    expected = len(order) * unit_size
    if weight.shape[0] != expected:
        logger.warning("[S2FT] skipping row reorder for shape=%s expected_rows=%s", tuple(weight.shape), expected)
        return
    weight = weight.reshape(len(order), unit_size, weight.shape[-1])
    linear.weight.data = weight[order, :, :].reshape(-1, weight.shape[-1])
    if linear.bias is not None:
        bias = linear.bias.data.reshape(len(order), unit_size)
        linear.bias.data = bias[order, :].reshape(-1)


def _reorder_columns_by_units(linear, order, unit_size=1):
    weight = linear.weight.data
    expected = len(order) * unit_size
    if weight.shape[1] != expected:
        logger.warning("[S2FT] skipping column reorder for shape=%s expected_cols=%s", tuple(weight.shape), expected)
        return
    weight = weight.reshape(weight.shape[0], len(order), unit_size)
    linear.weight.data = weight[:, order, :].reshape(weight.shape[0], -1)


def _convert_mha_layer_to_s2(model, selected):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    replacements = 0
    for layer_idx, layer in enumerate(_get_layers(model)):
        selected_v = set(selected["v"].get(layer_idx, []))
        selected_o = set(selected["o"].get(layer_idx, []))
        only_v = sorted(selected_v - selected_o)
        only_o = sorted(selected_o - selected_v)
        vo = sorted(selected_v & selected_o)
        order = only_v + vo + only_o
        order.extend(head for head in range(model.config.num_attention_heads) if head not in order)

        if only_v or vo:
            layer.self_attn.v_proj = _replace_with_s2_column(
                layer.self_attn.v_proj,
                start=0,
                end=(len(only_v) + len(vo)) * head_dim,
            )
            replacements += 1
        if only_o or vo:
            layer.self_attn.o_proj = _replace_with_s2_row(
                layer.self_attn.o_proj,
                start=len(only_v) * head_dim,
                end=(len(only_v) + len(vo) + len(only_o)) * head_dim,
            )
            replacements += 1

        _reorder_rows_by_units(layer.self_attn.q_proj, order, head_dim)
        _reorder_rows_by_units(layer.self_attn.k_proj, order, head_dim)
        _reorder_rows_by_units(layer.self_attn.v_proj, order, head_dim)
        _reorder_columns_by_units(layer.self_attn.o_proj, order, head_dim)
    return replacements


def _convert_ffn_layer_to_s2(model, selected):
    replacements = 0
    for layer_idx, layer in enumerate(_get_layers(model)):
        selected_u = set(selected["u"].get(layer_idx, []))
        selected_d = set(selected["d"].get(layer_idx, []))
        only_u = sorted(selected_u - selected_d)
        only_d = sorted(selected_d - selected_u)
        ud = sorted(selected_u & selected_d)
        order = only_u + ud + only_d
        order.extend(channel for channel in range(model.config.intermediate_size) if channel not in order)

        if only_u or ud:
            layer.mlp.up_proj = _replace_with_s2_column(
                layer.mlp.up_proj,
                start=0,
                end=len(only_u) + len(ud),
            )
            replacements += 1
        if only_d or ud:
            layer.mlp.down_proj = _replace_with_s2_row(
                layer.mlp.down_proj,
                start=len(only_u),
                end=len(only_u) + len(ud) + len(only_d),
            )
            replacements += 1

        _reorder_rows_by_units(layer.mlp.up_proj, order)
        _reorder_rows_by_units(layer.mlp.gate_proj, order)
        _reorder_columns_by_units(layer.mlp.down_proj, order)
    return replacements


def _only_optimize_s2_parameters(model):
    for name, param in model.named_parameters():
        param.requires_grad = "s2" in name
    return model


def _restore_s2_linear_modules(model):
    for name, module in list(model.named_modules()):
        if not isinstance(module, (S2ColumnLinear, S2RowLinear)):
            continue
        module.fuse_s2_weight()
        new_linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        new_linear.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            new_linear.bias.data.copy_(module.bias.data)
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_linear)
    return model


def train_s2ft(
    model_name_or_path, dataset_path, output_dir, num_train_epochs=1, max_steps=-1,
    per_device_train_batch_size=1, gradient_accumulation_steps=1, learning_rate=5e-5,
    lr_scheduler_type="linear", max_seq_length=512, target_params=170_000_000,
    bf16=True, hf_token=None, seed=42, report_to="none",
    s2ft_calibration_steps=None, s2ft_calibration_batch_size=None,
    s2ft_v_ratio=None, s2ft_o_ratio=None, s2ft_u_ratio=None, s2ft_d_ratio=None,
    s2ft_selection_method=None,
    **kwargs,
):
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

    with open(dataset_path) as f:
        raw = json.load(f)
    train_dataset = TextDataset(raw.get("instances", []), tokenizer, max_seq_length)

    selection_start = time.time()
    if s2ft_calibration_steps is None:
        s2ft_calibration_steps = int(os.environ.get("S2FT_CALIBRATION_STEPS", "100"))
    if s2ft_calibration_batch_size is None:
        s2ft_calibration_batch_size = int(os.environ.get("S2FT_CALIBRATION_BATCH_SIZE", "1"))
    selection_method = s2ft_selection_method or os.environ.get("S2FT_SELECTION_METHOD", "random")
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    ratios, ratio_source = _resolve_ratios(
        model=model,
        target_params=target_params,
        v_ratio=s2ft_v_ratio,
        o_ratio=s2ft_o_ratio,
        u_ratio=s2ft_u_ratio,
        d_ratio=s2ft_d_ratio,
    )
    logger.info(
        "[S2FT] selection_method=%s, ratio_source=%s, v_ratio=%.6f, o_ratio=%.6f, u_ratio=%.6f, d_ratio=%.6f",
        selection_method,
        ratio_source,
        ratios["v"],
        ratios["o"],
        ratios["u"],
        ratios["d"],
    )

    selected, completed_calibration_steps = _select_units(
        model=model,
        train_dataset=train_dataset,
        ratios=ratios,
        method=selection_method,
        calibration_steps=s2ft_calibration_steps,
        calibration_batch_size=s2ft_calibration_batch_size,
        seed=seed,
    )
    replacements = _convert_mha_layer_to_s2(model, selected)
    replacements += _convert_ffn_layer_to_s2(model, selected)
    _only_optimize_s2_parameters(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "[S2FT] replacements=%d, selected_v=%d, selected_o=%d, selected_u=%d, selected_d=%d, "
        "trainable=%d, completed_calibration_steps=%d",
        replacements,
        sum(len(v) for v in selected["v"].values()),
        sum(len(v) for v in selected["o"].values()),
        sum(len(v) for v in selected["u"].values()),
        sum(len(v) for v in selected["d"].values()),
        trainable,
        completed_calibration_steps,
    )
    logger.info(f"[S2FT] weight_selection_seconds={time.time() - selection_start:.2f}")

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
        dataloader_num_workers=_dataloader_num_workers(),
        remove_unused_columns=False,
        deepspeed=_deepspeed_config(os.path.join(RAPA_HOME, "LMFlow", "configs", "rapa", "ds_zero1_sift.json")),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer)
    resume_training = os.environ.get("RESUME_TRAINING", "false").lower() in {"1", "true", "yes"}
    existing_checkpoint = get_last_checkpoint(output_dir)
    resume_checkpoint = existing_checkpoint if resume_training else None
    if existing_checkpoint and not resume_training:
        logger.warning(f"[S2FT] Ignoring existing checkpoint because RESUME_TRAINING=false: {existing_checkpoint}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    train_monitor = _PhaseMemoryMonitor().start()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_monitor.stop()
    train_seconds = time.time() - train_start
    peak_allocated_mb = 0
    peak_reserved_mb = 0
    if torch.cuda.is_available():
        peak_allocated_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
        peak_reserved_mb = torch.cuda.max_memory_reserved() // (1024 * 1024)
    logger.info(f"[S2FT] train_wall_seconds={train_seconds:.2f}")
    logger.info(f"[S2FT] train_peak_gpu_allocated_mb={peak_allocated_mb}")
    logger.info(f"[S2FT] train_peak_gpu_reserved_mb={peak_reserved_mb}")
    logger.info(f"[S2FT] train_peak_cpu_pss_mb={train_monitor.peak_pss_mb}")
    logger.info(f"[S2FT] train_peak_cpu_rss_mb={train_monitor.peak_rss_mb}")

    if os.environ.get("RAPA_SKIP_SAVE", "false").lower() in {"1", "true", "yes"}:
        logger.info("[S2FT] RAPA_SKIP_SAVE=true; skipping model save for profiling")
        return output_dir

    model = _restore_s2_linear_modules(model)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[S2FT] Model saved to {output_dir}")
    return output_dir
