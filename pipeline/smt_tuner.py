"""SMT (Sparse Matrix Tuning) with author-code-style block selection.

SMT first runs a short dense warmup/calibration phase, accumulates gradients for
candidate linear weights, selects 256x256 submatrices, and fine-tunes only those
selected submatrices.  The default path follows the official PEFT example:
attention-only q/k/v blocks, no-restriction global top-k within the attention
group, and the author's mean(abs-after-block-mean) score.
"""
import heapq
import logging
import os
import sys
import json
import re
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)
RAPA_HOME = os.environ.get("RAPA_HOME", "/data/nksol0405/LLM/rapa")

BLOCK_DIM = 256  # SMT block dimension

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


class BlockSparseLinear(nn.Module):
    """Author-style SMT Linear: optimize selected 256x256 weight blocks only."""

    def __init__(self, orig_linear: nn.Linear, selected_blocks):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.weight = orig_linear.weight
        self.weight.requires_grad = False
        if orig_linear.bias is not None:
            self.bias = orig_linear.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        self.index_list = [(int(rb), int(cb)) for rb, cb in selected_blocks]
        selected_weight = torch.empty(
            len(self.index_list) * BLOCK_DIM,
            BLOCK_DIM,
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        for i, (rb, cb) in enumerate(self.index_list):
            selected_weight[i * BLOCK_DIM:(i + 1) * BLOCK_DIM, :] = self.weight.data[
                rb * BLOCK_DIM:(rb + 1) * BLOCK_DIM,
                cb * BLOCK_DIM:(cb + 1) * BLOCK_DIM,
            ]
        self.selected_weight = nn.Parameter(selected_weight, requires_grad=True)

    def _merge_selected_weight_(self):
        with torch.no_grad():
            for i, (rb, cb) in enumerate(self.index_list):
                self.weight.data[
                    rb * BLOCK_DIM:(rb + 1) * BLOCK_DIM,
                    cb * BLOCK_DIM:(cb + 1) * BLOCK_DIM,
                ] = self.selected_weight.data[i * BLOCK_DIM:(i + 1) * BLOCK_DIM, :]

    def forward(self, x):
        self._merge_selected_weight_()
        out = _SMTLinearFn.apply(x, self.selected_weight, self.index_list, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def merge_and_get_weight(self):
        self._merge_selected_weight_()
        return self.weight.data


class _SMTLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, selected_weight, index_list, weight):
        input_blocks = [
            input_tensor[:, :, cb * BLOCK_DIM:(cb + 1) * BLOCK_DIM]
            for _, cb in index_list
        ]
        ctx.input_blocks = input_blocks
        ctx.index_list = index_list
        ctx.save_for_backward(weight)
        return torch.matmul(input_tensor, weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        grad_weight = torch.empty(
            len(ctx.input_blocks) * BLOCK_DIM,
            BLOCK_DIM,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        grad_output_t = grad_output.permute(0, 2, 1)
        for i, ((rb, _), input_block) in enumerate(zip(ctx.index_list, ctx.input_blocks)):
            grad_weight[i * BLOCK_DIM:(i + 1) * BLOCK_DIM, :] = torch.sum(
                torch.matmul(
                    grad_output_t[:, rb * BLOCK_DIM:(rb + 1) * BLOCK_DIM, :],
                    input_block,
                ),
                dim=0,
            )
        grad_input = torch.matmul(grad_output, weight)
        return grad_input, grad_weight, None, None


def find_target_linear_layers(model, target_modules):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules)
    }


def _parse_csv_env(name, default):
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


def _env_int(name):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _dataloader_num_workers():
    return int(os.environ.get("RAPA_DATALOADER_NUM_WORKERS", "0"))


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _module_short_name(name):
    return name.rsplit(".", 1)[-1]


def _layer_number(name):
    match = _LAYER_RE.search(f".{name}.")
    return int(match.group(1)) if match else -1


def split_smt_candidate_layers(model, attention_modules, mlp_modules):
    attention_layers = {}
    mlp_layers = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short = _module_short_name(name)
        if "self_attn" in name and short in attention_modules:
            attention_layers[name] = module
        elif "mlp" in name and short in mlp_modules:
            mlp_layers[name] = module
    return attention_layers, mlp_layers


def _available_blocks(layers):
    return sum(
        (module.out_features // BLOCK_DIM) * (module.in_features // BLOCK_DIM)
        for module in layers.values()
    )


def compute_smt_block_budgets(attention_layers, mlp_layers, target_params):
    """Derive official num_submatrix_attn/mlp-style budgets from target_params."""
    params_per_block = BLOCK_DIM * BLOCK_DIM
    requested_total = max(1, target_params // params_per_block)
    available_attn = _available_blocks(attention_layers)
    available_mlp = _available_blocks(mlp_layers)

    explicit_attn = _env_int("SMT_NUM_SUBMATRIX_ATTN")
    explicit_mlp = _env_int("SMT_NUM_SUBMATRIX_MLP")
    allocation = os.environ.get("SMT_BUDGET_ALLOCATION", "attention_only").lower()

    if explicit_attn is not None or explicit_mlp is not None:
        attn_blocks = explicit_attn if explicit_attn is not None else max(0, requested_total - (explicit_mlp or 0))
        mlp_blocks = explicit_mlp if explicit_mlp is not None else max(0, requested_total - attn_blocks)
    elif allocation in {"attention_only", "attn_only", "official_peft"}:
        attn_blocks, mlp_blocks = requested_total, 0
    elif allocation == "mlp_only":
        attn_blocks, mlp_blocks = 0, requested_total
    elif allocation == "equal":
        attn_blocks = requested_total // 2
        mlp_blocks = requested_total - attn_blocks
    elif allocation == "capacity":
        total_available = max(1, available_attn + available_mlp)
        attn_blocks = round(requested_total * available_attn / total_available)
        mlp_blocks = requested_total - attn_blocks
    else:
        logger.warning("[SMT] Unknown SMT_BUDGET_ALLOCATION=%s; using attention_only", allocation)
        attn_blocks, mlp_blocks = requested_total, 0

    attn_blocks = min(max(0, attn_blocks), available_attn)
    mlp_blocks = min(max(0, mlp_blocks), available_mlp)
    selected_total = attn_blocks + mlp_blocks
    logger.info(
        "[SMT] requested_blocks=%s, attn_blocks=%s/%s, mlp_blocks=%s/%s, "
        "trainable_params=%s, allocation=%s",
        requested_total,
        attn_blocks,
        available_attn,
        mlp_blocks,
        available_mlp,
        selected_total * params_per_block,
        allocation,
    )
    return attn_blocks, mlp_blocks


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


def _move_batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def accumulate_gradient_tensors(
    model,
    layers,
    train_dataset,
    calibration_steps=100,
    calibration_batch_size=1,
    data_collator=None,
):
    """Accumulate raw dense gradients like the official SMT warmup trainer."""
    device = next(model.parameters()).device
    grads = {}
    for name, module in layers.items():
        row_blocks = module.out_features // BLOCK_DIM
        col_blocks = module.in_features // BLOCK_DIM
        if row_blocks > 0 and col_blocks > 0:
            grads[name] = torch.zeros(
                (row_blocks * BLOCK_DIM, col_blocks * BLOCK_DIM),
                dtype=torch.float32,
            )

    if not grads:
        return grads, 0

    for param in model.parameters():
        param.requires_grad = False
        param.grad = None
    for name in grads:
        layers[name].weight.requires_grad = True

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.train()

    loader = DataLoader(
        train_dataset,
        batch_size=calibration_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=data_collator,
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
            if grad is None or name not in grads:
                continue

            row_blocks = module.out_features // BLOCK_DIM
            col_blocks = module.in_features // BLOCK_DIM
            trimmed = grad[: row_blocks * BLOCK_DIM, : col_blocks * BLOCK_DIM]
            grads[name].add_(trimmed.detach().float().cpu())
            module.weight.grad = None

        completed_steps = step
        if step == 1 or step % 10 == 0:
            logger.info(f"[SMT] calibration_progress={step}/{calibration_steps}")

    model.zero_grad(set_to_none=True)
    for name in grads:
        layers[name].weight.requires_grad = False

    return grads, completed_steps


def _block_scores_from_grads(grads, calculation_strategy):
    scores = {}
    for name, grad in grads.items():
        row_blocks = grad.shape[0] // BLOCK_DIM
        col_blocks = grad.shape[1] // BLOCK_DIM
        block_grad = grad.reshape(row_blocks, BLOCK_DIM, col_blocks, BLOCK_DIM)
        if calculation_strategy == "mean_abs":
            score = block_grad.mean(dim=(1, 3)).abs()
        elif calculation_strategy in {"abs_mean", "absmean"}:
            score = block_grad.abs().mean(dim=(1, 3))
        elif calculation_strategy == "L1":
            score = block_grad.abs().sum(dim=(1, 3))
        elif calculation_strategy == "L2":
            score = torch.sqrt((block_grad.abs() ** 2).sum(dim=(1, 3)))
        else:
            raise ValueError(f"Unsupported SMT calculation_strategy={calculation_strategy}")
        scores[name] = score
    return scores


def select_top_gradient_blocks(scores, total_blocks, selection_strategy="no_restriction"):
    """Select SMT blocks using the official no_restriction/norm_dist semantics."""
    if total_blocks <= 0:
        return {}
    if selection_strategy == "norm_dist":
        selected = {}
        for name, score in scores.items():
            flat_scores = score.reshape(-1)
            k = min(total_blocks, flat_scores.numel())
            values, indices = torch.topk(flat_scores, k=k)
            selected[name] = [
                (int(idx // score.shape[1]), int(idx % score.shape[1]))
                for idx in indices.tolist()
            ]
        return selected

    heap = []
    for name, score in scores.items():
        flat_scores = score.reshape(-1)
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
    warmup_steps=0,
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
    legacy_target_modules = os.environ.get("SMT_TARGET_MODULES")
    if legacy_target_modules:
        legacy_modules = [item.strip() for item in legacy_target_modules.split(",") if item.strip()]
        attention_modules = [m for m in legacy_modules if m in {"q_proj", "k_proj", "v_proj", "o_proj"}]
        mlp_modules = [m for m in legacy_modules if m in {"gate_proj", "up_proj", "down_proj"}]
    else:
        attention_modules = _parse_csv_env("SMT_ATTENTION_TARGET_MODULES", "q_proj,k_proj,v_proj")
        mlp_modules = _parse_csv_env("SMT_MLP_TARGET_MODULES", "gate_proj,up_proj,down_proj")
    selection_strategy = os.environ.get("SMT_SELECTION_STRATEGY", "no_restriction")
    calculation_strategy = os.environ.get("SMT_CALCULATION_STRATEGY", "mean_abs")
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

    train_dataset, data_collator, dynamic_padding = build_lmflow_text_dataset(
        dataset_path, tokenizer, max_seq_length
    )
    logger.info(
        "[SMT] data_processing=lmflow_text, samples=%s, dynamic_padding=%s, "
        "label_pad_token_id=-100, attention_mask=true",
        len(train_dataset),
        dynamic_padding,
    )

    selection_start = time.time()
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    attention_layers, mlp_layers = split_smt_candidate_layers(model, attention_modules, mlp_modules)
    attn_blocks, mlp_blocks = compute_smt_block_budgets(attention_layers, mlp_layers, target_params)
    target_layers = {**attention_layers, **mlp_layers}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    calibration_monitor = _PhaseMemoryMonitor().start()
    calibration_start = time.time()
    logger.info(
        f"[SMT] Starting gradient calibration: steps={smt_calibration_steps}, "
        f"batch_size={smt_calibration_batch_size}, attention_modules={attention_modules}, "
        f"mlp_modules={mlp_modules}, selection_strategy={selection_strategy}, "
        f"calculation_strategy={calculation_strategy}"
    )
    gradient_tensors, completed_calibration_steps = accumulate_gradient_tensors(
        model=model,
        layers=target_layers,
        train_dataset=train_dataset,
        calibration_steps=smt_calibration_steps,
        calibration_batch_size=smt_calibration_batch_size,
        data_collator=data_collator,
    )
    calibration_monitor.stop()
    calibration_seconds = time.time() - calibration_start
    calibration_peak_allocated = 0
    calibration_peak_reserved = 0
    if torch.cuda.is_available():
        calibration_peak_allocated = torch.cuda.max_memory_allocated() // (1024 * 1024)
        calibration_peak_reserved = torch.cuda.max_memory_reserved() // (1024 * 1024)
    logger.info(
        f"[SMT] calibration_seconds={calibration_seconds:.2f}, "
        f"completed_steps={completed_calibration_steps}"
    )
    logger.info(f"[SMT] calibration_peak_allocated_mb={calibration_peak_allocated}")
    logger.info(f"[SMT] calibration_peak_reserved_mb={calibration_peak_reserved}")
    logger.info(f"[SMT] calibration_peak_cpu_pss_mb={calibration_monitor.peak_pss_mb}")
    logger.info(f"[SMT] calibration_peak_cpu_rss_mb={calibration_monitor.peak_rss_mb}")
    attention_grads = {name: gradient_tensors[name] for name in attention_layers if name in gradient_tensors}
    mlp_grads = {name: gradient_tensors[name] for name in mlp_layers if name in gradient_tensors}
    attention_scores = _block_scores_from_grads(attention_grads, calculation_strategy)
    mlp_scores = _block_scores_from_grads(mlp_grads, calculation_strategy)
    selected_blocks = {}
    selected_blocks.update(select_top_gradient_blocks(attention_scores, attn_blocks, selection_strategy))
    selected_blocks.update(select_top_gradient_blocks(mlp_scores, mlp_blocks, selection_strategy))
    del gradient_tensors, attention_grads, mlp_grads, attention_scores, mlp_scores

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
                "attention_modules": attention_modules,
                "mlp_modules": mlp_modules,
                "legacy_target_modules": legacy_target_modules,
                "target_params": target_params,
                "block_dim": BLOCK_DIM,
                "selection_strategy": selection_strategy,
                "calculation_strategy": calculation_strategy,
                "smt_budget_allocation": os.environ.get("SMT_BUDGET_ALLOCATION", "attention_only"),
                "num_submatrix_attn": attn_blocks,
                "num_submatrix_mlp": mlp_blocks,
                "requested_calibration_steps": smt_calibration_steps,
                "completed_calibration_steps": completed_calibration_steps,
                "calibration_batch_size": smt_calibration_batch_size,
                "calibration_seconds": calibration_seconds,
                "calibration_peak_allocated_mb": calibration_peak_allocated,
                "calibration_peak_reserved_mb": calibration_peak_reserved,
                "calibration_peak_cpu_pss_mb": calibration_monitor.peak_pss_mb,
                "calibration_peak_cpu_rss_mb": calibration_monitor.peak_rss_mb,
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
        warmup_steps=warmup_steps,
        bf16=bf16,
        save_strategy=save_strategy,
        logging_steps=5,
        report_to=report_to,
        seed=seed,
        dataloader_num_workers=_dataloader_num_workers(),
        remove_unused_columns=False,
        deepspeed=_deepspeed_config(os.path.join(RAPA_HOME, "LMFlow", "configs", "rapa", "ds_zero1_sift.json")),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    resume_training = os.environ.get("RESUME_TRAINING", "false").lower() in {"1", "true", "yes"}
    last_checkpoint = get_last_checkpoint(output_dir) if resume_training else None
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if os.environ.get("RAPA_SKIP_SAVE", "false").lower() in {"1", "true", "yes"}:
        logger.info("[SMT] RAPA_SKIP_SAVE=true; skipping model save for profiling")
        return output_dir

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
