"""LT-SFT (Lottery Ticket Sparse Fine-Tuning).

This follows the original composable-sft LT-SFT training shape as closely as
this pipeline allows: run dense fine-tuning to find lottery-ticket parameters,
rank maskable parameters globally by |theta_search - theta_0|, restore the
previous model state, then sparsely fine-tune the same parameters while zeroing
gradients outside the selected mask.
"""
import logging
import os
import json
import time
import gc
import threading

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)
RAPA_HOME = os.environ.get("RAPA_HOME", "/data/nksol0405/LLM/rapa")

try:
    from lmflow.pipeline.rapa.sift_tuner import SparseLinear, restore_linear_modules
except ImportError:
    try:
        from .sift_tuner import SparseLinear, restore_linear_modules
    except ImportError:
        from sift_tuner import SparseLinear, restore_linear_modules

try:
    from lmflow.pipeline.rapa.data_utils import build_lmflow_text_dataset
except ImportError:
    try:
        from .data_utils import build_lmflow_text_dataset
    except ImportError:
        from data_utils import build_lmflow_text_dataset


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


def _deepspeed_config(default_path):
    value = os.environ.get("DS_CONFIG", default_path)
    if value.lower() in {"", "0", "false", "none", "no"}:
        return None
    return value


def _dataloader_num_workers():
    return int(os.environ.get("RAPA_DATALOADER_NUM_WORKERS", "0"))


class SparseGradientMaskTrainer(Trainer):
    """Trainer that applies LT-SFT masks by zeroing non-ticket gradients."""

    def __init__(self, *args, gradient_masks=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_masks = dict(gradient_masks or {})

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        self._mask_gradients()
        return loss

    def _mask_gradients(self):
        if not self.gradient_masks:
            return
        for name, param in self.model.named_parameters():
            if name not in self.gradient_masks or param.grad is None:
                continue
            grad_flat = param.grad.view(-1)
            selected = self.gradient_masks[name]
            if selected.numel() == 0:
                grad_flat.zero_()
                continue
            if selected.device != grad_flat.device:
                selected = selected.to(grad_flat.device)
                self.gradient_masks[name] = selected
            if selected.numel() >= grad_flat.numel():
                continue
            selected_grad = grad_flat.index_select(0, selected)
            grad_flat.zero_()
            grad_flat.scatter_(0, selected, selected_grad)


def _target_linear_weight_names(model, target_modules):
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in module_name for target in target_modules):
            names.append(module_name + ".weight")
    return set(names)


def _named_maskable_params(model, maskable_names):
    return [(name, param) for name, param in model.named_parameters() if name in maskable_names]


def _copy_params_to_cpu(named_params):
    return {
        name: param.detach().cpu().clone()
        for name, param in named_params
    }


def _restore_params(named_params, saved_params):
    with torch.no_grad():
        for name, param in named_params:
            param.data.copy_(saved_params[name].to(device=param.device, dtype=param.dtype))


def _set_trainable_params(model, trainable_names):
    trainable_names = set(trainable_names)
    for name, param in model.named_parameters():
        param.requires_grad = name in trainable_names


def _replace_selected_weights_with_sparse_deltas(model, selected_indices):
    replacements = 0
    trainable = 0
    for weight_name, flat_idx in selected_indices.items():
        if flat_idx.numel() == 0 or not weight_name.endswith(".weight"):
            continue
        module_name = weight_name[:-len(".weight")]
        parent = model
        parts = module_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        module = getattr(parent, parts[-1])
        if not isinstance(module, nn.Linear):
            continue
        sparse = SparseLinear(module, flat_idx.cpu())
        setattr(parent, parts[-1], sparse)
        replacements += 1
        trainable += sparse.sparse_delta.numel()
    return replacements, trainable


def _ltsft_chunk_size():
    return int(os.environ.get("LTSFT_SELECTION_CHUNK_SIZE", "16000000"))


def _mask_selected_in_chunk(diff, selected, start):
    if selected is None or selected.numel() == 0:
        return diff
    selected = selected.cpu()
    inside = selected[(selected >= start) & (selected < start + diff.numel())]
    if inside.numel() > 0:
        diff[inside - start] = float("-inf")
    return diff


def _iter_changed_chunks(named_params, original_params, selected_indices, chunk_size):
    for name, param in named_params:
        param_flat = param.detach().reshape(-1)
        original_flat = original_params[name].reshape(-1)
        selected = selected_indices.get(name)
        total = param_flat.numel()
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            diff = (
                param_flat[start:end].detach().to(device="cpu", dtype=torch.float32)
                - original_flat[start:end].to(dtype=torch.float32)
            ).abs()
            yield name, start, _mask_selected_in_chunk(diff, selected, start)


def _count_changed_at_least(named_params, original_params, selected_indices, threshold, chunk_size):
    count = 0
    for _, _, diff in _iter_changed_chunks(named_params, original_params, selected_indices, chunk_size):
        count += int(torch.count_nonzero(diff >= threshold).item())
    return count


def _max_changed(named_params, original_params, selected_indices, chunk_size):
    max_value = 0.0
    for _, _, diff in _iter_changed_chunks(named_params, original_params, selected_indices, chunk_size):
        finite = diff[torch.isfinite(diff)]
        if finite.numel() > 0:
            max_value = max(max_value, float(finite.max().item()))
    return max_value


def _find_global_change_threshold(named_params, original_params, selected_indices, k, chunk_size):
    max_value = _max_changed(named_params, original_params, selected_indices, chunk_size)
    if max_value <= 0:
        return 0.0

    low = 0.0
    high = max_value
    steps = int(os.environ.get("LTSFT_THRESHOLD_BISECT_STEPS", "32"))
    for _ in range(steps):
        mid = (low + high) / 2.0
        if _count_changed_at_least(named_params, original_params, selected_indices, mid, chunk_size) >= k:
            low = mid
        else:
            high = mid
    return low


def _select_global_topk_changed_params(named_params, original_params, selected_indices, k):
    """Select the next globally largest changed coordinates, matching LT-SFT.

    The original implementation computes a global masking threshold over all
    still-frozen maskable parameters and then selects every coordinate whose
    absolute delta reaches that threshold. Ties may therefore select slightly
    more than k parameters; this preserves that behavior.
    """
    total_maskable = sum(param.numel() for _, param in named_params)
    already_selected = sum(selected_indices.get(name, torch.empty(0, dtype=torch.long)).numel()
                           for name, _ in named_params)
    remaining = total_maskable - already_selected
    if remaining <= 0 or k <= 0:
        return 0, None
    k = min(k, remaining)

    chunk_size = _ltsft_chunk_size()
    if k == remaining:
        threshold = None
    else:
        threshold = _find_global_change_threshold(named_params, original_params, selected_indices, k, chunk_size)

    remaining_to_take = k
    newly_selected = 0
    new_indices_by_name = {name: [] for name, _ in named_params}
    for name, start, diff in _iter_changed_chunks(named_params, original_params, selected_indices, chunk_size):
        if threshold is None:
            take = torch.ones(diff.numel(), dtype=torch.bool)
        else:
            # Cap the number of selected coordinates to k.  With very short
            # mask-search runs many coordinates can tie at threshold 0; selecting
            # every tie would build enormous index tensors and can trigger OOM.
            take = diff > threshold
            if remaining_to_take > int(torch.count_nonzero(take).item()):
                need_ties = remaining_to_take - int(torch.count_nonzero(take).item())
                ties = (diff == threshold).nonzero(as_tuple=False).flatten()
                if ties.numel() > 0:
                    take[ties[:need_ties]] = True
        local_idx = take.nonzero(as_tuple=False).flatten().to(torch.long)
        if local_idx.numel() > remaining_to_take:
            local_idx = local_idx[:remaining_to_take]
        if local_idx.numel() > 0:
            new_indices_by_name[name].append((local_idx + start).cpu())
            remaining_to_take -= local_idx.numel()
        if remaining_to_take <= 0:
            break

    for name, _ in named_params:
        current = selected_indices.get(name)
        if new_indices_by_name[name]:
            new_idx = torch.cat(new_indices_by_name[name])
        else:
            new_idx = torch.empty(0, dtype=torch.long)
        if current is None or current.numel() == 0:
            selected_indices[name] = new_idx.cpu()
        elif new_idx.numel() > 0:
            selected_indices[name] = torch.cat([current.cpu(), new_idx.cpu()])
        newly_selected += new_idx.numel()

    return newly_selected, threshold


def train_ltsft(
    model_name_or_path, dataset_path, output_dir, num_train_epochs=1, max_steps=-1,
    per_device_train_batch_size=1, gradient_accumulation_steps=1, learning_rate=5e-5,
    lr_scheduler_type="linear", warmup_steps=0, max_seq_length=512, target_params=170_000_000,
    bf16=True, hf_token=None, seed=42, report_to="none",
    ltsft_mask_search_steps=None,
    ltsft_n_ft_iterations=None,
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    tok_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16 if bf16 else torch.float32, **tok_kwargs)

    train_dataset, data_collator, dynamic_padding = build_lmflow_text_dataset(
        dataset_path, tokenizer, max_seq_length
    )
    logger.info(
        "[LT-SFT] data_processing=lmflow_text, samples=%s, dynamic_padding=%s, "
        "label_pad_token_id=-100, attention_mask=true",
        len(train_dataset),
        dynamic_padding,
    )

    selection_start = time.time()
    maskable_names = _target_linear_weight_names(model, target_modules)
    maskable_params = _named_maskable_params(model, maskable_names)
    total_target = sum(param.numel() for _, param in maskable_params)
    if total_target == 0:
        raise ValueError("[LT-SFT] No maskable target Linear weights were found.")
    target_params = min(target_params, total_target)
    if ltsft_mask_search_steps is None:
        ltsft_mask_search_steps = int(os.environ.get("LTSFT_MASK_SEARCH_STEPS", "100"))
    if ltsft_n_ft_iterations is None:
        ltsft_n_ft_iterations = int(os.environ.get("LTSFT_N_FT_ITERATIONS", "1"))
    if ltsft_n_ft_iterations <= 0:
        raise ValueError("ltsft_n_ft_iterations must be >= 1")
    logger.info(
        f"[LT-SFT] maskable_params={total_target:,}, target_params={target_params:,}, "
        f"mask_search_steps={ltsft_mask_search_steps}, n_ft_iterations={ltsft_n_ft_iterations}"
    )

    original_params = _copy_params_to_cpu(maskable_params)
    selected_indices = {name: torch.empty(0, dtype=torch.long) for name, _ in maskable_params}

    _set_trainable_params(model, maskable_names)

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    ds_config = _deepspeed_config(os.path.join(RAPA_HOME, "LMFlow", "configs", "rapa", "ds_zero1_sift.json"))
    per_iteration_target = target_params // ltsft_n_ft_iterations
    remainder = target_params % ltsft_n_ft_iterations

    for iteration in range(ltsft_n_ft_iterations):
        logger.info(f"[LT-SFT] Fine-tuning iteration {iteration + 1}/{ltsft_n_ft_iterations}")
        maskable_params = _named_maskable_params(model, maskable_names)
        previous_params = original_params if iteration == 0 else _copy_params_to_cpu(maskable_params)
        _set_trainable_params(model, maskable_names)

        search_args = TrainingArguments(
            output_dir=os.path.join(output_dir, f"lt_mask_search_iter_{iteration + 1}"),
            num_train_epochs=num_train_epochs,
            max_steps=ltsft_mask_search_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            bf16=bf16,
            save_strategy="no",
            logging_steps=5,
            report_to=report_to,
            seed=seed,
            dataloader_num_workers=_dataloader_num_workers(),
            remove_unused_columns=False,
            deepspeed=ds_config,
        )
        logger.info("[LT-SFT] Starting dense lottery-ticket mask search")
        search_trainer = Trainer(
            model=model,
            args=search_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        search_trainer.train()
        del search_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        maskable_params = _named_maskable_params(model, maskable_names)
        select_k = per_iteration_target + (1 if iteration < remainder else 0)
        newly_selected, threshold = _select_global_topk_changed_params(
            maskable_params,
            original_params,
            selected_indices,
            select_k,
        )
        effective_trainable = sum(idx.numel() for idx in selected_indices.values())
        logger.info(
            f"[LT-SFT] iteration={iteration + 1}, threshold={threshold}, "
            f"newly_selected={newly_selected:,}, effective_trainable={effective_trainable:,}"
        )

        _restore_params(maskable_params, previous_params)
        if previous_params is not original_params:
            del previous_params
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _set_trainable_params(model, set())
        sparse_replacements, sparse_trainable = _replace_selected_weights_with_sparse_deltas(model, selected_indices)
        optimizer_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"[LT-SFT] sparse_replacements={sparse_replacements}, sparse_delta_trainable={sparse_trainable:,}, "
            f"optimizer_trainable={optimizer_trainable:,}, "
            f"effective_sparse_trainable={effective_trainable:,}"
        )
        logger.info(f"[LT-SFT] weight_selection_seconds={time.time() - selection_start:.2f}")

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
            dataloader_num_workers=_dataloader_num_workers(),
            remove_unused_columns=False,
            deepspeed=ds_config,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        resume_checkpoint = get_last_checkpoint(output_dir) if iteration == 0 else None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sparse_train_start = time.time()
        sparse_monitor = _PhaseMemoryMonitor().start()
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        sparse_monitor.stop()
        sparse_train_seconds = time.time() - sparse_train_start
        sparse_peak_allocated_mb = 0
        sparse_peak_reserved_mb = 0
        if torch.cuda.is_available():
            sparse_peak_allocated_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024)
            sparse_peak_reserved_mb = int(torch.cuda.max_memory_reserved() / 1024 / 1024)
        logger.info(f"[LT-SFT] sparse_train_wall_seconds={sparse_train_seconds:.2f}")
        logger.info(f"[LT-SFT] sparse_train_peak_gpu_allocated_mb={sparse_peak_allocated_mb}")
        logger.info(f"[LT-SFT] sparse_train_peak_gpu_reserved_mb={sparse_peak_reserved_mb}")
        logger.info(f"[LT-SFT] sparse_train_peak_cpu_pss_mb={sparse_monitor.peak_pss_mb}")
        logger.info(f"[LT-SFT] sparse_train_peak_cpu_rss_mb={sparse_monitor.peak_rss_mb}")
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    meta = {
        "method": "ltsft",
        "masking": "global_topk_gradient_mask",
        "maskable_params": total_target,
        "target_params": target_params,
        "effective_sparse_trainable": sum(idx.numel() for idx in selected_indices.values()),
        "n_ft_iterations": ltsft_n_ft_iterations,
        "mask_search_steps": ltsft_mask_search_steps,
        "target_modules": target_modules,
    }
    with open(os.path.join(output_dir, "ltsft_mask_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if os.environ.get("RAPA_SKIP_SAVE", "false").lower() in {"1", "true", "yes"}:
        logger.info("[LT-SFT] RAPA_SKIP_SAVE=true; skipping model save for profiling")
        return output_dir

    model = restore_linear_modules(model)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[LT-SFT] Model saved to {output_dir}")
    return output_dir
