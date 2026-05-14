#!/usr/bin/env python
"""SACA-style instruction tuning entrypoint for SMT and S2FT.

The training shape intentionally mirrors ``finetune_rapa_saca_style.py``:
load/format instruction data, build ``DataCollatorForCausalLM``, prepare the
sparse trainable structures, then run ``model.train(); trainer.train()``.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


IGNORE_INDEX = -100
LOGGER = logging.getLogger(__name__)
SPARSE_FT_ROOT = os.environ.get(
    "SPARSE_FT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if SPARSE_FT_ROOT not in sys.path:
    sys.path.insert(0, SPARSE_FT_ROOT)

from pipeline.s2ft_tuner import (  # noqa: E402
    _convert_ffn_layer_to_s2,
    _convert_mha_layer_to_s2,
    _only_optimize_s2_parameters,
    _resolve_ratios,
    _restore_s2_linear_modules,
    _select_units,
)
from pipeline.smt_tuner import (  # noqa: E402
    BLOCK_DIM,
    BlockSparseLinear,
    _block_scores_from_grads,
    accumulate_gradient_tensors,
    compute_smt_block_budgets,
    replace_with_block_sparse_linear,
    select_top_gradient_blocks,
    split_smt_candidate_layers,
)


torch.backends.cuda.matmul.allow_tf32 = True

ALPACA_PROMPT_DICT = {
    "prompt_input": "### Human:{instruction} Input: {input}\n### Assistant:",
    "prompt_no_input": "### Human:{instruction}\n### Assistant:",
}


@dataclass
class DataCollatorForCausalLM:
    tokenizer: object
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        full = [
            f"{self.tokenizer.bos_token}{example['input']}{example['output']}{self.tokenizer.eos_token}"
            for example in instances
        ]
        tokenized_sources = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_full = self.tokenizer(
            full,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = []
        labels = []
        for source_ids, full_ids in zip(tokenized_sources["input_ids"], tokenized_full["input_ids"]):
            if self.predict_with_generate:
                input_ids.append(torch.tensor(source_ids))
                continue
            input_ids.append(torch.tensor(full_ids))
            if self.train_on_source:
                labels.append(torch.tensor(full_ids))
            else:
                labels.append(
                    torch.tensor(
                        [IGNORE_INDEX for _ in range(len(source_ids))]
                        + full_ids[len(source_ids):]
                    )
                )

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if not self.predict_with_generate:
            data_dict["labels"] = pad_sequence(
                labels,
                batch_first=True,
                padding_value=IGNORE_INDEX,
            )
        return data_dict


def load_saca_data(dataset_name):
    if dataset_name == "alpaca":
        return load_dataset("tatsu-lab/alpaca")
    if dataset_name == "alpaca-clean":
        return load_dataset("yahma/alpaca-cleaned")
    if dataset_name == "chip2":
        return load_dataset("laion/OIG", data_files="unified_chip2.jsonl")
    if dataset_name == "self-instruct":
        return load_dataset("yizhongw/self_instruct", name="self_instruct")
    if dataset_name == "hh-rlhf":
        return load_dataset("Anthropic/hh-rlhf")
    if dataset_name == "longform":
        return load_dataset("akoksal/LongForm")
    if dataset_name == "oasst1":
        return load_dataset("timdettmers/openassistant-guanaco")
    if os.path.isfile(dataset_name):
        return load_dataset("json", data_files={"train": dataset_name})
    raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")


def _extract_alpaca_dataset(example):
    prompt_format = (
        ALPACA_PROMPT_DICT["prompt_input"]
        if example.get("input", "") != ""
        else ALPACA_PROMPT_DICT["prompt_no_input"]
    )
    return {"input": prompt_format.format(**example)}


def format_saca_dataset(dataset, dataset_format):
    if dataset_format in {"alpaca", "alpaca-clean"}:
        return dataset.map(_extract_alpaca_dataset, remove_columns=["instruction"])
    if dataset_format == "chip2":
        return dataset.map(
            lambda x: {
                "input": x["turns"].split("\n<bot>: ")[0].replace("<human>: ", ""),
                "output": x["turns"].split("\n<bot>: ")[1],
            }
        )
    if dataset_format == "self-instruct":
        for old, new in [["prompt", "input"], ["completion", "output"]]:
            dataset = dataset.rename_column(old, new)
        return dataset
    if dataset_format == "hh-rlhf":
        return dataset.map(lambda x: {"input": "", "output": x["chosen"]})
    if dataset_format == "oasst1":
        return dataset.map(lambda x: {"input": "", "output": x["text"]})
    return dataset


def load_train_dataset(args):
    if args.task != "instruct":
        raise NotImplementedError("This runner currently supports task=instruct only.")
    dataset_name = args.dataset_path if args.dataset_path else args.dataset
    dataset_format = "input-output" if args.dataset_path else args.dataset
    dataset = load_saca_data(dataset_name)
    dataset = format_saca_dataset(dataset, dataset_format)
    train_ds = dataset["train"]
    if args.train_samples:
        train_ds = train_ds.select(range(min(args.train_samples, len(train_ds))))
    return train_ds


def parse_csv_env(name, default):
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="right",
        use_fast=False,
        legacy=True,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token

    assert (
        tokenizer("### Response:", add_special_tokens=False)["input_ids"]
        + tokenizer(
            f"{'' if isinstance(tokenizer, LlamaTokenizer) else ' '}Negative",
            add_special_tokens=False,
        )["input_ids"]
        == tokenizer("### Response: Negative", add_special_tokens=False)["input_ids"]
    )
    assert (
        tokenizer(f"{tokenizer.bos_token}test", add_special_tokens=False)["input_ids"]
        == [tokenizer.bos_token_id] + tokenizer("test", add_special_tokens=False)["input_ids"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.quantize else torch.float32,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def restore_block_sparse_linear_modules(model):
    for name, module in list(model.named_modules()):
        if not isinstance(module, BlockSparseLinear):
            continue
        merged_weight = module.merge_and_get_weight()
        new_linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=merged_weight.device,
            dtype=merged_weight.dtype,
        )
        new_linear.weight.data.copy_(merged_weight)
        if module.bias is not None:
            new_linear.bias.data.copy_(module.bias.data)
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_linear)
    return model


def prepare_smt_model(model, train_ds, data_collator, args):
    attention_modules = parse_csv_env("SMT_ATTENTION_TARGET_MODULES", "q_proj,k_proj,v_proj")
    mlp_modules = parse_csv_env("SMT_MLP_TARGET_MODULES", "gate_proj,up_proj,down_proj")
    selection_strategy = os.environ.get("SMT_SELECTION_STRATEGY", "no_restriction")
    calculation_strategy = os.environ.get("SMT_CALCULATION_STRATEGY", "mean_abs")
    calibration_steps = args.smt_calibration_steps
    if calibration_steps is None:
        calibration_steps = int(os.environ.get("SMT_CALIBRATION_STEPS", "100"))
    calibration_batch_size = args.smt_calibration_batch_size
    if calibration_batch_size is None:
        calibration_batch_size = int(os.environ.get("SMT_CALIBRATION_BATCH_SIZE", "1"))

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    attention_layers, mlp_layers = split_smt_candidate_layers(model, attention_modules, mlp_modules)
    attn_blocks, mlp_blocks = compute_smt_block_budgets(
        attention_layers,
        mlp_layers,
        args.target_params,
    )
    target_layers = {**attention_layers, **mlp_layers}

    start = time.time()
    gradient_tensors, completed_steps = accumulate_gradient_tensors(
        model=model,
        layers=target_layers,
        train_dataset=train_ds,
        calibration_steps=calibration_steps,
        calibration_batch_size=calibration_batch_size,
        data_collator=data_collator,
    )
    attention_grads = {name: gradient_tensors[name] for name in attention_layers if name in gradient_tensors}
    mlp_grads = {name: gradient_tensors[name] for name in mlp_layers if name in gradient_tensors}
    attention_scores = _block_scores_from_grads(attention_grads, calculation_strategy)
    mlp_scores = _block_scores_from_grads(mlp_grads, calculation_strategy)
    selected_blocks = {}
    selected_blocks.update(select_top_gradient_blocks(attention_scores, attn_blocks, selection_strategy))
    selected_blocks.update(select_top_gradient_blocks(mlp_scores, mlp_blocks, selection_strategy))

    for param in model.parameters():
        param.requires_grad = False
    replacements = replace_with_block_sparse_linear(model, selected_blocks)
    selected_block_count = sum(len(blocks) for blocks in selected_blocks.values())

    meta = {
        "method": "smt",
        "block_dim": BLOCK_DIM,
        "attention_modules": attention_modules,
        "mlp_modules": mlp_modules,
        "target_params": args.target_params,
        "selection_strategy": selection_strategy,
        "calculation_strategy": calculation_strategy,
        "smt_budget_allocation": os.environ.get("SMT_BUDGET_ALLOCATION", "attention_only"),
        "num_submatrix_attn": attn_blocks,
        "num_submatrix_mlp": mlp_blocks,
        "requested_calibration_steps": calibration_steps,
        "completed_calibration_steps": completed_steps,
        "calibration_batch_size": calibration_batch_size,
        "selected_blocks": selected_block_count,
        "selected_layers": len(selected_blocks),
        "replacements": replacements,
        "selection_seconds": time.time() - start,
    }
    LOGGER.info("[SMT] selection_meta=%s", meta)
    return model, meta


def prepare_s2ft_model(model, train_ds, data_collator, args):
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    calibration_steps = args.s2ft_calibration_steps
    if calibration_steps is None:
        calibration_steps = int(os.environ.get("S2FT_CALIBRATION_STEPS", "100"))
    calibration_batch_size = args.s2ft_calibration_batch_size
    if calibration_batch_size is None:
        calibration_batch_size = int(os.environ.get("S2FT_CALIBRATION_BATCH_SIZE", "1"))
    selection_method = args.s2ft_selection_method or os.environ.get("S2FT_SELECTION_METHOD", "random")

    start = time.time()
    ratios, ratio_source = _resolve_ratios(
        model=model,
        target_params=args.target_params,
        v_ratio=args.s2ft_v_ratio,
        o_ratio=args.s2ft_o_ratio,
        u_ratio=args.s2ft_u_ratio,
        d_ratio=args.s2ft_d_ratio,
    )
    selected, completed_steps = _select_units(
        model=model,
        train_dataset=train_ds,
        ratios=ratios,
        method=selection_method,
        calibration_steps=calibration_steps,
        calibration_batch_size=calibration_batch_size,
        seed=args.seed,
        data_collator=data_collator,
    )
    replacements = _convert_mha_layer_to_s2(model, selected)
    replacements += _convert_ffn_layer_to_s2(model, selected)
    _only_optimize_s2_parameters(model)

    meta = {
        "method": "s2ft",
        "target_params": args.target_params,
        "ratio_source": ratio_source,
        "ratios": ratios,
        "selection_method": selection_method,
        "requested_calibration_steps": calibration_steps,
        "completed_calibration_steps": completed_steps,
        "calibration_batch_size": calibration_batch_size,
        "selected_v": sum(len(v) for v in selected["v"].values()),
        "selected_o": sum(len(v) for v in selected["o"].values()),
        "selected_u": sum(len(v) for v in selected["u"].values()),
        "selected_d": sum(len(v) for v in selected["d"].values()),
        "replacements": replacements,
        "selection_seconds": time.time() - start,
    }
    LOGGER.info("[S2FT] selection_meta=%s", meta)
    return model, meta


def prepare_sparse_model(model, train_ds, data_collator, args):
    method = args.method or args.custom_mode
    if method == "smt":
        return prepare_smt_model(model, train_ds, data_collator, args)
    if method == "s2ft":
        return prepare_s2ft_model(model, train_ds, data_collator, args)
    raise ValueError(f"Unsupported method={method!r}; expected smt or s2ft.")


def restore_sparse_model(model, method):
    if method == "smt":
        return restore_block_sparse_linear_modules(model)
    if method == "s2ft":
        return _restore_s2_linear_modules(model)
    return model


def run(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    method = args.method or args.custom_mode
    if method not in {"smt", "s2ft"}:
        raise ValueError(f"Unsupported method={method!r}; expected smt or s2ft.")

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.deepspeed:
        os.environ["DS_CONFIG"] = args.deepspeed
    os.environ["RAPA_DATALOADER_NUM_WORKERS"] = str(args.dataloader_num_workers)

    tokenizer_model, tokenizer = load_model_and_tokenizer(args)
    train_ds = load_train_dataset(args)
    effective_source_max_len = args.max_seq_length or args.source_max_len
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=effective_source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=False,
    )

    model, selection_meta = prepare_sparse_model(tokenizer_model, train_ds, data_collator, args)
    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total number of parameters: {total_params}")

    training_args = TrainingArguments(
        output_dir=args.training_output_dir or os.path.join(args.save_dir, "training_output"),
        optim="adamw_torch",
        remove_unused_columns=False,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        dataloader_num_workers=args.dataloader_num_workers,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        weight_decay=args.wd,
        save_strategy="no",
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        gradient_accumulation_steps=args.accumulation_steps,
        bf16=args.quantize,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        deepspeed=args.deepspeed,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    model.train()
    trainer.train()

    model = restore_sparse_model(model, method)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    with open(os.path.join(args.save_dir, "saca_style_train_meta.json"), "w") as f:
        json.dump(
            {
                "method": method,
                "model": args.model,
                "dataset": args.dataset,
                "output_dir": args.save_dir,
                "target_params": args.target_params,
                "source_max_len": effective_source_max_len,
                "target_max_len": args.target_max_len,
                "train_on_source": args.train_on_source,
                "bf16": args.quantize,
                "deepspeed": args.deepspeed,
                "selection": selection_meta,
                "trainable_params": trainable_params,
                "total_params": total_params,
            },
            f,
            indent=2,
        )
    print(f"Full model saved to {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["smt", "s2ft"], default=None)
    parser.add_argument("--custom_mode", choices=["smt", "s2ft"], default=None)
    parser.add_argument("--task", default="instruct", choices=["instruct"])
    parser.add_argument("--dataset", default="oasst1")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--dataset_cache_dir", default=None)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--train_bs", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--target_params", type=int, default=170_000_000)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--source_max_len", type=int, default=768)
    parser.add_argument("--target_max_len", type=int, default=256)
    parser.add_argument("--train_on_source", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_project", default="oasst1_mtbench_smt_s2ft_saca_style")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--run_group", default="default")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--save_full_model", action="store_true")
    parser.add_argument("--training_output_dir", default=None)
    parser.add_argument("--metrics_enabled", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--smt_calibration_steps", type=int, default=None)
    parser.add_argument("--smt_calibration_batch_size", type=int, default=None)
    parser.add_argument("--s2ft_calibration_steps", type=int, default=None)
    parser.add_argument("--s2ft_calibration_batch_size", type=int, default=None)
    parser.add_argument("--s2ft_v_ratio", type=float, default=None)
    parser.add_argument("--s2ft_o_ratio", type=float, default=None)
    parser.add_argument("--s2ft_u_ratio", type=float, default=None)
    parser.add_argument("--s2ft_d_ratio", type=float, default=None)
    parser.add_argument(
        "--s2ft_selection_method",
        choices=["random", "small_activation", "activation", "large_activation", "large"],
        default=None,
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    print("======= args =======")
    for key, value in vars(parsed_args).items():
        print(f"{key}: {value}")
    print("====================")
    run(parsed_args)
