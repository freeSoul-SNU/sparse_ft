#!/usr/bin/env python
"""Standalone RAPA instruction tuning, mirroring SACA's finetune.py behavior.

This entrypoint intentionally avoids LMFlow model wrappers and LMFlow merge code.
It trains with the local RAPA PEFT implementation and can save either an adapter
or a fully merged Hugging Face model via PEFT's merge_and_unload().
"""

import argparse
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_RAPA_BASE = os.environ.get("RAPA_HOME", "/home/nksol0405/LLM/rapa")
DEFAULT_RAPA_PEFT_DIR = os.environ.get("PEFT_DIR", os.path.join(DEFAULT_RAPA_BASE, "peft"))
DEFAULT_SACA_IT_DIR = os.environ.get(
    "SACA_INSTRUCTION_TUNING_DIR",
    "/home/nksol0405/LLM/saca/instruction-tuning",
)

rapa_peft_src = os.path.join(DEFAULT_RAPA_PEFT_DIR, "src")
if rapa_peft_src not in sys.path:
    sys.path.insert(0, rapa_peft_src)
if DEFAULT_SACA_IT_DIR not in sys.path:
    sys.path.insert(0, DEFAULT_SACA_IT_DIR)

from peft import (  # noqa: E402
    LoraConfig,
    PacaConfig,
    RapaConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import wandb  # noqa: E402
from utils import (  # noqa: E402
    DataCollatorForCausalLM,
    GradientLogCallback,
    MetricEvalCallback,
    compute_metrics,
    find_all_linear_names,
    format_dataset,
    generate_samples,
    get_parameters_count,
    get_subset,
    load_data,
    preprocess_logits_for_metrics,
)


torch.backends.cuda.matmul.allow_tf32 = True


def build_eval_sets(tokenizer, eval_samples):
    imdb = load_dataset("imdb")

    def _imdb_to_alpaca(examples, instruction, answers, cut_off=1000):
        output = []
        instruction_col = []
        input_col = []
        for i in range(len(examples["text"])):
            instruction_col.append(instruction)
            input_col.append(f'"{examples["text"][i][:cut_off]}"')
            output.append(answers[0] if examples["label"][i] == 0 else answers[1])
        return {"output": output, "instruction": instruction_col, "input": input_col}

    def imdb_to_alpaca_easy(examples):
        return _imdb_to_alpaca(
            examples,
            'Given the following review, classify its sentiment. Answer with the exact sentence - "Review is negative." or "Review is positive.", but without quotes.',
            ["Review is negative.", "Review is positive."],
        )

    def imdb_to_alpaca_quotes(examples):
        return _imdb_to_alpaca(
            examples,
            'Given the following review, classify its sentiment. Answer with the exact sentence - "Review is negative." or "Review is positive.", with quotes.',
            ['"Review is negative."', '"Review is positive."'],
        )

    def imdb_to_alpaca_brackets(examples):
        return _imdb_to_alpaca(
            examples,
            'Given the following review, classify its sentiment. Answer with the exact sentence - "Review is negative." or "Review is positive.", but without quotes and put your answer in square brackets.',
            ["[Review is negative.]", "[Review is positive.]"],
        )

    eval_sets = {}
    converters = {
        "easy": imdb_to_alpaca_easy,
        "quotes": imdb_to_alpaca_quotes,
        "brackets": imdb_to_alpaca_brackets,
    }
    for name, converter in converters.items():
        ds = imdb["test"] if not eval_samples else get_subset(imdb["test"], eval_samples)
        ds = ds.map(converter, batched=True, remove_columns=imdb["train"].column_names)
        eval_sets[name] = format_dataset(ds, "alpaca-clean")

    return eval_sets


def load_train_dataset(args):
    if args.task == "instruct":
        dataset = load_data(args.dataset)
        dataset = format_dataset(dataset, args.dataset)
        return (
            dataset["train"]
            if not args.train_samples
            else dataset["train"].select(range(args.train_samples))
        )

    if args.task == "imdb":
        imdb = load_dataset("imdb")
        train_ds = imdb["train"] if not args.train_samples else get_subset(imdb["train"], args.train_samples)
        raise NotImplementedError("IMDB training is kept out of this standalone runner; use task=instruct.")

    raise NotImplementedError(f"Unsupported task: {args.task}")


def load_model(args):
    if args.nf4:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            trust_remote_code=args.trust_remote_code,
        )
        model = prepare_model_for_kbit_training(model)
        model.config.torch_dtype = torch.bfloat16
        return model

    # Match SACA finetune.py: non-NF4 models are loaded as bf16.
    return AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )


def resolve_target_modules(model, args):
    if args.target_modules == "lm_head":
        return ["lm_head"]
    if args.target_modules == "all":
        return find_all_linear_names(model, lm_head=True)
    if args.target_modules == "attention_only":
        return find_all_linear_names(model, attention_only=True)
    return find_all_linear_names(model)


def build_peft_config(args, target_modules):
    if args.custom_mode == "lora":
        return LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.custom_mode == "lora_pissa":
        return LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            init_lora_weights="pissa_niter_2",
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.custom_mode == "paca":
        return PacaConfig(
            r=args.lora_r,
            paca_alpha=args.lora_alpha,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.custom_mode == "rapa":
        return RapaConfig(
            r=args.lora_r,
            rapa_alpha=args.lora_alpha,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    raise ValueError(f"Unsupported custom_mode for this runner: {args.custom_mode}")


def run(args):
    run_id = args.run_id if args.run_id else wandb.util.generate_id()

    set_seed(args.seed)
    wandb.init(
        id=run_id,
        name=None if args.run_name is None else args.run_name,
        group=None if args.run_group is None else args.run_group,
        project=args.run_project,
        mode="offline" if args.offline else "online",
    )
    wandb.config.update({"seed_val": args.seed})
    wandb.config.update(dict(args._get_kwargs()))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="right",
        use_fast=False,
        legacy=True,
        trust_remote_code=args.trust_remote_code,
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

    train_ds = load_train_dataset(args)
    eval_ds = build_eval_sets(tokenizer, args.eval_samples)
    model = load_model(args)
    print(model)

    if args.custom_mode != "full":
        target_modules = resolve_target_modules(model, args)
        print("target_modules:", target_modules)
        config = build_peft_config(args, target_modules)
        model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir=args.training_output_dir,
        optim="adamw_torch",
        remove_unused_columns=False,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        dataloader_num_workers=args.dataloader_num_workers,
        num_train_epochs=args.epochs,
        weight_decay=args.wd,
        save_strategy="no",
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        gradient_accumulation_steps=args.accumulation_steps,
        bf16=args.quantize,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        deepspeed=args.deepspeed,
    )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=False,
    )

    callbacks = []
    if args.metrics_enabled:
        metric_ds = train_ds.select(range(args.metric_samples))
        callbacks.append(MetricEvalCallback(metric_ds, tokenizer, model, args.metric_bs))
    callbacks.append(GradientLogCallback(model=model))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    params_trainable = get_parameters_count(model, requires_grad=True)
    params_total = get_parameters_count(model, requires_grad=False)
    print(f"Trainable parameters: {params_trainable}")
    print(f"Total number of parameters: {params_total}")
    wandb.config.update({"params_trainable": params_trainable, "params_total": params_total})
    wandb.log({"params_trainable": params_trainable, "params_total": params_total})

    if args.default_model_eval:
        with torch.autocast("cuda"):
            model.eval()
            with torch.no_grad():
                start = time.time()
                print(f"eval took {time.time() - start} seconds")
                if args.generate_samples:
                    to_eval = [data_collator(eval_ds[n].select([0])) for n in eval_ds]
                    to_eval += [data_collator(train_ds.select([0]))]
                    generate_samples(model, tokenizer, to_eval)

    model.train()
    trainer.train()

    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_full_model:
        print("Merging PEFT weights into the base model...")
        merged_model = model.merge_and_unload()
        if merged_model is None:
            merged_model = model
        merged_model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        model = merged_model
        print(f"Full model saved to {args.save_dir}")
    else:
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Adapter weights saved to {args.save_dir}")

    with torch.autocast("cuda"):
        model.eval()
        with torch.no_grad():
            start = time.time()
            print(f"final eval took {time.time() - start} seconds")
            if args.generate_samples:
                samples_after = generate_samples(model, tokenizer, [])
                wandb.log({"generations": wandb.Table(data=[["", x] for x in samples_after], columns=["before", "after"])})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom_mode",
        type=str,
        default="rapa",
        choices=["full", "lora", "lora_pissa", "paca", "rapa"],
    )
    parser.add_argument("--lora_r", type=int, default=602)
    parser.add_argument("--lora_alpha", type=int, default=1)
    parser.add_argument("--target_modules", type=str, default="no_head")
    parser.add_argument("--task", type=str, default="instruct", choices=["instruct", "imdb"])
    parser.add_argument("--dataset", type=str, default="oasst1")
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--metric_samples", type=int, default=100)
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_bs", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--metric_bs", type=int, default=10)
    parser.add_argument("--eval_bs", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--metrics_enabled", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--generate_samples", action="store_true")
    parser.add_argument("--default_model_eval", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--run_project", type=str, default="oasst1_mtbench_rapa_saca_style")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--run_group", type=str, default="default")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_full_model", action="store_true")
    parser.add_argument("--training_output_dir", type=str, default="training_output")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--source_max_len", type=int, default=768)
    parser.add_argument("--target_max_len", type=int, default=256)
    parser.add_argument("--train_on_source", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("======= args =======")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("====================")
    run(args)
