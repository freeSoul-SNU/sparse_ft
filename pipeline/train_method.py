#!/usr/bin/env python
"""
Unified training entrypoint for all RAPA methods within LMFlow.
Dispatches to the correct tuner based on --method argument.
Training is launched from the shell scripts through srun, then deepspeed.
"""
import argparse
import logging
import os
import sys
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LMFLOW_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAPA_PIPELINE = os.path.join(LMFLOW_DIR, "src")
if RAPA_PIPELINE not in sys.path:
    sys.path.insert(0, RAPA_PIPELINE)


def parse_args():
    p = argparse.ArgumentParser(description="Unified RAPA training script")
    p.add_argument("--method", required=True, choices=["sift", "spiel", "smt", "s2ft", "ltsft"],
                   help="Fine-tuning method to use")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="If >0, override num_train_epochs (used for smoke tests)")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--target_params", type=int, default=170_000_000,
                   help="Target number of trainable parameters (~170M)")
    p.add_argument("--smt_calibration_steps", type=int, default=None,
                   help="SMT gradient calibration steps before block selection (default: env SMT_CALIBRATION_STEPS or 100)")
    p.add_argument("--smt_calibration_batch_size", type=int, default=None,
                   help="SMT gradient calibration batch size (default: env SMT_CALIBRATION_BATCH_SIZE or 1)")
    p.add_argument("--sift_calibration_steps", type=int, default=None,
                   help="SIFT gradient calibration steps before top-k selection (default: env SIFT_CALIBRATION_STEPS or 1)")
    p.add_argument("--sift_calibration_batch_size", type=int, default=None,
                   help="SIFT gradient calibration batch size (default: env SIFT_CALIBRATION_BATCH_SIZE or 1)")
    p.add_argument("--s2ft_calibration_steps", type=int, default=None,
                   help="S2FT activation calibration steps before head/channel selection (default: env S2FT_CALIBRATION_STEPS or 100)")
    p.add_argument("--s2ft_calibration_batch_size", type=int, default=None,
                   help="S2FT activation calibration batch size (default: env S2FT_CALIBRATION_BATCH_SIZE or 1)")
    p.add_argument("--ltsft_mask_search_steps", type=int, default=None,
                   help="LT-SFT dense lottery-ticket mask-search steps (default: env LTSFT_MASK_SEARCH_STEPS or 100)")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--smoke_test", action="store_true",
                   help="Run smoke test (max_steps=20, small dataset subset)")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Local rank for deepspeed (auto-set)")
    return p.parse_args()


def truncate_dataset_for_smoke(dataset_path, n=100):
    """Create a tiny dataset subset for smoke testing."""
    import json, hashlib
    out_dir = os.path.join(os.environ.get("RAPA_HOME", "/home/mms/freeSoul/llm/rapa"), "data")
    os.makedirs(out_dir, exist_ok=True)
    key = hashlib.md5(f"{dataset_path}_{n}".encode()).hexdigest()[:8]
    out_path = os.path.join(out_dir, f"smoke_{key}.json")
    if os.path.exists(out_path):
        return out_path
    with open(dataset_path) as f:
        raw = json.load(f)
    raw["instances"] = raw.get("instances", [])[:n]
    with open(out_path, "w") as f:
        json.dump(raw, f)
    return out_path


def main():
    args = parse_args()

    # Smoke test adjustments
    dataset_path = args.dataset_path
    max_steps = args.max_steps
    if args.smoke_test:
        logger.info("[smoke] Using subset of 100 examples for smoke test")
        dataset_path = truncate_dataset_for_smoke(args.dataset_path, n=100)
        if max_steps < 0:
            max_steps = 20

    os.makedirs(args.output_dir, exist_ok=True)

    common_kwargs = dict(
        model_name_or_path=args.model_name_or_path,
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        max_seq_length=args.max_seq_length,
        target_params=args.target_params,
        bf16=args.bf16,
        hf_token=args.hf_token,
        seed=args.seed,
        report_to=args.report_to,
    )
    if args.smt_calibration_steps is not None:
        common_kwargs["smt_calibration_steps"] = args.smt_calibration_steps
    if args.smt_calibration_batch_size is not None:
        common_kwargs["smt_calibration_batch_size"] = args.smt_calibration_batch_size
    if args.sift_calibration_steps is not None:
        common_kwargs["sift_calibration_steps"] = args.sift_calibration_steps
    if args.sift_calibration_batch_size is not None:
        common_kwargs["sift_calibration_batch_size"] = args.sift_calibration_batch_size
    if args.s2ft_calibration_steps is not None:
        common_kwargs["s2ft_calibration_steps"] = args.s2ft_calibration_steps
    if args.s2ft_calibration_batch_size is not None:
        common_kwargs["s2ft_calibration_batch_size"] = args.s2ft_calibration_batch_size
    if args.ltsft_mask_search_steps is not None:
        common_kwargs["ltsft_mask_search_steps"] = args.ltsft_mask_search_steps
    # Inject max_steps into TrainingArguments via env
    if max_steps > 0:
        os.environ["RAPA_MAX_STEPS"] = str(max_steps)
        common_kwargs["max_steps"] = max_steps

    try:
        from lmflow.pipeline.rapa import METHODS
    except ImportError:
        from pipeline import METHODS
    train_fn = METHODS[args.method]

    logger.info(f"=== Starting {args.method.upper()} training ===")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Target params: {args.target_params:,}")
    if max_steps > 0:
        logger.info(f"Max steps: {max_steps}")

    output = train_fn(**common_kwargs)
    logger.info(f"=== {args.method.upper()} training complete. Output: {output} ===")

    # Save training metadata
    meta = {
        "method": args.method,
        "model": args.model_name_or_path,
        "dataset": args.dataset_path,
        "output_dir": args.output_dir,
        "target_params": args.target_params,
        "smoke_test": args.smoke_test,
    }
    with open(os.path.join(args.output_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
