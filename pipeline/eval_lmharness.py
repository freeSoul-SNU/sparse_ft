#!/usr/bin/env python
"""MMLU / CSR evaluation using vLLM backend via lm-eval-harness."""
import argparse
import json
import logging
import os
import subprocess
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# CSR subtasks (commonsense reasoning)
CSR_TASKS = "hellaswag,winogrande,arc_easy,arc_challenge,piqa,boolq,openbookqa"
MMLU_TASK = "mmlu"


def run_eval(model_path, task, num_fewshot, num_gpus=8):
    """Run lm_eval with vLLM backend."""
    if task == "csr":
        task_str = CSR_TASKS
    elif task == "mmlu":
        task_str = "mmlu"
    else:
        task_str = task

    output_path = os.path.join(model_path, f"eval_{task}.json")

    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", f"pretrained={model_path},tensor_parallel_size={num_gpus},dtype=bfloat16,max_model_len=2048",
        "--tasks", task_str,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "auto",
        "--output_path", output_path,
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    start = time.time()
    timeout = int(os.environ.get("LM_EVAL_TIMEOUT", "21600"))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error(f"lm_eval failed:\n{result.stderr[-2000:]}")
        return None, elapsed

    # Parse results
    logger.info(f"lm_eval completed in {elapsed:.0f}s")
    logger.info(result.stdout[-1000:])

    # Try to find lm-eval result json. Recent lm-eval versions can write files
    # such as eval_mmlu_<timestamp>.json next to the requested output_path.
    scores = {}
    search_roots = []
    if os.path.isfile(output_path):
        search_roots.append(os.path.dirname(output_path))
    elif os.path.isdir(output_path):
        search_roots.append(output_path)
    search_roots.append(model_path)

    seen_roots = set()
    for search_root in search_roots:
        if not search_root or search_root in seen_roots or not os.path.isdir(search_root):
            continue
        seen_roots.add(search_root)
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if not d.startswith("checkpoint-")]
            for f in files:
                if not f.endswith(".json"):
                    continue
                if f in {"config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json"}:
                    continue
                if f.startswith("trainer_state") or f.endswith(".index.json"):
                    continue
                try:
                    with open(os.path.join(root, f)) as fh:
                        data = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    continue
                if "results" in data:
                    for task_name, metrics in data["results"].items():
                        if not isinstance(metrics, dict):
                            continue
                        acc = metrics.get("acc,none", metrics.get("acc_norm,none", metrics.get("acc", None)))
                        if acc is not None:
                            scores[task_name] = acc * 100 if acc < 1 else acc

    if not scores:
        # Try parsing from stdout
        for line in result.stdout.split("\n"):
            if "|" in line and "acc" in line.lower():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    try:
                        name = parts[1]
                        acc = float(parts[3]) * 100 if float(parts[3]) < 1 else float(parts[3])
                        scores[name] = acc
                    except (ValueError, IndexError):
                        pass

    return scores, elapsed


def append_results(results_file, method, task, scores, elapsed):
    """Append to results.md."""
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "a") as f:
        if task == "mmlu":
            avg = scores.get("mmlu", sum(scores.values()) / len(scores) if scores else 0)
            f.write(f"\n## MMLU Results (5-shot)\n\n")
            f.write(f"| Method | Avg | Details |\n|--------|-----|--------|\n")
            detail_keys = ["mmlu_humanities", "mmlu_other", "mmlu_social_sciences", "mmlu_stem"]
            detail_scores = [(k, scores[k]) for k in detail_keys if k in scores]
            if not detail_scores:
                detail_scores = sorted(scores.items())[:5]
            detail = ", ".join(f"{k}:{v:.1f}" for k, v in detail_scores)
            f.write(f"| {method} | {avg:.1f} | {detail} |\n")
        elif task == "csr":
            avg = sum(scores.values()) / len(scores) if scores else 0
            f.write(f"\n## CSR Results (0-shot)\n\n")
            f.write(f"| Method | Avg | Details |\n|--------|-----|--------|\n")
            detail = ", ".join(f"{k}:{v:.1f}" for k, v in sorted(scores.items()))
            f.write(f"| {method} | {avg:.1f} | {detail} |\n")

    # Copy to workspace when that mount exists.
    import shutil
    try:
        workspace_results = "/workspace/rapa/results.md"
        os.makedirs(os.path.dirname(workspace_results), exist_ok=True)
        shutil.copy2(results_file, workspace_results)
    except OSError as exc:
        logger.warning(f"Could not copy results to {workspace_results}: {exc}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--task", required=True, choices=["mmlu", "csr"])
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--results_file", default="/home1/irteam/rapa/results/results.md")
    p.add_argument("--num_gpus", type=int, default=8)
    args = p.parse_args()

    scores, elapsed = run_eval(args.model_path, args.task, args.num_fewshot, args.num_gpus)
    if scores:
        append_results(args.results_file, args.method, args.task, scores, elapsed)
        logger.info(f"[{args.method}] {args.task} scores: {scores}")
    else:
        logger.error(f"[{args.method}] {args.task} evaluation failed")


if __name__ == "__main__":
    main()
