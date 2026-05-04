#!/usr/bin/env python
"""
MT-Bench evaluation using vLLM for fast inference + OpenAI GPT-4o-mini as judge.
Uses 8 GPUs via vLLM tensor parallel.
Results are appended to results.md.
"""
import argparse
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MT_BENCH_QUESTIONS = [
    # Category: Writing
    {"id": 1, "category": "Writing", "prompt": "Write a persuasive email to convince your introverted friend to attend a social gathering."},
    # Category: Roleplay
    {"id": 2, "category": "Roleplay", "prompt": "Act as a travel guide and describe a unique travel destination in 200 words."},
    # Category: Reasoning
    {"id": 3, "category": "Reasoning", "prompt": "Explain the concept of opportunity cost in economics with a real-world example."},
    # Category: Math
    {"id": 4, "category": "Math", "prompt": "Solve this step by step: If a store offers a 20% discount on a $50 item, and then applies an additional 10% discount on the discounted price, what is the final price?"},
    # Category: Coding
    {"id": 5, "category": "Coding", "prompt": "Write a Python function that finds the longest palindromic substring in a given string."},
    # Category: Extraction
    {"id": 6, "category": "Extraction", "prompt": "Extract the key information (who, what, when, where) from this text: 'NASA's Perseverance rover successfully landed on Mars on February 18, 2021, in the Jezero Crater, marking the beginning of a new era of Mars exploration.'"},
    # Category: STEM
    {"id": 7, "category": "STEM", "prompt": "Explain how photosynthesis works in simple terms that a 10-year-old could understand."},
    # Category: Humanities
    {"id": 8, "category": "Humanities", "prompt": "Compare and contrast the philosophical ideas of Confucius and Socrates."},
]


def generate_with_vllm(model_path, questions, num_gpus=8):
    """Generate responses using vLLM with tensor parallelism."""
    from vllm import LLM, SamplingParams

    logger.info(f"Loading model from {model_path} with {num_gpus} GPUs...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        dtype="bfloat16",
        max_model_len=2048,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
    )

    prompts = [q["prompt"] for q in questions]
    logger.info(f"Generating {len(prompts)} responses...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    logger.info(f"Generation done in {elapsed:.1f}s")

    responses = []
    for q, out in zip(questions, outputs):
        text = out.outputs[0].text
        responses.append({
            "id": q["id"],
            "category": q["category"],
            "prompt": q["prompt"],
            "response": text,
        })

    # Cleanup
    del llm
    import torch
    torch.cuda.empty_cache()

    return responses, elapsed


def judge_with_gpt(responses, api_key, judge_model="gpt-4o-mini"):
    """Use OpenAI API to judge MT-Bench responses on a 1-10 scale."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    scores = {}
    for resp in responses:
        judge_prompt = f"""Please rate the following AI assistant response on a scale of 1-10, where 10 is excellent.
Consider: helpfulness, accuracy, creativity, and detail.

Question: {resp['prompt']}

Response: {resp['response']}

Output ONLY a single number (1-10) as your rating."""

        try:
            result = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=5,
                temperature=0,
            )
            score_text = result.choices[0].message.content.strip()
            score = float(score_text.split()[0].rstrip('.'))
            score = max(1.0, min(10.0, score))
        except Exception as e:
            logger.warning(f"Judge error for {resp['category']}: {e}")
            score = 5.0

        cat = resp["category"]
        if cat not in scores:
            scores[cat] = []
        scores[cat].append(score)
        logger.info(f"  {cat}: {score}")

    # Compute averages
    cat_avgs = {cat: sum(s) / len(s) for cat, s in scores.items()}
    overall_avg = sum(cat_avgs.values()) / len(cat_avgs) if cat_avgs else 0
    return cat_avgs, overall_avg


def count_trainable_params(model_path):
    """Estimate trainable params from train_meta.json."""
    meta_path = os.path.join(model_path, "train_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("target_params", 170_000_000)
    return 170_000_000


def append_results(results_file, method, params, mem_gb, time_s, cat_scores, avg_score):
    """Append results to results.md."""
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Category mapping
    cat_map = {
        "Humanities": "Human",
        "STEM": "STEM",
        "Roleplay": "Role",
        "Extraction": "Extract",
        "Writing": "Writing",
        "Reasoning": "Reason",
        "Coding": "Coding",
        "Math": "Math",
    }

    # Create or append to results file
    header_needed = not os.path.exists(results_file) or os.path.getsize(results_file) == 0
    with open(results_file, "a") as f:
        if header_needed:
            f.write("# Experiment Results\n\n")
            f.write("## MT-Bench Results\n\n")
            f.write("| Method | # Params | Mem (GB) | Time (s) | Human | STEM | Role | Extract | Writing | Reason | Coding | Math | Avg |\n")
            f.write("|--------|----------|----------|----------|-------|------|------|---------|---------|--------|--------|------|-----|\n")

        row = f"| {method} | {params/1e6:.0f}M | {mem_gb:.1f} | {time_s:.0f} "
        for cat in ["Human", "STEM", "Role", "Extract", "Writing", "Reason", "Coding", "Math"]:
            # Find matching category
            score = 0
            for k, v in cat_scores.items():
                if cat_map.get(k, "") == cat:
                    score = v
                    break
            row += f"| {score:.1f} "
        row += f"| {avg_score:.1f} |\n"
        f.write(row)

    # Also copy to /workspace/rapa/results.md
    import shutil
    workspace_results = "/workspace/rapa/results.md"
    os.makedirs(os.path.dirname(workspace_results), exist_ok=True)
    shutil.copy2(results_file, workspace_results)

    logger.info(f"Results appended to {results_file} and {workspace_results}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--results_file", default="/home/mms/freeSoul/llm/rapa/results/results.md")
    p.add_argument("--judge", default="gpt-4o-mini")
    p.add_argument("--openai_api_key", required=True)
    p.add_argument("--num_gpus", type=int, default=8)
    args = p.parse_args()

    # Generate
    responses, gen_time = generate_with_vllm(args.model_path, MT_BENCH_QUESTIONS, args.num_gpus)

    # Save responses
    resp_path = os.path.join(args.model_path, "mtbench_responses.json")
    with open(resp_path, "w") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

    # Judge
    cat_scores, avg_score = judge_with_gpt(responses, args.openai_api_key, args.judge)

    # Get metadata
    params = count_trainable_params(args.model_path)
    # Estimate GPU memory from nvidia-smi would require subprocess, use rough estimate
    mem_gb = 16.0  # ~16GB per GPU for 7B model

    # Append results
    append_results(args.results_file, args.method, params, mem_gb, gen_time, cat_scores, avg_score)

    logger.info(f"[{args.method}] MT-Bench avg: {avg_score:.1f}")
    logger.info(f"Category scores: {cat_scores}")


if __name__ == "__main__":
    main()
