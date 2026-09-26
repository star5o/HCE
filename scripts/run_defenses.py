#!/usr/bin/env python3
"""CLI Script to Benchmark Test-Time Defenses on Confident Hallucinations.

Evaluates:
  1. Standard Unconstrained Greedy (Baseline Control)
  2. Defensive System Prompting (Hedging)
  3. Iterative Self-Correction (2-turn critic)
  4. Self-Consistency Voting (K=5)
  5. Layer-Contrastive Decoding (DoLa)
"""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import evaluate_single_query
from src.evaluation.defenses import (
    build_hedging_prompt,
    build_self_correction_prompt,
    evaluate_self_consistency,
    evaluate_dola_decoding,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference-time defenses")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model identifier")
    parser.add_argument("--data_file", type=str, default="data/sample_queries.jsonl")
    parser.add_argument("--output_file", type=str, default="defense_results.json")
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading model for defense benchmark: {args.model_name_or_path}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.device != "cuda":
        model = model.to(args.device)

    data_path = Path(args.data_file)
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print(f"Evaluating defenses across {len(samples)} samples...")
    stats = {
        "baseline": {"correct": 0, "gate_b": 0},
        "hedging": {"correct": 0, "abstain_or_hedge": 0},
        "self_correction": {"correct": 0, "changed": 0},
        "self_consistency": {"correct": 0, "mean_agreement": []},
        "dola": {"correct": 0},
    }

    for item in tqdm(samples, desc="Testing defenses"):
        q = item["question"]
        aliases = item["gold_aliases"]

        # 1. Baseline
        base_res = evaluate_single_query(
            model=model, tokenizer=tokenizer, prompt=f"Question: {q}\nAnswer:", gold_aliases=aliases, device=args.device
        )
        if base_res["is_correct"]:
            stats["baseline"]["correct"] += 1
        if base_res["gate"] == "Gate_B":
            stats["baseline"]["gate_b"] += 1

        # 2. Hedging Prompt
        hedge_prompt = build_hedging_prompt(q)
        hedge_res = evaluate_single_query(
            model=model, tokenizer=tokenizer, prompt=hedge_prompt, gold_aliases=aliases, device=args.device
        )
        if hedge_res["is_correct"]:
            stats["hedging"]["correct"] += 1
        if "not know" in hedge_res["prediction"].lower() or "uncertain" in hedge_res["prediction"].lower():
            stats["hedging"]["abstain_or_hedge"] += 1

        # 3. Self-Correction
        sc_prompt = build_self_correction_prompt(q, base_res["prediction"])
        sc_res = evaluate_single_query(
            model=model, tokenizer=tokenizer, prompt=sc_prompt, gold_aliases=aliases, device=args.device
        )
        if sc_res["is_correct"]:
            stats["self_correction"]["correct"] += 1
        if sc_res["prediction"] != base_res["prediction"]:
            stats["self_correction"]["changed"] += 1

        # 4. Self-Consistency
        sc_vote = evaluate_self_consistency(
            model=model, tokenizer=tokenizer, prompt=f"Question: {q}\nAnswer:", gold_aliases=aliases, k=5, device=args.device
        )
        if sc_vote["is_correct"]:
            stats["self_consistency"]["correct"] += 1
        stats["self_consistency"]["mean_agreement"].append(sc_vote["agreement"])

        # 5. DoLa
        dola_res = evaluate_dola_decoding(
            model=model, tokenizer=tokenizer, prompt=f"Question: {q}\nAnswer:", gold_aliases=aliases, device=args.device
        )
        if dola_res["is_correct"]:
            stats["dola"]["correct"] += 1

    n = len(samples)
    print("\n" + "=" * 50)
    print("DEFENSES BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Total Evaluated: {n}")
    print(f"Baseline Accuracy:        {stats['baseline']['correct'] / n * 100:.2f}% (Gate B errors: {stats['baseline']['gate_b']})")
    print(f"Hedging Prompt Accuracy:  {stats['hedging']['correct'] / n * 100:.2f}% (Abstentions: {stats['hedging']['abstain_or_hedge']})")
    print(f"Self-Correction Accuracy: {stats['self_correction']['correct'] / n * 100:.2f}% (Revisions: {stats['self_correction']['changed']})")
    mean_agr = sum(stats["self_consistency"]["mean_agreement"]) / n if n else 0
    print(f"Self-Consistency (K=5):   {stats['self_consistency']['correct'] / n * 100:.2f}% (Mean Agreement: {mean_agr:.2f})")
    print(f"DoLa Decoding Accuracy:   {stats['dola']['correct'] / n * 100:.2f}%")
    print("=" * 50)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
