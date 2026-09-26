#!/usr/bin/env python3
"""CLI Script to Evaluate Model on Factual Question-Answering Benchmarks.

Measures factual accuracy, sequence confidence p_bar, first-token margin M_1,
entropy H_1, Gate A/B/C/D distribution, and Expected Calibration Error (ECE).
"""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add root src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import evaluate_single_query
from src.evaluation.metrics import summarize_gates


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on factual benchmarks")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model identifier or local path")
    parser.add_argument("--data_file", type=str, default="data/sample_queries.jsonl", help="Path to input jsonl file")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Path to save output results")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of evaluation samples")
    parser.add_argument("--tau", type=float, default=0.95, help="Confidence threshold for Gate stratification")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading model: {args.model_name_or_path}")

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

    # Load data
    data_path = Path(args.data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print(f"Evaluating {len(samples)} factual queries...")
    results = []
    confidences = []
    accuracies = []

    for item in tqdm(samples, desc="Evaluating"):
        question = item.get("question", "")
        gold_aliases = item.get("gold_aliases", [])
        if not question or not gold_aliases:
            continue

        prompt = f"Question: {question}\nAnswer:"
        res = evaluate_single_query(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            gold_aliases=gold_aliases,
            device=args.device,
        )
        res["question_id"] = item.get("id", item.get("sample_id", ""))
        results.append(res)
        confidences.append(res["p_bar"])
        accuracies.append(res["is_correct"])

    summary = summarize_gates(confidences, accuracies, tau=args.tau)
    
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Samples: {summary.get('total', 0)}")
    print(f"Accuracy:      {summary.get('accuracy', 0.0) * 100:.2f}%")
    print(f"ECE:           {summary.get('ece', 0.0):.4f}")
    gate_counts = summary.get("gate_counts", {})
    gate_rates = summary.get("gate_rates", {})
    for g in ["Gate_A", "Gate_B", "Gate_C", "Gate_D"]:
        print(f"  {g}: {gate_counts.get(g, 0)} ({gate_rates.get(g, 0.0) * 100:.2f}%)")
    print("=" * 50)

    output_payload = {
        "model": args.model_name_or_path,
        "summary": summary,
        "details": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
