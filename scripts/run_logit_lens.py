#!/usr/bin/env python3
"""CLI Script for Layer-wise Logit Lens Analysis.

Tracks early commitment depth l*, layer-wise margin M_l, and decision entropy H_l.
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

from src.probing.logit_lens import run_logit_lens_probe


def parse_args():
    parser = argparse.ArgumentParser(description="Run Logit Lens layer-wise probing")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model identifier or local path")
    parser.add_argument("--data_file", type=str, default="data/sample_queries.jsonl", help="Path to input jsonl file")
    parser.add_argument("--output_file", type=str, default="logit_lens_results.json", help="Path to save output results")
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples to probe")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading model for Logit Lens: {args.model_name_or_path}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
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

    print(f"Running Logit Lens on {len(samples)} queries...")
    results = []
    l_star_norms = []

    for item in tqdm(samples, desc="Probing"):
        question = item.get("question", "")
        prompt = f"Question: {question}\nAnswer:"
        probe_res = run_logit_lens_probe(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=args.device,
        )
        probe_res["question_id"] = item.get("id", item.get("sample_id", ""))
        probe_res["question"] = question
        results.append(probe_res)
        l_star_norms.append(probe_res["l_star_norm"])

    avg_l_star = sum(l_star_norms) / len(l_star_norms) if l_star_norms else 0.0

    print("\n" + "=" * 50)
    print("LOGIT LENS SUMMARY")
    print("=" * 50)
    print(f"Probed Samples:          {len(results)}")
    print(f"Mean Normalized l* (l*/L): {avg_l_star:.4f}")
    print("=" * 50)

    output_payload = {
        "model": args.model_name_or_path,
        "mean_l_star_norm": avg_l_star,
        "probes": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"Probing results saved to {args.output_file}")


if __name__ == "__main__":
    main()
