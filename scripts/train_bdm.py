#!/usr/bin/env python3
"""CLI Script to Train Policy using Bounded Dynamic Margin DPO (BDM-DPO).

Caps late-layer margin expansion during preference optimization using
reference epistemic entropy:
    M_max(H) = M_0 * exp( -gamma * H(pi_ref) / ln|V| )
"""
import argparse
import math
import os
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

# Add root src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bdm.loss import BDMDPOConfig, BDMDPOloss


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with BDM-DPO")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base/Reference model path")
    parser.add_argument("--output_dir", type=str, default="./runs/bdm_dpo_checkpoint")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO regularization weight")
    parser.add_argument("--m0", type=float, default=3.0, help="Baseline margin ceiling (value used for all reported results)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Entropy scaling factor (value used for all reported results)")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_lora", action="store_true", help="Train LoRA adapter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(f"BDM-DPO Training Setup: beta={args.beta}, M0={args.m0}, gamma={args.gamma}")
    print(f"Model: {args.model_name_or_path}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load Policy Model
    print("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.device != "cuda":
        policy_model = policy_model.to(args.device)

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, peft_config)
        policy_model.print_trainable_parameters()

    # Configure BDM loss
    loss_config = BDMDPOConfig(beta=args.beta, m0=args.m0, gamma=args.gamma)
    loss_fn = BDMDPOloss(loss_config)

    optimizer = AdamW(policy_model.parameters(), lr=args.lr)

    print("\nBDM-DPO Module initialized successfully.")
    print("The exact training script that produced all reported results is scripts/original/bdm_dpo_train.py.")
    print("To execute full multi-node training on UltraFeedback / HH-RLHF preference pairs,")
    print("refer to the instructions in scripts/reproduce_pipeline.sh.")


if __name__ == "__main__":
    main()
