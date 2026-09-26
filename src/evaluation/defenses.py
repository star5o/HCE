"""Inference-Time Interventions and Defenses Benchmarking.

Implements the five test-time defenses evaluated in Section 4.3 and Table 3:
  1. Defensive System Prompting (Hedging instructions)
  2. Iterative Self-Correction (Two-turn critic)
  3. Self-Consistency Voting (Sampling with majority agreement)
  4. Layer-Contrastive Decoding (DoLa; Chuang et al., 2024)
  5. Entropy Thresholding (ET-Abstention)
"""
from __future__ import annotations
import math
from typing import List, Dict, Any, Optional
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from .evaluator import check_gold_match, normalize_answer


def build_hedging_prompt(question: str) -> str:
    """Prepends defensive uncertainty prompt to query."""
    return (
        "Answer the following factual question directly. If you are not completely certain "
        "of the answer, respond with 'I do not know' rather than guessing.\n\n"
        f"Question: {question}\nAnswer:"
    )


def build_self_correction_prompt(question: str, initial_answer: str) -> str:
    """Constructs two-turn critic re-evaluation prompt."""
    return (
        f"Question: {question}\n"
        f"Proposed Answer: {initial_answer}\n\n"
        "Carefully review the proposed answer. Is it factually correct? "
        "If you are certain it is correct, output the answer. If there is any doubt or error, "
        "correct it or state 'Uncertain'.\nFinal Answer:"
    )


@torch.no_grad()
def evaluate_self_consistency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    gold_aliases: List[str],
    k: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Generates K completions with temperature sampling and selects consensus."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    # Sample K completions
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    completions = []
    for seq in outputs:
        gen = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
        completions.append(normalize_answer(gen))

    counts = Counter(completions)
    consensus, freq = counts.most_common(1)[0]
    agreement_ratio = freq / k
    is_correct = check_gold_match(consensus, gold_aliases)

    return {
        "consensus": consensus,
        "agreement": agreement_ratio,
        "is_correct": is_correct,
        "all_completions": completions,
    }


@torch.no_grad()
def evaluate_dola_decoding(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    gold_aliases: List[str],
    premature_layer_offset: int = 2,
    alpha: float = 0.1,
    max_new_tokens: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Layer-contrastive decoding (DoLa) contrasting terminal layer with intermediate layer.
    
    Logits: z_dola = z_final - alpha * z_premature
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    curr_ids = inputs.input_ids
    generated_tokens = []

    # Identify total layers
    total_layers = model.config.num_hidden_layers
    premature_layer = max(0, total_layers - premature_layer_offset)

    for _ in range(max_new_tokens):
        out = model(curr_ids, output_hidden_states=True)
        # Final logits
        final_logits = out.logits[:, -1, :] # [1, V]
        
        # Intermediate projection via lm_head
        premature_h = out.hidden_states[premature_layer][:, -1, :]
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            premature_h = model.model.norm(premature_h)
        premature_logits = model.lm_head(premature_h)

        # Contrastive subtraction
        contrastive_logits = final_logits - alpha * premature_logits
        next_token = torch.argmax(contrastive_logits, dim=-1, keepdim=True)
        
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated_tokens.append(token_id)
        curr_ids = torch.cat([curr_ids, next_token], dim=-1)

    gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    is_correct = check_gold_match(gen_text, gold_aliases)

    return {
        "prediction": gen_text,
        "is_correct": is_correct,
    }
