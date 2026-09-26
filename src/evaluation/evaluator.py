"""Evaluation Pipeline for Factual Queries with Alias Matching and Gate Stratification."""
from __future__ import annotations
import math
import re
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from .metrics import classify_gate, summarize_gates, compute_ece


def normalize_answer(s: str) -> str:
    """Lowercases text, strips punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def check_gold_match(prediction: str, gold_aliases: List[str]) -> bool:
    """Word-boundary regular expression matching against ground-truth aliases."""
    norm_pred = normalize_answer(prediction)
    for alias in gold_aliases:
        norm_alias = normalize_answer(alias)
        if not norm_alias:
            continue
        pattern = r"\b" + re.escape(norm_alias) + r"\b"
        if re.search(pattern, norm_pred):
            return True
    return False


@torch.no_grad()
def evaluate_single_query(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    gold_aliases: List[str],
    max_new_tokens: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Generates completion under deterministic greedy decoding (T=0) and computes confidence metrics.
    
    Computes:
      - Generated completion text
      - First-token decision metrics: M_1 (margin), H_1 (entropy)
      - Sequence geometric mean probability p_bar = exp( (1/T) * sum(ln p_t) )
      - Correctness via alias matching
      - Gate classification (A, B, C, D)
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    # Generate with output scores to capture logits
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    gen_ids = outputs.sequences[0, prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Step-by-step token probabilities and logits
    token_logps = []
    first_token_margin = 0.0
    first_token_entropy = 0.0

    for step_idx, step_logits in enumerate(outputs.scores):
        if step_idx >= len(gen_ids):
            break
        token_id = gen_ids[step_idx].item()
        logits = step_logits[0]  # [vocab_size]
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        token_logp = log_probs[token_id].item()
        token_logps.append(token_logp)

        if step_idx == 0:
            top_vals, _ = torch.topk(logits, k=2)
            first_token_margin = float(top_vals[0] - top_vals[1])
            ent = -(probs * log_probs).sum().item()
            first_token_entropy = float(ent)

    # Sequence geometric mean probability: exp( mean(log p_t) )
    if token_logps:
        mean_logp = sum(token_logps) / len(token_logps)
        p_bar = math.exp(mean_logp)
    else:
        p_bar = 0.0

    is_correct = check_gold_match(gen_text, gold_aliases)
    gate = classify_gate(p_bar, is_correct, tau=0.95)

    return {
        "prompt": prompt,
        "prediction": gen_text,
        "gold_aliases": gold_aliases,
        "is_correct": is_correct,
        "p_bar": p_bar,
        "m_1": first_token_margin,
        "h_1": first_token_entropy,
        "gate": gate,
    }
