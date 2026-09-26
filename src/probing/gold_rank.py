"""Ground-Truth Token Rank Tracking and Latency Profiling across Layers."""
from __future__ import annotations
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_gold_token_candidates(tokenizer: PreTrainedTokenizer, gold_aliases: List[str]) -> List[int]:
    """Extracts first-token tokenizations for all aliases, including space prefix variants."""
    token_candidates = set()
    for alias in gold_aliases:
        alias = alias.strip()
        if not alias:
            continue
        # Plain tokenization
        ids_plain = tokenizer.encode(alias, add_special_tokens=False)
        if ids_plain:
            token_candidates.add(ids_plain[0])
        # Leading whitespace tokenization
        ids_space = tokenizer.encode(" " + alias, add_special_tokens=False)
        if ids_space:
            token_candidates.add(ids_space[0])
    return list(token_candidates)


@torch.no_grad()
def evaluate_gold_token_rank(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    gold_aliases: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Computes the minimal rank of ground-truth entity tokens at intermediate and final layers.
    
    Returns:
      - minimal gold rank at intermediate layer (L/2)
      - minimal gold rank at final layer (L)
      - rank transition (L/2 -> L)
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states) - 1

    candidate_ids = get_gold_token_candidates(tokenizer, gold_aliases)
    if not candidate_ids:
        return {"error": "No valid token candidates found for aliases"}

    norm_layer = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm_layer = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        norm_layer = model.transformer.ln_f

    lm_head = model.lm_head

    layer_ranks = {}
    for l_idx in [num_layers // 2, num_layers - 2, num_layers]:
        h = hidden_states[l_idx][:, -1, :]
        if norm_layer is not None:
            h = norm_layer(h)
        logits = lm_head(h)[0]

        # Calculate minimal rank among all candidates
        best_rank = len(logits)
        for cid in candidate_ids:
            target_logit = logits[cid].item()
            rank = (logits > target_logit).sum().item() + 1
            if rank < best_rank:
                best_rank = rank
        layer_ranks[f"layer_{l_idx}"] = best_rank

    return {
        "mid_layer_rank": layer_ranks.get(f"layer_{num_layers // 2}"),
        "penultimate_rank": layer_ranks.get(f"layer_{num_layers - 2}"),
        "final_rank": layer_ranks.get(f"layer_{num_layers}"),
        "all_layer_ranks": layer_ranks,
    }
