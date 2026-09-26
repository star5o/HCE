"""Layer-wise Logit Lens and Early Commitment Probing.

Implements the internal representation dynamics analysis described in Section 4:
  - Decodes hidden states at each layer l in [1, L] through the final unembedding head
  - Measures top-1 prediction, decision entropy H(P_l), and logit margin M_{1,l} = z_{(1)} - z_{(2)}
  - Identifies earliest permanent commitment depth l*
"""
from __future__ import annotations
import math
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


@torch.no_grad()
def run_logit_lens_probe(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    target_token_id: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Applies the Logit Lens at the final prompt token across all transformer layers.
    
    Args:
        model: Transformer causal language model
        tokenizer: Tokenizer
        prompt: Input query prompt
        target_token_id: Optional token ID of final generated or true token
        
    Returns:
        Dictionary containing layer-wise margins, entropies, top-1 tokens, and l*
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Forward pass requesting all hidden states
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # Tuple of L+1 tensors [1, seq_len, hidden_dim]
    num_layers = len(hidden_states) - 1

    # Focus on the representation at the final prompt position
    layer_margins = []
    layer_entropies = []
    layer_top1_ids = []
    layer_target_ranks = []

    # Get final norm layer if available
    norm_layer = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm_layer = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        norm_layer = model.transformer.ln_f

    lm_head = model.lm_head

    for layer_idx in range(1, num_layers + 1):
        h = hidden_states[layer_idx][:, -1, :] # [1, hidden_dim]
        if norm_layer is not None:
            h = norm_layer(h)
        
        logits = lm_head(h)[0] # [vocab_size]
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # Margin: z_(1) - z_(2)
        top_vals, top_indices = torch.topk(logits, k=2)
        margin = float(top_vals[0] - top_vals[1])
        top1_id = top_indices[0].item()

        # Decision Entropy
        entropy = float(-(probs * log_probs).sum().item())

        layer_margins.append(margin)
        layer_entropies.append(entropy)
        layer_top1_ids.append(top1_id)

        if target_token_id is not None:
            # Rank of target token (0-indexed)
            target_logit = logits[target_token_id].item()
            rank = (logits > target_logit).sum().item() + 1
            layer_target_ranks.append(rank)

    # Determine earliest commitment layer l* where top-1 prediction permanently matches final layer top-1
    final_top1 = layer_top1_ids[-1]
    l_star = num_layers
    for l in range(num_layers, 0, -1):
        if layer_top1_ids[l - 1] == final_top1:
            l_star = l
        else:
            break

    return {
        "num_layers": num_layers,
        "l_star": l_star,
        "l_star_norm": l_star / num_layers,
        "layer_margins": layer_margins,
        "layer_entropies": layer_entropies,
        "layer_top1_ids": layer_top1_ids,
        "layer_target_ranks": layer_target_ranks,
        "final_margin": layer_margins[-1],
        "final_entropy": layer_entropies[-1],
    }
