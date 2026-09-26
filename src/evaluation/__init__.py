"""Evaluation modules for factual accuracy, confidence, and test-time defenses."""
from .evaluator import evaluate_single_query, check_gold_match, normalize_answer
from .metrics import compute_ece, classify_gate, summarize_gates, compute_brier_score
from .defenses import (
    build_hedging_prompt,
    build_self_correction_prompt,
    evaluate_self_consistency,
    evaluate_dola_decoding,
)

__all__ = [
    "evaluate_single_query",
    "check_gold_match",
    "normalize_answer",
    "compute_ece",
    "classify_gate",
    "summarize_gates",
    "compute_brier_score",
    "build_hedging_prompt",
    "build_self_correction_prompt",
    "evaluate_self_consistency",
    "evaluate_dola_decoding",
]
