"""Evaluation Metrics for Confidence Reliability and the Alignment Paradox."""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE; Guo et al., 2017).
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.asarray(confidences)
    accuracies = np.asarray(accuracies)
    assert len(confidences) == len(accuracies)
    
    if len(confidences) == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def classify_gate(confidence: float, is_correct: bool, tau: float = 0.95) -> str:
    """Categorizes generation into Gate A, B, C, or D under confidence threshold tau.
    
    Gate A: conf >= tau, correct (Confident correct)
    Gate B: conf >= tau, incorrect (Confident hallucination / high-confidence error)
    Gate C: conf < tau, incorrect (Uncertain error)
    Gate D: conf < tau, correct (Uncertain correct)
    """
    if confidence >= tau:
        return "Gate_A" if is_correct else "Gate_B"
    else:
        return "Gate_D" if is_correct else "Gate_C"


def summarize_gates(confidences: List[float], accuracies: List[bool], tau: float = 0.95) -> Dict[str, Any]:
    """Computes distribution and counts across Gate A, B, C, and D."""
    n = len(confidences)
    if n == 0:
        return {}

    counts = {"Gate_A": 0, "Gate_B": 0, "Gate_C": 0, "Gate_D": 0}
    for conf, acc in zip(confidences, accuracies):
        g = classify_gate(conf, acc, tau)
        counts[g] += 1

    return {
        "total": n,
        "accuracy": float(np.mean(accuracies)),
        "ece": compute_ece(np.array(confidences), np.array(accuracies)),
        "gate_counts": counts,
        "gate_rates": {g: c / n for g, c in counts.items()},
    }


def compute_brier_score(confidences: np.ndarray, accuracies: np.ndarray) -> float:
    """Brier score = (1/N) * sum((p_i - y_i)^2)."""
    return float(np.mean((np.asarray(confidences) - np.asarray(accuracies, dtype=float)) ** 2))
