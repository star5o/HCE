"""Linear Probes for Residual Stream Truthfulness Representation."""
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


class LinearProbe(nn.Module):
    """Logistic regression probe on hidden activations."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


def train_and_eval_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> Dict[str, float]:
    """Trains a regularized logistic regression probe on hidden states and reports AUROC and accuracy."""
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(train_features, train_labels)

    pred_probs = clf.predict_proba(test_features)[:, 1]
    preds = clf.predict(test_features)

    auroc = roc_auc_score(test_labels, pred_probs)
    acc = accuracy_score(test_labels, preds)

    return {
        "auroc": float(auroc),
        "accuracy": float(acc),
    }
