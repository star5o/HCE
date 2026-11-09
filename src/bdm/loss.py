"""Bounded Dynamic Margin DPO (BDM-DPO) and SimPO Loss Implementation.

Formulation:
    Standard DPO optimizes:
        L_DPO = -E [ log sigma( beta * ( log(pi(y_w)/ref(y_w)) - log(pi(y_l)/ref(y_l)) ) ) ]
    
    When gradient pressure persists across all finite margins on factual queries,
    post-training alignment drives logit margins to infinity even when the reference
    distribution exhibits high epistemic uncertainty (entropy H(pi_ref) approx ln|V|).

    BDM-DPO bounds the effective margin by reference epistemic entropy:
        M_max(H) = M_0 * exp( -gamma * (H(pi_ref) / ln|V|) )
    where H(pi_ref) is the reference model's mean per-token entropy over the prompt tokens.
    The bound applies to the raw margin Delta, before scaling by beta.
        Delta_bounded = min( Delta_{theta, ref}, M_max(H) )
        L_BDM_DPO = -E [ log sigma( beta * Delta_bounded ) ]

When the implicit margin Delta reaches M_max, the gradient with respect to further
margin expansion vanishes, halting unconstrained logit blowout while preserving
normal preference alignment on supported facts.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BDMDPOConfig:
    """Configuration for Bounded Dynamic Margin (BDM) alignment."""
    # Defaults match the configuration used for every reported result
    # (beta=0.1, M0=3.0, gamma=2.0, hard clamp).
    beta: float = 0.1
    m0: float = 3.0
    gamma: float = 2.0
    smooth_clamp: bool = False
    clamp_temp: float = 5.0
    label_smoothing: float = 0.0


class BDMDPOloss(nn.Module):
    """BDM-DPO Loss with entropy-dependent dynamic margin ceiling."""

    def __init__(self, config: Optional[BDMDPOConfig] = None):
        super().__init__()
        self.config = config or BDMDPOConfig()

    def compute_reference_entropy(
        self,
        ref_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes mean Shannon entropy of the reference distribution across valid tokens.
        
        Args:
            ref_logits: Tensor of shape [batch_size, seq_len, vocab_size]
            mask: Optional binary mask of shape [batch_size, seq_len]
            
        Returns:
            Scalar entropy values of shape [batch_size]
        """
        probs = F.softmax(ref_logits.float(), dim=-1)
        log_probs = F.log_softmax(ref_logits.float(), dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1)

        if mask is not None:
            token_entropy = token_entropy * mask
            seq_len = mask.sum(dim=-1).clamp(min=1.0)
            return token_entropy.sum(dim=-1) / seq_len
        return token_entropy.mean(dim=-1)

    def dynamic_margin_bound(self, ref_entropy: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Computes the dynamic ceiling: M_max(H) = M_0 * exp(-gamma * H / ln|V|)."""
        max_entropy = math.log(vocab_size)
        norm_entropy = (ref_entropy / max_entropy).clamp(min=0.0, max=1.0)
        return self.config.m0 * torch.exp(-self.config.gamma * norm_entropy)

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        reference_entropy: Optional[torch.Tensor] = None,
        vocab_size: int = 32000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates BDM-DPO loss.
        
        Args:
            policy_chosen_logps: Sum of log probs for preferred response under policy [batch]
            policy_rejected_logps: Sum of log probs for dispreferred response under policy [batch]
            reference_chosen_logps: Sum of log probs for preferred response under reference [batch]
            reference_rejected_logps: Sum of log probs for dispreferred response under reference [batch]
            reference_entropy: Mean token entropy of reference model on the query [batch]
            vocab_size: Size of model vocabulary |V|
            
        Returns:
            Tuple of (loss, chosen_rewards, rejected_rewards, effective_margins)
        """
        pi_chosen_ratio = policy_chosen_logps - reference_chosen_logps
        pi_rejected_ratio = policy_rejected_logps - reference_rejected_logps
        raw_margin = pi_chosen_ratio - pi_rejected_ratio

        if reference_entropy is not None:
            m_max = self.dynamic_margin_bound(reference_entropy, vocab_size)
            if self.config.smooth_clamp:
                # Smooth soft-min: M - softplus(c * (M - x)) / c
                c = self.config.clamp_temp
                bounded_margin = m_max - F.softplus(c * (m_max - raw_margin)) / c
            else:
                bounded_margin = torch.minimum(raw_margin, m_max)
        else:
            bounded_margin = raw_margin

        scaled_margin = self.config.beta * bounded_margin

        if self.config.label_smoothing > 0.0:
            loss = (
                - (1.0 - self.config.label_smoothing) * F.logsigmoid(scaled_margin)
                - self.config.label_smoothing * F.logsigmoid(-scaled_margin)
            ).mean()
        else:
            loss = -F.logsigmoid(scaled_margin).mean()

        chosen_rewards = (self.config.beta * pi_chosen_ratio).detach()
        rejected_rewards = (self.config.beta * pi_rejected_ratio).detach()

        return loss, chosen_rewards, rejected_rewards, bounded_margin.detach()


class BDMSimPOLoss(nn.Module):
    """BDM-SimPO: Simple Preference Optimization with Dynamic Margin Bound."""

    def __init__(self, beta: float = 2.0, gamma_target: float = 1.0, m0: float = 3.8):
        super().__init__()
        self.beta = beta
        self.gamma_target = gamma_target
        self.m0 = m0

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_entropy: Optional[torch.Tensor] = None,
        vocab_size: int = 32000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates BDM-SimPO loss."""
        # Length-normalized logps
        margin = policy_chosen_logps - policy_rejected_logps

        if reference_entropy is not None:
            max_entropy = math.log(vocab_size)
            norm_entropy = (reference_entropy / max_entropy).clamp(min=0.0, max=1.0)
            m_max = self.m0 * torch.exp(-norm_entropy)
            bounded_margin = torch.minimum(margin, m_max)
        else:
            bounded_margin = margin

        scaled_diff = self.beta * (bounded_margin - self.gamma_target)
        loss = -F.logsigmoid(scaled_diff).mean()
        return loss, bounded_margin.detach()
