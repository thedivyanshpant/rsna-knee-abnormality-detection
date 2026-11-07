"""
Phase D1: Weighted Soft-BCE Loss Function.

Implements Section 27:
- Computes binary cross entropy with continuous Phase B soft targets Y in [0.0, 1.0].
- Applies Phase B sample weights W in [0.0, 1.0].
- Properly handles abstentions (W = 0.0) without division by zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_soft_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """Compute the weighted binary cross-entropy loss over continuous soft targets.
    
    Args:
        logits: (B, 12) unnormalized network predictions.
        targets: (B, 12) continuous label probabilities from Phase B.
        weights: (B, 12) sample weights from Phase B (0 for abstentions).
        
    Returns:
        Scalar loss tensor normalized by the sum of active sample weights.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted_bce = bce * weights
    return weighted_bce.sum() / weights.sum().clamp(min=1.0)


class PhaseD1Loss(nn.Module):
    """PyTorch loss module wrapping weighted soft BCE loss."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        return weighted_soft_bce_loss(logits, targets, weights)
