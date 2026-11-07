"""
Phase C3: Training Loss Module.

Exact formulation: Confidence-weighted soft BCE loss.
L = sum(W * BCEWithLogits(logits, Y)) / sum(W)
where:
  Y: continuous soft label probabilities from Phase B (52,884 rows)
  W: continuous sample weights from Phase B
  Abstained targets have W=0 and contribute zero gradient.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_soft_bce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Confidence-weighted binary cross entropy loss with logits.

    Args:
        logits: Tensor of shape (..., 12) containing raw unnormalized network logits.
        targets: Tensor of shape (..., 12) containing continuous soft targets Y in [0, 1].
        weights: Tensor of shape (..., 12) containing sample weights W >= 0.

    Returns:
        Scalar loss tensor normalized by sum of active weights.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    weighted_bce = bce * weights

    loss = weighted_bce.sum() / weights.sum().clamp(min=1.0)

    return loss


class WeightedSoftBCELoss(nn.Module):
    """PyTorch loss module wrapper for weighted_soft_bce_loss."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return weighted_soft_bce_loss(logits, targets, weights)
