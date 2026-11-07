"""
Phase C3: Multi-Series 3D Vision Architecture.

Maps:
Study (5 canonical MRI sequence slots)
  -> Shared 3D Encoder
  -> 5 Slot Feature Vectors
  -> Masked Slot Aggregation (Masked Pooling or Masked Attention using slot_mask)
  -> Study-Level Representation
  -> 12-Target Prediction Head
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_SLOTS = 5
NUM_TARGETS = 12

SLOT_NAMES = [
    "sagittal_fluid_sensitive",  # Slot 1
    "coronal_fluid_sensitive",   # Slot 2
    "axial_fluid_sensitive",     # Slot 3
    "sagittal_non_fs_t1",        # Slot 4
    "coronal_non_fs_t1",         # Slot 5
]

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


class ResBlock3D(nn.Module):
    """3D residual block with two conv layers and projection downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return F.relu(out, inplace=True)


class SmallResNet3DEncoder(nn.Module):
    """Shared 3D ResNet encoder extracting feature vectors from single 3D MRI volumes.

    Input: (N, 1, 32, 256, 256)
    Output: (N, embed_dim) where embed_dim=128
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Stem: (B, 1, 32, 256, 256) -> (B, 16, 16, 128, 128)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        # Residual layers
        self.layer1 = nn.Sequential(
            ResBlock3D(16, 32, stride=2),  # -> (B, 32, 8, 64, 64)
            ResBlock3D(32, 32, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(32, 64, stride=2),  # -> (B, 64, 4, 32, 32)
            ResBlock3D(64, 64, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResBlock3D(64, embed_dim, stride=2),  # -> (B, 128, 2, 16, 16)
            ResBlock3D(embed_dim, embed_dim, stride=1),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return x.flatten(1)  # (N, embed_dim)


class MaskedSlotAttention(nn.Module):
    """Multi-head attention pooling over sequence slots respecting slot_mask."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable study-level query vector
        self.study_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, slot_features: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_features: (B, 5, embed_dim)
            slot_mask: (B, 5) where 1 indicates available slot, 0 indicates missing slot
        Returns:
            study_embedding: (B, embed_dim)
        """
        B = slot_features.shape[0]
        query = self.study_query.expand(B, -1, -1)  # (B, 1, embed_dim)

        # In PyTorch MultiheadAttention, key_padding_mask is True for padded/missing positions
        key_padding_mask = (slot_mask == 0)  # (B, 5)

        # If an entire study has no available slots (all masked), avoid NaNs by unmasking all
        all_masked = key_padding_mask.all(dim=-1, keepdim=True)
        key_padding_mask = key_padding_mask & ~all_masked

        attn_out, _ = self.mha(
            query=query,
            key=slot_features,
            value=slot_features,
            key_padding_mask=key_padding_mask,
        )
        study_repr = self.norm(attn_out.squeeze(1))  # (B, embed_dim)
        return study_repr


class MultiSeriesKneeModel(nn.Module):
    """Phase C3 Baseline Multi-Series 3D Knee Abnormality Detection Model.

    Flow:
        Input:
          volumes: (B, 5, 1, 32, 256, 256) — 5 canonical sequence slots
          slot_mask: (B, 5) — binary mask indicating present vs zero-padded slots
        Encoder:
          Shared 3D CNN processes each available volume -> (B, 5, embed_dim)
        Aggregation:
          'masked_pool': masked weighted average pooling across slots
          'masked_attention': learned query attention over available slots
        Head:
          Dropout -> Linear(embed_dim, 12) -> logits
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_targets: int = NUM_TARGETS,
        num_slots: int = NUM_SLOTS,
        aggregation: str = "masked_pool",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.embed_dim = embed_dim
        self.num_targets = num_targets
        self.aggregation_type = aggregation

        # Shared 3D Backbone
        self.encoder = SmallResNet3DEncoder(in_channels=1, embed_dim=embed_dim)

        # Slot-specific learned positional/type bias (to distinguish anatomical views)
        self.slot_embeddings = nn.Parameter(torch.randn(1, num_slots, embed_dim) * 0.02)

        # Aggregation
        if aggregation == "masked_attention":
            self.aggregator = MaskedSlotAttention(embed_dim=embed_dim)
        elif aggregation == "masked_pool":
            self.aggregator = None
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}. Must be 'masked_pool' or 'masked_attention'")

        # Prediction head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_targets),
        )

    def forward(self, volumes: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volumes: (B, 5, 1, 32, 256, 256)
            slot_mask: (B, 5)
        Returns:
            logits: (B, 12)
        """
        B, S, C, D, H, W = volumes.shape
        assert S == self.num_slots, f"Expected {self.num_slots} slots, got {S}"

        # Flatten batch and slots for shared encoder pass
        flat_vols = volumes.view(B * S, C, D, H, W)
        flat_features = self.encoder(flat_vols)  # (B * S, embed_dim)
        slot_features = flat_features.view(B, S, self.embed_dim)  # (B, S, embed_dim)

        # Add slot type embeddings
        slot_features = slot_features + self.slot_embeddings

        # Aggregation
        if self.aggregation_type == "masked_attention":
            study_repr = self.aggregator(slot_features, slot_mask)
        else:
            # Masked average pooling
            # Zero out inactive slot features
            mask_expanded = slot_mask.unsqueeze(-1)  # (B, S, 1)
            active_features = slot_features * mask_expanded
            sum_features = active_features.sum(dim=1)  # (B, embed_dim)
            slot_counts = slot_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
            study_repr = sum_features / slot_counts  # (B, embed_dim)

        logits = self.head(study_repr)  # (B, 12)
        return logits


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
