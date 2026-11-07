"""
Phase D1: DINOv2 ViT-S/14 Multi-Slot Pathology Query Vision Model.

Implements Sections 21, 22, 23, 24, 25, 26:
- Backbone: DINOv2 ViT-S/14 (Meta LVD-142M).
- Freezing: Blocks 0-5 frozen, Blocks 6-11 + LayerNorm trainable.
- Feature Representation: Concatenation of CLS (384) + Mean Patch Tokens (384) -> 768-d per slice group.
- Temporal / Slice Aggregation: Mean pooling across 12 central slice groups per slot.
- SlotHead:
    * Projection: 768 -> 256 (LayerNorm -> Linear -> GELU).
    * Learnable slot embeddings: (6, 256).
    * Learnable pathology queries: (12, 256).
    * Inductive anatomical priors: +0.55 on indicated slots via SLOT_PRIOR_TABLE.
    * Masked attention: -10000.0 mask on missing slots (slot_mask = 0).
    * Per-pathology projection to scalar logits: (B, 12).
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

PATHOLOGY_NAMES = [
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

# Clinical anatomical inductive slot priors (Section 26)
SLOT_PRIOR_TABLE = {
    "ACL":              (0, 3, 5),
    "MCL":              (1, 4),
    "Medial Meniscus":  (0, 1, 3, 4),
    "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA":        (1, 4, 5),
    "Lateral OA":       (1, 4, 5),
    "PF OA":            (0, 2, 5),
    "Effusion":         (0, 2),
    "Synovitis":        (0, 2),
    "Baker's":          (0,),
    "Contusion":        (0, 1, 2),
    "Fracture":         (0, 1, 2, 4, 5),
}


class SlotHead(nn.Module):
    """SlotHead module performing masked pathology-query cross-attention with inductive priors."""

    def __init__(
        self,
        dim: int = 768,
        n_slot: int = 6,
        n_out: int = 12,
        hidden: int = 256,
        p: float = 0.2,
        prior: bool = True,
        prior_weight: float = 0.55,
    ):
        super().__init__()
        self.dim = dim
        self.n_slot = n_slot
        self.n_out = n_out
        self.hidden = hidden
        self.use_prior = prior
        self.prior_weight = prior_weight

        # 1. Slot projection 768 -> 256 (Section 24)
        self.slot_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
        )

        # 2. Learnable slot embeddings: (1, 6, 256)
        self.slot_embeddings = nn.Parameter(torch.randn(1, n_slot, hidden) * 0.02)

        # 3. Learnable pathology queries: (1, 12, 256)
        self.pathology_queries = nn.Parameter(torch.randn(1, n_out, hidden) * 0.02)

        # 4. Pathology classifier weights & biases
        self.classifier = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_out))

        self.dropout = nn.Dropout(p)

        # 5. Inductive prior matrix: (12, 6)
        prior_matrix = torch.zeros(n_out, n_slot, dtype=torch.float32)
        for i, path_name in enumerate(PATHOLOGY_NAMES):
            slots = SLOT_PRIOR_TABLE.get(path_name, ())
            for s in slots:
                prior_matrix[i, s] = self.prior_weight
        self.register_buffer("prior_matrix", prior_matrix)

    def forward(
        self, slot_feats: torch.Tensor, slot_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slot_feats: (B, 6, 768) slot representation vectors.
            slot_mask: (B, 6) binary mask (1.0 = present, 0.0 = missing).
            
        Returns:
            logits: (B, 12) study-level pathology logits.
            attn_weights: (B, 12, 6) attention probabilities.
        """
        B = slot_feats.size(0)

        # Project slot features and add slot positional embeddings: (B, 6, 256)
        k = self.slot_proj(slot_feats) + self.slot_embeddings
        v = k

        # Expand pathology queries across batch: (B, 12, 256)
        q = self.pathology_queries.expand(B, -1, -1)

        # Scaled dot-product: (B, 12, 6)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.hidden)

        # Add domain inductive priors (Section 26)
        if self.use_prior:
            scores = scores + self.prior_matrix.unsqueeze(0)

        # Mask missing slots with large negative value (-10000.0) (Section 25)
        # slot_mask: (B, 6) -> (B, 1, 6)
        mask_penalty = (1.0 - slot_mask.unsqueeze(1)) * -10000.0
        scores = scores + mask_penalty

        # Softmax over slots: (B, 12, 6)
        attn = F.softmax(scores, dim=-1)

        # Weighted slot aggregation per pathology: (B, 12, 256)
        context = torch.bmm(attn, v)
        context = self.dropout(context)

        # Output logits per pathology: (B, 12)
        logits = (context * self.classifier.unsqueeze(0)).sum(dim=-1) + self.bias

        return logits, attn


class PhaseD1Model(nn.Module):
    """Complete Phase D1 Architecture: DINOv2 ViT-S/14 Backbone + SlotHead."""

    def __init__(
        self,
        pretrained_backbone: str = "facebook/dinov2-small",
        freeze_blocks: int = 6,
        n_slot: int = 6,
        n_out: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_prior: bool = True,
        prior_weight: float = 0.55,
    ):
        super().__init__()
        self.n_slot = n_slot
        self.n_out = n_out

        # 1. DINOv2 Backbone (Section 21)
        self.backbone = Dinov2Model.from_pretrained(pretrained_backbone)
        self.feature_dim = 384 * 2  # CLS (384) + Patch mean (384) = 768

        # 2. Enforce Freezing Schedule (Section 22)
        # Embeddings + Blocks 0..(freeze_blocks-1) are FROZEN
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        for i in range(freeze_blocks):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = False

        # Blocks freeze_blocks..11 are TRAINABLE
        for i in range(freeze_blocks, 12):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        # Final LayerNorm is TRAINABLE
        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

        # 3. SlotHead (Section 24)
        self.head = SlotHead(
            dim=self.feature_dim,
            n_slot=n_slot,
            n_out=n_out,
            hidden=hidden_dim,
            p=dropout,
            prior=use_prior,
            prior_weight=prior_weight,
        )

    def extract_slice_features(self, images: torch.Tensor) -> torch.Tensor:
        """Pass 2.5D slice groups through DINOv2 and concatenate CLS + patch mean.
        
        Args:
            images: (N, 3, 224, 224) batch of slice groups.
        Returns:
            features: (N, 768) concatenated CLS + patch tokens.
        """
        outputs = self.backbone(images)
        last_hidden = outputs.last_hidden_state  # (N, 257, 384)

        cls_token = last_hidden[:, 0, :]  # (N, 384)
        patch_tokens = last_hidden[:, 1:, :]  # (N, 256, 384)
        patch_mean = patch_tokens.mean(dim=1)  # (N, 384)

        # Concatenate: (N, 768) (Section 23)
        return torch.cat([cls_token, patch_mean], dim=-1)

    def forward(
        self, images: torch.Tensor, slot_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 6, n_groups, 3, 224, 224) multi-slot 2.5D study tensor.
            slot_mask: (B, 6) slot presence mask.
            
        Returns:
            logits: (B, 12) study-level logits.
            attn: (B, 12, 6) attention weights.
        """
        B, S, G, C, H, W = images.shape
        device = images.device

        # Only pass populated slots through backbone to save compute and VRAM
        # Flatten: (B * S, G, C, H, W)
        flat_mask = slot_mask.view(-1)  # (B * S)
        populated_indices = torch.nonzero(flat_mask > 0.5, as_tuple=False).squeeze(-1)

        # Preallocate slot features: (B * S, 768)
        slot_feats_flat = torch.zeros(B * S, self.feature_dim, device=device, dtype=torch.float32)

        if len(populated_indices) > 0:
            flat_images = images.view(B * S, G, C, H, W)
            pop_images = flat_images[populated_indices]  # (N_pop, G, 3, H, W)
            N_pop = pop_images.size(0)

            # Flatten groups: (N_pop * G, 3, H, W)
            groups_flat = pop_images.view(N_pop * G, C, H, W)

            # Extract DINO features: (N_pop * G, 768)
            group_feats = self.extract_slice_features(groups_flat)

            # Reshape and average across groups: (N_pop, G, 768) -> (N_pop, 768)
            slot_feats_pop = group_feats.view(N_pop, G, self.feature_dim).mean(dim=1)

            slot_feats_flat[populated_indices] = slot_feats_pop

        # Reshape to (B, 6, 768)
        slot_feats = slot_feats_flat.view(B, S, self.feature_dim)

        # Forward through SlotHead
        logits, attn = self.head(slot_feats, slot_mask)

        return logits, attn

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Report total, trainable, and frozen parameter statistics (Section 22)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_percentage": round((trainable / total) * 100.0, 2),
        }
