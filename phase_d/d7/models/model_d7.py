#!/usr/bin/env python3
"""
Phase D7: Neural Architecture Suite for Backbone Capacity and Multi-Window Depth Coverage.
RSNA Knee Abnormality Detection

Models implemented:
1. PhaseD7ModelViTB:
   - Backbone: DINOv2 ViT-B/14 (Meta LVD-142M, 86.6M params)
   - Freezing: Blocks 0-5 frozen, Blocks 6-11 + LayerNorm trainable
   - Feature: CLS (768) + Mean Patch (768) -> 1,536-d per slice group
   - Slice pooling: Mean over groups -> (B, S, 1536)
   - SlotHead: 1536 -> 256 projection + Inductive Priors (+0.55) + Missing Slot Masking (-10000)

2. PhaseD7ModelViTS_MultiWindow:
   - Backbone: DINOv2 ViT-S/14 (Meta LVD-142M, 22.1M params)
   - Freezing: Blocks 0-5 frozen, Blocks 6-11 + LayerNorm trainable
   - Feature: CLS (384) + Mean Patch (384) -> 768-d per slice group
   - Multi-Window Aggregation: 3 depth windows x 4 groups = 12 groups
     Window pooling via learned depth attention across windows -> (B, S, 768)
   - SlotHead: 768 -> 256 projection + Inductive Priors (+0.55) + Missing Slot Masking (-10000)

3. PhaseD7ModelViTB_MultiWindow:
   - Backbone: DINOv2 ViT-B/14 (86.6M params)
   - Multi-Window Aggregation: 3 depth windows x 4 groups = 12 groups
     Window pooling via learned depth attention across windows -> (B, S, 1536)
   - SlotHead: 1536 -> 256 projection + Inductive Priors (+0.55) + Missing Slot Masking (-10000)
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

# Clinical anatomical inductive slot priors (Section 26 / D6)
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
    """SlotHead performing masked pathology-query cross-attention with inductive priors."""

    def __init__(
        self,
        dim: int = 768,
        n_slot: int = 6,
        n_out: int = 12,
        hidden: int = 256,
        p: float = 0.1,
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

        self.slot_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
        )

        self.slot_embeddings = nn.Parameter(torch.randn(1, n_slot, hidden) * 0.02)
        self.pathology_queries = nn.Parameter(torch.randn(1, n_out, hidden) * 0.02)

        prior_matrix = torch.zeros(n_out, n_slot, dtype=torch.float32)
        for p_idx, p_name in enumerate(PATHOLOGY_NAMES):
            if p_name in SLOT_PRIOR_TABLE:
                for s_idx in SLOT_PRIOR_TABLE[p_name]:
                    if s_idx < n_slot:
                        prior_matrix[p_idx, s_idx] = prior_weight
        self.register_buffer("prior_matrix", prior_matrix)

        self.dropout = nn.Dropout(p)
        self.classifier = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(
        self, slot_feats: torch.Tensor, slot_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = slot_feats.size(0)
        k = self.slot_proj(slot_feats) + self.slot_embeddings
        v = k

        q = self.pathology_queries.expand(B, -1, -1)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.hidden)

        if self.use_prior:
            scores = scores + self.prior_matrix.unsqueeze(0)

        slot_mask_float = slot_mask.float() if slot_mask.dtype == torch.bool else slot_mask
        mask_penalty = (1.0 - slot_mask_float.unsqueeze(1)) * -10000.0
        scores = scores + mask_penalty

        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn, v)
        context = self.dropout(context)

        logits = (context * self.classifier.unsqueeze(0)).sum(dim=-1) + self.bias
        return logits, attn


class MultiWindowSlotAggregator(nn.Module):
    """Hierarchical depth aggregator pooling multi-window slice representations per slot."""

    def __init__(self, feat_dim: int = 768, num_windows: int = 3, groups_per_window: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_windows = num_windows
        self.groups_per_window = groups_per_window

        self.attn_net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N_pop, G, feat_dim) where G = num_windows * groups_per_window (e.g. 12)
        Returns:
            slot_feat: (N_pop, feat_dim)
            weights: (N_pop, num_windows, 1)
        """
        N_pop, G, D = x.shape
        x_win = x.view(N_pop, self.num_windows, self.groups_per_window, D).mean(dim=2)
        scores = self.attn_net(x_win)
        weights = F.softmax(scores, dim=1)
        slot_feat = (x_win * weights).sum(dim=1)
        return slot_feat, weights


class PhaseD7ModelViTB(nn.Module):
    """D7-A Architecture: DINOv2 ViT-B/14 Backbone + SlotHead (1,536-d features)."""

    def __init__(
        self,
        pretrained_backbone: str = "facebook/dinov2-base",
        freeze_blocks: int = 6,
        n_slot: int = 6,
        n_out: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_prior: bool = True,
        prior_weight: float = 0.55,
    ):
        super().__init__()
        self.backbone_name = pretrained_backbone
        self.backbone = Dinov2Model.from_pretrained(pretrained_backbone)
        self.dino_embed_dim = self.backbone.config.hidden_size  # 768
        self.feature_dim = self.dino_embed_dim * 2             # 1536

        # Freeze initial blocks 0 to (freeze_blocks-1)
        for param in self.backbone.parameters():
            param.requires_grad = False

        num_layers = len(self.backbone.encoder.layer)
        for i in range(freeze_blocks, num_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

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
        outputs = self.backbone(images)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        patch_mean = last_hidden[:, 1:, :].mean(dim=1)
        return torch.cat([cls_token, patch_mean], dim=-1)

    def forward(
        self, images: torch.Tensor, slot_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, G, C, H, W = images.shape
        device = images.device

        flat_mask = slot_mask.view(-1)
        populated_indices = torch.nonzero(flat_mask > 0.5, as_tuple=False).squeeze(-1)
        slot_feats_flat = torch.zeros(B * S, self.feature_dim, device=device, dtype=torch.float32)

        if len(populated_indices) > 0:
            flat_images = images.view(B * S, G, C, H, W)
            pop_images = flat_images[populated_indices]
            N_pop = pop_images.size(0)

            groups_flat = pop_images.view(N_pop * G, C, H, W)
            group_feats = self.extract_slice_features(groups_flat)
            slot_feats_pop = group_feats.view(N_pop, G, self.feature_dim).mean(dim=1)
            slot_feats_flat[populated_indices] = slot_feats_pop

        slot_feats = slot_feats_flat.view(B, S, self.feature_dim)
        logits, attn = self.head(slot_feats, slot_mask)
        return logits, attn

    def get_parameter_summary(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_percentage": round((trainable / total) * 100.0, 2),
        }


class PhaseD7ModelViTS_MultiWindow(nn.Module):
    """D7-B Architecture: DINOv2 ViT-S/14 Backbone + Multi-Window Depth Aggregator + SlotHead."""

    def __init__(
        self,
        pretrained_backbone: str = "facebook/dinov2-small",
        freeze_blocks: int = 6,
        n_slot: int = 6,
        n_out: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_prior: bool = True,
        prior_weight: float = 0.55,
        num_windows: int = 3,
        groups_per_window: int = 4,
    ):
        super().__init__()
        self.backbone_name = pretrained_backbone
        self.backbone = Dinov2Model.from_pretrained(pretrained_backbone)
        self.dino_embed_dim = self.backbone.config.hidden_size  # 384
        self.feature_dim = self.dino_embed_dim * 2             # 768

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_layers = len(self.backbone.encoder.layer)
        for i in range(freeze_blocks, num_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

        self.aggregator = MultiWindowSlotAggregator(
            feat_dim=self.feature_dim,
            num_windows=num_windows,
            groups_per_window=groups_per_window
        )

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
        outputs = self.backbone(images)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        patch_mean = last_hidden[:, 1:, :].mean(dim=1)
        return torch.cat([cls_token, patch_mean], dim=-1)

    def forward(
        self, images: torch.Tensor, slot_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, G, C, H, W = images.shape
        device = images.device

        flat_mask = slot_mask.view(-1)
        populated_indices = torch.nonzero(flat_mask > 0.5, as_tuple=False).squeeze(-1)
        slot_feats_flat = torch.zeros(B * S, self.feature_dim, device=device, dtype=torch.float32)

        if len(populated_indices) > 0:
            flat_images = images.view(B * S, G, C, H, W)
            pop_images = flat_images[populated_indices]
            N_pop = pop_images.size(0)

            groups_flat = pop_images.view(N_pop * G, C, H, W)
            group_feats = self.extract_slice_features(groups_flat)
            # Group feats: (N_pop, 12, 768)
            group_feats_reshaped = group_feats.view(N_pop, G, self.feature_dim)
            slot_feats_pop, _ = self.aggregator(group_feats_reshaped)
            slot_feats_flat[populated_indices] = slot_feats_pop

        slot_feats = slot_feats_flat.view(B, S, self.feature_dim)
        logits, attn = self.head(slot_feats, slot_mask)
        return logits, attn

    def get_parameter_summary(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_percentage": round((trainable / total) * 100.0, 2),
        }


class PhaseD7ModelViTB_MultiWindow(nn.Module):
    """D7-C Architecture: DINOv2 ViT-B/14 Backbone + Multi-Window Depth Aggregator + SlotHead."""

    def __init__(
        self,
        pretrained_backbone: str = "facebook/dinov2-base",
        freeze_blocks: int = 6,
        n_slot: int = 6,
        n_out: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_prior: bool = True,
        prior_weight: float = 0.55,
        num_windows: int = 3,
        groups_per_window: int = 4,
    ):
        super().__init__()
        self.backbone_name = pretrained_backbone
        self.backbone = Dinov2Model.from_pretrained(pretrained_backbone)
        self.dino_embed_dim = self.backbone.config.hidden_size  # 768
        self.feature_dim = self.dino_embed_dim * 2             # 1536

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_layers = len(self.backbone.encoder.layer)
        for i in range(freeze_blocks, num_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

        self.aggregator = MultiWindowSlotAggregator(
            feat_dim=self.feature_dim,
            num_windows=num_windows,
            groups_per_window=groups_per_window
        )

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
        outputs = self.backbone(images)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        patch_mean = last_hidden[:, 1:, :].mean(dim=1)
        return torch.cat([cls_token, patch_mean], dim=-1)

    def forward(
        self, images: torch.Tensor, slot_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, G, C, H, W = images.shape
        device = images.device

        flat_mask = slot_mask.view(-1)
        populated_indices = torch.nonzero(flat_mask > 0.5, as_tuple=False).squeeze(-1)
        slot_feats_flat = torch.zeros(B * S, self.feature_dim, device=device, dtype=torch.float32)

        if len(populated_indices) > 0:
            flat_images = images.view(B * S, G, C, H, W)
            pop_images = flat_images[populated_indices]
            N_pop = pop_images.size(0)

            groups_flat = pop_images.view(N_pop * G, C, H, W)
            group_feats = self.extract_slice_features(groups_flat)
            group_feats_reshaped = group_feats.view(N_pop, G, self.feature_dim)
            slot_feats_pop, _ = self.aggregator(group_feats_reshaped)
            slot_feats_flat[populated_indices] = slot_feats_pop

        slot_feats = slot_feats_flat.view(B, S, self.feature_dim)
        logits, attn = self.head(slot_feats, slot_mask)
        return logits, attn

    def get_parameter_summary(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_percentage": round((trainable / total) * 100.0, 2),
        }


class PhaseD7ModelViTB_FullCombination(nn.Module):
    """
    D7-D Architecture: Full Combined D7 Configuration.
    Combines:
    - ViT-B/14 foundation backbone (87.4M params)
    - 6-slot D6 deterministic sequence resolution
    - 12 multi-window slice groups covering 96.9% volumetric depth
    - Learned depth positional embeddings per group
    - Direct depth-attention aggregator across all 12 groups (preserving slice resolution without intra-window mean-blurring)
    - Anatomical SlotHead with pathology queries and missing-sequence masking
    """

    def __init__(
        self,
        pretrained_backbone: str = "facebook/dinov2-base",
        freeze_blocks: int = 6,
        n_slot: int = 6,
        n_out: int = 12,
        num_groups: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_prior: bool = True,
        prior_weight: float = 0.55,
    ):
        super().__init__()
        self.backbone_name = pretrained_backbone
        self.backbone = Dinov2Model.from_pretrained(pretrained_backbone)
        self.dino_embed_dim = self.backbone.config.hidden_size  # 768
        self.feature_dim = self.dino_embed_dim * 2             # 1536
        self.num_groups = num_groups

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_layers = len(self.backbone.encoder.layer)
        for i in range(freeze_blocks, num_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

        # Depth-aware positional embedding across the 12 volumetric depth groups
        self.depth_pos_embed = nn.Parameter(torch.randn(1, num_groups, self.feature_dim) * 0.02)

        # Depth attention pooling across all 12 groups directly
        self.depth_attn = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

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
        outputs = self.backbone(images)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        patch_mean = last_hidden[:, 1:, :].mean(dim=1)
        return torch.cat([cls_token, patch_mean], dim=-1)

    def forward(
        self, images: torch.Tensor, slot_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, G, C, H, W = images.shape
        device = images.device

        flat_mask = slot_mask.view(-1)
        populated_indices = torch.nonzero(flat_mask > 0.5, as_tuple=False).squeeze(-1)
        slot_feats_flat = torch.zeros(B * S, self.feature_dim, device=device, dtype=torch.float32)

        if len(populated_indices) > 0:
            flat_images = images.view(B * S, G, C, H, W)
            pop_images = flat_images[populated_indices]
            N_pop = pop_images.size(0)

            groups_flat = pop_images.view(N_pop * G, C, H, W)
            group_feats = self.extract_slice_features(groups_flat)
            group_feats_reshaped = group_feats.view(N_pop, G, self.feature_dim)

            # Depth-aware attention pooling across all 12 groups directly
            feats_with_pos = group_feats_reshaped + self.depth_pos_embed
            depth_scores = self.depth_attn(feats_with_pos)  # (N_pop, G, 1)
            depth_weights = F.softmax(depth_scores, dim=1)
            slot_feats_pop = (group_feats_reshaped * depth_weights).sum(dim=1)

            slot_feats_flat[populated_indices] = slot_feats_pop

        slot_feats = slot_feats_flat.view(B, S, self.feature_dim)
        logits, attn = self.head(slot_feats, slot_mask)
        return logits, attn

    def get_parameter_summary(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_percentage": round((trainable / total) * 100.0, 2),
        }

