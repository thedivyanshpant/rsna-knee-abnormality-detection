#!/usr/bin/env python3
"""
Phase D8 Architectural Suite: Representation & Formulation Innovations.
RSNA Knee Abnormality Detection

Models:
- D8-A: PhaseD8Model_PathologyCrossAttention (ViT-S/14 + Direct Pathology-to-Visual-Token Cross-Attention)
- D8-B: PhaseD8Model_HighBudgetDepth (ViT-S/14 + 24 High-Budget Volumetric Depth Groups)
- D8-C: PhaseD8Model_HybridConvAttn (CoAtNet-Style Local Depthwise Conv + ViT Transformer Blocks)
- D8-D: PhaseD8Model_FullCombination (Hybrid Backbone + 24 High-Budget Groups + Pathology Cross-Attention)
"""

import math
from typing import Dict, List, Tuple, Any, Optional
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
NUM_PATHOLOGIES = len(PATHOLOGY_NAMES)
NUM_SLOTS = 6

# Canonical D6/D8 Clinical Slot Priors (12 Pathologies x 6 Slots)
CLINICAL_PRIORS_D8 = [
    #  S0(SAG_FL_FS) S1(COR_FL_FS) S2(AX_FL_FS) S3(SAG_FL_NOFS) S4(COR_T1) S5(SAG_T1)
    [  1.0,          0.2,          0.1,         0.55,           0.0,       0.3  ], # ACL
    [  0.1,          1.0,          0.2,         0.1,            0.6,       0.0  ], # MCL
    [  0.9,          0.7,          0.0,         0.55,           0.5,       0.4  ], # Medial Meniscus
    [  0.9,          0.7,          0.0,         0.55,           0.5,       0.4  ], # Lateral Meniscus
    [  0.3,          0.8,          0.1,         0.2,            0.8,       0.2  ], # Medial OA
    [  0.3,          0.8,          0.1,         0.2,            0.8,       0.2  ], # Lateral OA
    [  0.2,          0.2,          0.9,         0.2,            0.0,       0.2  ], # PF OA
    [  0.7,          0.6,          0.8,         0.4,            0.0,       0.2  ], # Effusion
    [  0.7,          0.5,          0.8,         0.4,            0.0,       0.2  ], # Synovitis
    [  0.9,          0.3,          0.7,         0.55,           0.0,       0.3  ], # Baker's
    [  0.7,          0.7,          0.3,         0.2,            0.2,       0.1  ], # Contusion
    [  0.5,          0.5,          0.4,         0.3,            0.4,       0.3  ], # Fracture
]


# ==============================================================================
# Model D8-A: Pathology-Conditioned Cross-Attention
# ==============================================================================

class PathologyCrossAttentionHead(nn.Module):
    """
    Direct Pathology-Conditioned Multi-Head Cross-Attention Head.
    Allows each of the 12 pathology queries to directly attend across all (S x G) visual
    slice tokens, modulated by anatomical slot priors and slot-presence masks.
    """

    def __init__(
        self,
        feat_dim: int = 768,
        n_queries: int = 12,
        n_slot: int = 6,
        groups_per_slot: int = 12,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        prior_weight: float = 0.55,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_queries = n_queries
        self.n_slot = n_slot
        self.groups_per_slot = groups_per_slot
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.prior_weight = prior_weight

        # 12 Pathology Queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden_dim) * 0.02)

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(feat_dim, hidden_dim)
        self.v_proj = nn.Linear(feat_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Prior matrix broadcast across (n_slot * groups_per_slot)
        priors_raw = torch.tensor(CLINICAL_PRIORS_D8, dtype=torch.float32) * prior_weight  # (12, 6)
        priors_expanded = priors_raw.unsqueeze(2).expand(12, 6, groups_per_slot).contiguous().view(12, 6 * groups_per_slot)
        self.register_buffer("prior_tokens", priors_expanded)  # (12, 72)

        # Slot-to-token indexing for masking
        slot_indices = torch.arange(n_slot).unsqueeze(1).expand(n_slot, groups_per_slot).contiguous().view(-1)
        self.register_buffer("slot_indices", slot_indices)  # (72,)

        # Pathology classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        token_feats: torch.Tensor,
        slot_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_feats: (B, S * G, feat_dim) visual tokens from all slots and slice groups
            slot_mask: (B, S) float mask (1.0 = present, 0.0 = missing)
        Returns:
            logits: (B, 12)
            attn_weights: (B, 12, S * G)
        """
        B, N_tok, _ = token_feats.shape
        H = self.num_heads
        D_h = self.head_dim

        # Project queries
        q = self.queries.expand(B, -1, -1)  # (B, 12, hidden_dim)
        q = self.q_proj(q).view(B, self.n_queries, H, D_h).transpose(1, 2)  # (B, H, 12, D_h)

        # Project keys and values
        k = self.k_proj(token_feats).view(B, N_tok, H, D_h).transpose(1, 2)  # (B, H, N_tok, D_h)
        v = self.v_proj(token_feats).view(B, N_tok, H, D_h).transpose(1, 2)  # (B, H, N_tok, D_h)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D_h)  # (B, H, 12, N_tok)

        # Add anatomical priors (broadcast across heads)
        scores = scores + self.prior_tokens.unsqueeze(0).unsqueeze(1)

        # Build token-level mask from slot_mask: (B, N_tok)
        token_mask = slot_mask[:, self.slot_indices]  # (B, 72)
        mask_penalty = (1.0 - token_mask).unsqueeze(1).unsqueeze(2) * -10000.0  # (B, 1, 1, 72)
        scores = scores + mask_penalty

        attn = F.softmax(scores, dim=-1)  # (B, H, 12, N_tok)
        context = torch.matmul(attn, v)   # (B, H, 12, D_h)
        context = context.transpose(1, 2).contiguous().view(B, self.n_queries, self.hidden_dim)
        context = self.out_proj(context)  # (B, 12, hidden_dim)

        logits = self.classifier(context).squeeze(-1)  # (B, 12)
        attn_mean = attn.mean(dim=1)                   # (B, 12, N_tok)
        return logits, attn_mean


class PhaseD8Model_PathologyCrossAttention(nn.Module):
    """
    D8-A: ViT-S/14 with Pathology-Conditioned Direct Token Cross-Attention.
    Preserves ViT-S backbone while allowing pathology queries to directly search across all
    available spatial visual tokens with clinical priors and missing-slot masking.
    """

    def __init__(
        self,
        pretrained_backbone: str = "facebook/dinov2-small",
        freeze_blocks: int = 6,
        n_slot: int = 6,
        n_out: int = 12,
        groups_per_slot: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        prior_weight: float = 0.55,
    ):
        super().__init__()
        self.backbone_name = pretrained_backbone
        self.backbone = Dinov2Model.from_pretrained(pretrained_backbone)
        self.dino_embed_dim = self.backbone.config.hidden_size  # 384
        self.feature_dim = self.dino_embed_dim * 2             # 768
        self.n_slot = n_slot
        self.groups_per_slot = groups_per_slot

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_layers = len(self.backbone.encoder.layer)
        for i in range(freeze_blocks, num_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

        self.head = PathologyCrossAttentionHead(
            feat_dim=self.feature_dim,
            n_queries=n_out,
            n_slot=n_slot,
            groups_per_slot=groups_per_slot,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
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
        tokens_flat = torch.zeros(B * S, G, self.feature_dim, device=device, dtype=torch.float32)

        if len(populated_indices) > 0:
            flat_images = images.view(B * S, G, C, H, W)
            pop_images = flat_images[populated_indices]
            N_pop = pop_images.size(0)

            groups_flat = pop_images.view(N_pop * G, C, H, W)
            group_feats = self.extract_slice_features(groups_flat)
            tokens_flat[populated_indices] = group_feats.view(N_pop, G, self.feature_dim)

        # Merge slots into unified token sequence: (B, S * G, feat_dim)
        study_tokens = tokens_flat.view(B, S * G, self.feature_dim)
        logits, attn = self.head(study_tokens, slot_mask)
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


# ==============================================================================
# Model D8-B: High-Budget Multi-Window Representation (24 Groups)
# ==============================================================================

class HighBudgetDepthAggregator(nn.Module):
    """
    Depth aggregator designed for 24 groups:
    - 4 groups: Peripheral Medial
    - 16 groups: Dense Central Articular Core (preserves central joint representation)
    - 4 groups: Peripheral Lateral
    Applies hierarchical attention pooling across the 3 anatomical regions.
    """

    def __init__(self, feat_dim: int = 768):
        super().__init__()
        self.feat_dim = feat_dim
        # Attention across the 16 central core groups
        self.central_attn = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # Attention across the 3 anatomical depth regions (medial, central, lateral)
        self.region_attn = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (N_pop, 24, D)
        N_pop, G, D = x.shape
        medial = x[:, 0:4, :].mean(dim=1)   # (N_pop, D)

        # Dense central core attention pooling
        central_slices = x[:, 4:20, :]       # (N_pop, 16, D)
        c_scores = self.central_attn(central_slices) # (N_pop, 16, 1)
        c_weights = F.softmax(c_scores, dim=1)
        central = (central_slices * c_weights).sum(dim=1) # (N_pop, D)

        lateral = x[:, 20:24, :].mean(dim=1) # (N_pop, D)

        # Stack the 3 anatomical regions
        regions = torch.stack([medial, central, lateral], dim=1) # (N_pop, 3, D)
        r_scores = self.region_attn(regions)
        r_weights = F.softmax(r_scores, dim=1)
        slot_feat = (regions * r_weights).sum(dim=1) # (N_pop, D)

        return slot_feat, r_weights


class PhaseD8Model_HighBudgetDepth(nn.Module):
    """D8-B: ViT-S/14 with 24 High-Budget Multi-Window Groups."""

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
    ):
        super().__init__()
        from phase_d.d7.models.model_d7 import SlotHead
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

        self.aggregator = HighBudgetDepthAggregator(feat_dim=self.feature_dim)

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


# ==============================================================================
# Model D8-C: Hybrid Convolution + Attention (CoAtNet Style)
# ==============================================================================

class LocalDepthwiseConvStem(nn.Module):
    """
    CoAtNet-Style Local Depthwise Convolution Feature Extractor.
    Extracts fine-grained spatial bone/cartilage edges using large 7x7 depthwise kernels
    and inverted bottleneck expansion with Squeeze-and-Excitation.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.GELU()

        # Depthwise 7x7
        self.dw = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.GELU()

        # Pointwise projection
        self.pw = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.dw(x)))
        x = self.bn3(self.pw(x))
        return x


class PhaseD8Model_HybridConvAttn(nn.Module):
    """
    D8-C: CoAtNet-Style Hybrid Convolution + Attention Architecture.
    Combines local depthwise convolution spatial feature extraction with ViT self-attention
    and SlotHead contextual sequence reasoning.
    """

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
    ):
        super().__init__()
        from phase_d.d7.models.model_d7 import SlotHead
        self.backbone_name = pretrained_backbone
        self.backbone = Dinov2Model.from_pretrained(pretrained_backbone)
        self.dino_embed_dim = self.backbone.config.hidden_size  # 384

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_layers = len(self.backbone.encoder.layer)
        for i in range(freeze_blocks, num_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

        # Local Depthwise Conv Stem
        self.conv_stem = LocalDepthwiseConvStem(in_channels=3, out_channels=64)
        self.conv_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_proj = nn.Linear(64, self.dino_embed_dim)

        # Combined feature dim = ViT (768) + Conv (384) = 1152
        self.feature_dim = self.dino_embed_dim * 2 + self.dino_embed_dim

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
        # ViT Global Tokens
        outputs = self.backbone(images)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        patch_mean = last_hidden[:, 1:, :].mean(dim=1)
        vit_feat = torch.cat([cls_token, patch_mean], dim=-1)  # (N, 768)

        # Local Conv Features
        conv_out = self.conv_stem(images)                      # (N, 64, 112, 112)
        conv_pooled = self.conv_pool(conv_out).squeeze(-1).squeeze(-1) # (N, 64)
        conv_feat = self.conv_proj(conv_pooled)                # (N, 384)

        return torch.cat([vit_feat, conv_feat], dim=-1)        # (N, 1152)

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
            slot_feats_pop = group_feats_reshaped.mean(dim=1)
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


# ==============================================================================
# Model D8-D: Full Combined D8 Architecture
# ==============================================================================

class PhaseD8Model_FullCombination(nn.Module):
    """
    D8-D: Full Combined D8 Configuration.
    Combines:
    - Hybrid Local-Global Feature Extractor (ConvNeXt-style stem + ViT-S/14 backbone)
    - 24 High-Budget Multi-Window Slice Groups (Dense Central Core + Peripheral Margins)
    - Pathology-Conditioned Direct Token Cross-Attention with Clinical Priors and Slot Masking
    """

    def __init__(
        self,
        pretrained_backbone: str = "facebook/dinov2-small",
        freeze_blocks: int = 6,
        n_slot: int = 6,
        n_out: int = 12,
        groups_per_slot: int = 24,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        prior_weight: float = 0.55,
    ):
        super().__init__()
        self.backbone_name = pretrained_backbone
        self.backbone = Dinov2Model.from_pretrained(pretrained_backbone)
        self.dino_embed_dim = self.backbone.config.hidden_size  # 384
        self.groups_per_slot = groups_per_slot

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_layers = len(self.backbone.encoder.layer)
        for i in range(freeze_blocks, num_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

        # Local Depthwise Conv Stem
        self.conv_stem = LocalDepthwiseConvStem(in_channels=3, out_channels=64)
        self.conv_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_proj = nn.Linear(64, self.dino_embed_dim)

        self.feature_dim = self.dino_embed_dim * 2 + self.dino_embed_dim  # 1152

        # Pathology-Conditioned Cross-Attention Head for 24 groups
        self.head = PathologyCrossAttentionHead(
            feat_dim=self.feature_dim,
            n_queries=n_out,
            n_slot=n_slot,
            groups_per_slot=groups_per_slot,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            prior_weight=prior_weight,
        )

    def extract_slice_features(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(images)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        patch_mean = last_hidden[:, 1:, :].mean(dim=1)
        vit_feat = torch.cat([cls_token, patch_mean], dim=-1)

        conv_out = self.conv_stem(images)
        conv_pooled = self.conv_pool(conv_out).squeeze(-1).squeeze(-1)
        conv_feat = self.conv_proj(conv_pooled)

        return torch.cat([vit_feat, conv_feat], dim=-1)

    def forward(
        self, images: torch.Tensor, slot_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, G, C, H, W = images.shape
        device = images.device

        flat_mask = slot_mask.view(-1)
        populated_indices = torch.nonzero(flat_mask > 0.5, as_tuple=False).squeeze(-1)
        tokens_flat = torch.zeros(B * S, G, self.feature_dim, device=device, dtype=torch.float32)

        if len(populated_indices) > 0:
            flat_images = images.view(B * S, G, C, H, W)
            pop_images = flat_images[populated_indices]
            N_pop = pop_images.size(0)

            groups_flat = pop_images.view(N_pop * G, C, H, W)
            group_feats = self.extract_slice_features(groups_flat)
            tokens_flat[populated_indices] = group_feats.view(N_pop, G, self.feature_dim)

        study_tokens = tokens_flat.view(B, S * G, self.feature_dim)
        logits, attn = self.head(study_tokens, slot_mask)
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
