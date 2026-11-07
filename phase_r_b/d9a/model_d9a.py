#!/usr/bin/env python3
"""
Phase D9-A Model Architecture: Full-Depth Hybrid Convolution + Attention (CoAtNet Style).
RSNA Knee Abnormality Detection

Preserves the exact D8-C hybrid architecture:
- DINOv2 ViT-S/14 backbone (Blocks 0-5 frozen, 6-11 trainable)
- Local Depthwise Convolution Stem (CoAtNet-style, 7x7 depthwise kernels, GELU, BN)
- Combined feature representation: ViT CLS + patch mean (768) + Conv features (384) = 1152 dim
- Full 12-group depth aggregation across 32-slice volume
- SlotHead multi-slot contextual reasoning with clinical prior (+0.55) across 6 slots
Total Parameters: 22,391,436 (Trainable: 10,987,020, 49.07%)
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase_d.d8.models.model_d8 import (
    PhaseD8Model_HybridConvAttn,
    LocalDepthwiseConvStem,
    PATHOLOGY_NAMES
)

class PhaseD9AModel_FullDepthHybrid(PhaseD8Model_HybridConvAttn):
    """
    D9-A Full-Depth Hybrid Convolution + Attention Model.
    Inherits identical parameters, forward pass, and head from PhaseD8Model_HybridConvAttn.
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
        super().__init__(
            pretrained_backbone=pretrained_backbone,
            freeze_blocks=freeze_blocks,
            n_slot=n_slot,
            n_out=n_out,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_prior=use_prior,
            prior_weight=prior_weight,
        )
