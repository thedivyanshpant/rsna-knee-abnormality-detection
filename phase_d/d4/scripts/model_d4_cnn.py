"""
Phase D4: RadImageNet ResNet-50 + Anatomical SlotHead Architecture.
RSNA Knee Abnormality Detection

Identical architecture to Phase D2:
- Backbone: ResNet-50 pretrained on RadImageNet (Lab-Rasool/RadImageNet).
- Input: Governed 2.5D slice groups (6 slots, 12 groups of 3 slices, 224x224).
- Freezing schedule: Lower layers (conv1, bn1, layer1, layer2) frozen;
  higher layers (layer3, layer4) trainable.
- Slot-level aggregation: Group pooling (mean over 12 groups) -> 2048-d slot vector.
- Slot projection: LayerNorm(2048) -> Linear(2048, 256) -> GELU() -> Dropout(0.2).
- Anatomical SlotHead: 12 learnable pathology queries, +0.55 clinical inductive priors,
  and -10,000.0 missing-slot attention penalty.
- Output: 12 pathology logits.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

NUM_SLOTS_D4 = 6
NUM_TARGETS = 12

# Clinical Inductive Prior Table (Identical to Phase D1/D2)
# Maps each pathology query to its primary diagnostic slots with +0.55 logit bias
SLOT_PRIOR_TABLE = {
    "ACL": [0],                  # SAG_FLUID_FS
    "MCL": [1],                  # COR_FLUID_FS
    "Medial Meniscus": [0],      # SAG_FLUID_FS
    "Lateral Meniscus": [0],     # SAG_FLUID_FS
    "Medial OA": [1],            # COR_FLUID_FS
    "Lateral OA": [1],           # COR_FLUID_FS
    "PF OA": [2],                # AX_FLUID_FS
    "Effusion": [2],             # AX_FLUID_FS
    "Synovitis": [2],            # AX_FLUID_FS
    "Baker's": [2],              # AX_FLUID_FS
    "Contusion": [1],            # COR_FLUID_FS
    "Fracture": [4, 5],          # COR_T1, SAG_T1
}

TARGET_NAMES = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

RADIMAGENET_CHECKPOINT_PATH = os.environ.get(
    "RADIMAGENET_CHECKPOINT_PATH",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Lab-Rasool--RadImageNet/"
        "snapshots/14460ee4c1276f6925611a63aa9a54a05d39eae0/ResNet50.pt"
    )
)


class RadImageNetResNet50SlotHead(nn.Module):
    """RadImageNet ResNet-50 Backbone with Anatomical SlotHead."""

    def __init__(
        self,
        radimagenet_ckpt_path: str = RADIMAGENET_CHECKPOINT_PATH,
        freeze_lower_blocks: bool = True,
        slot_dim: int = 256,
        dropout: float = 0.2,
        prior_weight: float = 0.55,
        use_prior: bool = True,
    ):
        super().__init__()
        self.num_slots = NUM_SLOTS_D4
        self.num_targets = NUM_TARGETS
        self.slot_dim = slot_dim
        self.prior_weight = prior_weight
        self.use_prior = use_prior

        # 1. Base ResNet-50 architecture
        base_model = models.resnet50(weights=None)
        
        # Load RadImageNet pretrained weights
        if os.path.exists(radimagenet_ckpt_path):
            ckpt = torch.load(radimagenet_ckpt_path, map_location="cpu")
            mapped = {k.replace("backbone.", ""): v for k, v in ckpt.items()}
            seq = nn.Sequential(*list(base_model.children())[:-1])  # conv1 to avgpool
            seq.load_state_dict(mapped, strict=True)
            print(f"[MODEL D4] Successfully loaded RadImageNet weights from: {radimagenet_ckpt_path}")
        else:
            raise FileNotFoundError(f"RadImageNet checkpoint not found at: {radimagenet_ckpt_path}")

        # 2. Decompose into modules for selective freezing
        self.conv1 = seq[0]
        self.bn1 = seq[1]
        self.relu = seq[2]
        self.maxpool = seq[3]
        self.layer1 = seq[4]
        self.layer2 = seq[5]
        self.layer3 = seq[6]
        self.layer4 = seq[7]
        self.avgpool = seq[8]  # (B, 2048, 1, 1)

        # 3. Freezing schedule: freeze conv1, bn1, layer1, layer2
        if freeze_lower_blocks:
            frozen_modules = [self.conv1, self.bn1, self.layer1, self.layer2]
            for mod in frozen_modules:
                for param in mod.parameters():
                    param.requires_grad = False
            print("[MODEL D4] Frozen lower blocks: conv1, bn1, layer1, layer2.")
            print("[MODEL D4] Trainable blocks: layer3, layer4.")

        # 4. Slot Projection: LayerNorm(2048) -> Linear(2048, 256) -> GELU() -> Dropout(0.2)
        self.slot_norm = nn.LayerNorm(2048)
        self.slot_proj = nn.Sequential(
            nn.Linear(2048, slot_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 5. Anatomical SlotHead: 12 Pathology Queries (12, 256)
        self.pathology_queries = nn.Parameter(torch.randn(self.num_targets, slot_dim) * 0.02)

        # Cross-attention projections
        self.q_proj = nn.Linear(slot_dim, slot_dim)
        self.k_proj = nn.Linear(slot_dim, slot_dim)
        self.v_proj = nn.Linear(slot_dim, slot_dim)

        # Pathology classifier heads (independent linear classifier per target)
        self.classifier = nn.Linear(slot_dim, 1)

        # 6. Build clinical prior bias matrix (12, 6)
        prior_matrix = torch.zeros(self.num_targets, self.num_slots)
        if self.use_prior:
            for t_idx, t_name in enumerate(TARGET_NAMES):
                if t_name in SLOT_PRIOR_TABLE:
                    preferred_slots = SLOT_PRIOR_TABLE[t_name]
                    for s in preferred_slots:
                        if s < self.num_slots:
                            prior_matrix[t_idx, s] = self.prior_weight
        self.register_buffer("prior_bias", prior_matrix)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-50 backbone up to global average pool."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 2048)

    def forward(
        self,
        images: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.
        Args:
            images: (B, num_slots, num_groups, 3, 224, 224)
            slot_mask: (B, num_slots) float tensor: 1.0 for valid slot, 0.0 for missing
        Returns:
            logits: (B, 12) pathology logits
            aux: auxiliary attention weights dictionary
        """
        B, S, G, C, H, W = images.shape
        assert S == self.num_slots, f"Expected {self.num_slots} slots, got {S}"

        # Flatten batch, slot, group dimensions for CNN backbone forward pass
        images_flat = images.view(B * S * G, C, H, W)
        features_flat = self.extract_features(images_flat)  # (B*S*G, 2048)

        # Reshape to (B, S, G, 2048) and pool across groups -> (B, S, 2048)
        features = features_flat.view(B, S, G, 2048)
        slot_features = features.mean(dim=2)  # Group mean pooling -> (B, S, 2048)

        # Slot projection -> (B, S, slot_dim)
        slot_emb = self.slot_norm(slot_features)
        slot_emb = self.slot_proj(slot_emb)  # (B, S, 256)

        # Queries: expand to (B, 12, slot_dim)
        Q = self.q_proj(self.pathology_queries).unsqueeze(0).expand(B, -1, -1)  # (B, 12, 256)
        K = self.k_proj(slot_emb)                                              # (B, 6, 256)
        V = self.v_proj(slot_emb)                                              # (B, 6, 256)

        # Scaled dot-product attention scores
        scale = 1.0 / (self.slot_dim ** 0.5)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * scale  # (B, 12, 6)

        # Add clinical inductive prior bias (12, 6)
        attn_scores = attn_scores + self.prior_bias.unsqueeze(0)

        # Mask missing slots with -10,000.0 penalty
        mask_expanded = slot_mask.unsqueeze(1)  # (B, 1, 6)
        attn_scores = attn_scores.masked_fill(mask_expanded == 0.0, -10000.0)

        # Softmax over slots
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 12, 6)

        # Context vectors: (B, 12, 256)
        context = torch.bmm(attn_weights, V)  # (B, 12, 256)

        # Logits per pathology: (B, 12, 1) -> (B, 12)
        logits = self.classifier(context).squeeze(-1)

        return logits, {"attn_weights": attn_weights}
