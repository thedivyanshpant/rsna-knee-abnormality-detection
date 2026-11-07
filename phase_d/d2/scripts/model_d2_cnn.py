"""
Phase D2: RadImageNet ResNet-50 + Anatomical SlotHead Architecture.

Implements the CNN-based visual representation specified in Phase D2:
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

NUM_SLOTS_D2 = 6
NUM_TARGETS = 12

# Clinical Inductive Prior Table (Identical to Phase D1)
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
        self.num_slots = NUM_SLOTS_D2
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
            print(f"[MODEL D2] Successfully loaded RadImageNet weights from: {radimagenet_ckpt_path}")
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
            print("[MODEL D2] Frozen lower blocks: conv1, bn1, layer1, layer2.")
            print("[MODEL D2] Trainable blocks: layer3, layer4.")

        # 4. Slot projection: maps 2048-d ResNet feature to 256-d slot embedding
        self.slot_proj = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, slot_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 5. SlotHead Multi-Query Cross-Attention Components
        self.pathology_queries = nn.Parameter(torch.randn(1, self.num_targets, slot_dim) * 0.02)
        self.slot_key_proj = nn.Linear(slot_dim, slot_dim)
        self.slot_val_proj = nn.Linear(slot_dim, slot_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Parameter(torch.randn(self.num_targets, slot_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_targets))

        # 6. Clinical Inductive Prior Bias Matrix: shape (12, 6)
        target_names = list(SLOT_PRIOR_TABLE.keys())
        prior_matrix = torch.zeros(self.num_targets, self.num_slots)
        for t_idx, t_name in enumerate(target_names):
            preferred_slots = SLOT_PRIOR_TABLE.get(t_name, [])
            for s_idx in preferred_slots:
                prior_matrix[t_idx, s_idx] = self.prior_weight
        self.register_buffer("prior_matrix", prior_matrix)

    def extract_slice_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pass slice triplets through ResNet-50 backbone.
        
        Args:
            x: (N, 3, 224, 224) slice triplets.
        Returns:
            features: (N, 2048) pooled feature vectors.
        """
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f).flatten(1)  # (N, 2048)
        return f

    def forward(
        self, images: torch.Tensor, slot_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for multi-slot knee study.
        
        Args:
            images: (B, 6, 12, 3, 224, 224) multi-slot 2.5D study tensor.
            slot_mask: (B, 6) slot presence mask (1.0 for present, 0.0 for missing).
            
        Returns:
            logits: (B, 12) predicted study-level logits.
            attn: (B, 12, 6) cross-attention weights over slots.
        """
        B, S, G, C, H, W = images.shape
        device = images.device

        flat_mask = slot_mask.view(-1)  # (B * S)
        pop_idx = torch.nonzero(flat_mask > 0.5, as_tuple=False).squeeze(-1)

        slot_features = torch.zeros(B * S, 2048, device=device, dtype=images.dtype)
        if len(pop_idx) > 0:
            flat_x = images.view(B * S, G, C, H, W)[pop_idx]  # (N_pop, 12, 3, 224, 224)
            flat_x = flat_x.view(-1, C, H, W)                 # (N_pop * 12, 3, 224, 224)
            
            slice_feats = self.extract_slice_features(flat_x) # (N_pop * 12, 2048)
            # Pool across 12 groups per slot
            group_pooled = slice_feats.view(len(pop_idx), G, 2048).mean(dim=1)  # (N_pop, 2048)
            if slot_features.dtype != group_pooled.dtype:
                slot_features = slot_features.to(group_pooled.dtype)
            slot_features[pop_idx] = group_pooled

        # Project to slot embeddings: (B, 6, 256)
        slot_embeds = self.slot_proj(slot_features).view(B, S, self.slot_dim)

        # Multi-query cross-attention
        k = self.slot_key_proj(slot_embeds)  # (B, 6, 256)
        v = self.slot_val_proj(slot_embeds)  # (B, 6, 256)
        q = self.pathology_queries.expand(B, -1, -1)  # (B, 12, 256)

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.slot_dim ** 0.5)  # (B, 12, 6)

        # Apply clinical inductive prior bias
        if self.use_prior:
            scores = scores + self.prior_matrix.unsqueeze(0).to(scores.dtype)

        # Apply missing-slot penalty (-10000.0)
        mask_penalty = (1.0 - slot_mask.unsqueeze(1)) * -10000.0
        scores = scores + mask_penalty.to(scores.dtype)

        attn = F.softmax(scores, dim=-1)  # (B, 12, 6)
        context = torch.bmm(attn, v)      # (B, 12, 256)
        context = self.dropout(context)

        # Final pathology logits
        logits = (context * self.classifier.unsqueeze(0)).sum(dim=-1) + self.bias  # (B, 12)
        return logits, attn
