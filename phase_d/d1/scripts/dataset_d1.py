"""
Phase D1: 2.5D Multi-Slice Multi-Slot Knee Abnormality Dataset.

Implements Sections 13, 15, 16, 17, 18, 19, 20, 27:
- Six canonical slots with deterministic mapping and collision resolution.
- Missing slots represented by zero tensor + slot_mask = 0.
- Central slice band: SLICE_BAND = (0.2, 0.8) (central 60% of volume).
- Central slice sampling: CACHE_SLICES = 12 evenly sampled slices.
- 2.5D group formation: Triplet grouping (slice[i-1], slice[i], slice[i+1]) -> (3, H, W).
- Physical 130-mm center crop / spatial resizing to 224x224.
- Robust percentile normalization [P1, P99] -> [0, 1] followed by ImageNet normalization.
- Frozen Phase B continuous soft targets and sample weights (zero-weighted for abstentions).
- Strict Gold-58 isolation assertion at initialization.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

NUM_SLOTS_D1 = 6
NUM_TARGETS = 12

CANONICAL_SLOTS_D1 = [
    "SAG_FLUID_FS",    # Slot 0
    "COR_FLUID_FS",    # Slot 1
    "AX_FLUID_FS",     # Slot 2
    "SAG_FLUID_NOFS",  # Slot 3
    "COR_T1",          # Slot 4
    "SAG_T1",          # Slot 5
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def extract_2p5d_groups_from_volume(
    volume: np.ndarray,
    spacing: np.ndarray,
    n_groups: int = 12,
    slice_band: Tuple[float, float] = (0.2, 0.8),
    target_resolution: Tuple[int, int] = (224, 224),
    target_fov_mm: float = 130.0,
    transform=None
) -> torch.Tensor:
    """Extract 2.5D 3-slice groups from a 3D MRI volume.
    
    Args:
        volume: 3D numpy array of shape (D, H, W).
        spacing: Voxel spacing (depth_mm, height_mm, width_mm).
        n_groups: Number of central 2.5D slice groups to sample (default 12).
        slice_band: Relative depth interval to sample from (default (0.2, 0.8)).
        target_resolution: Output (H, W) image resolution (default (224, 224)).
        target_fov_mm: Desired physical field of view in mm (default 130.0).
        transform: Optional spatial/intensity augmentation.
        
    Returns:
        Tensor of shape (n_groups, 3, target_resolution[0], target_resolution[1]).
    """
    D, H, W = volume.shape
    vol_f32 = volume.astype(np.float32)

    # 1. Percentile normalization [P1, P99] -> [0.0, 1.0] (Section 20)
    p1 = float(np.percentile(vol_f32, 1.0))
    p99 = float(np.percentile(vol_f32, 99.0))
    denom = max(p99 - p1, 1e-6)
    vol_norm = np.clip((vol_f32 - p1) / denom, 0.0, 1.0)

    # 2. Central slice sampling (Sections 15, 16)
    # Ensure slice indices leave room for triplet grouping: i-1 >= 0 and i+1 < D
    min_slice = max(1, int(np.floor(D * slice_band[0])))
    max_slice = min(D - 2, int(np.ceil(D * slice_band[1])))
    if min_slice > max_slice:
        min_slice, max_slice = 1, max(1, D - 2)

    sampled_indices = np.linspace(min_slice, max_slice, n_groups, dtype=int)

    # 3. Physical 130-mm center cropping (Section 18)
    # In-plane pixel spacing in mm
    px = float(spacing[1]) if len(spacing) > 1 and spacing[1] > 0 else 0.5
    want_px = int(round(target_fov_mm / px))
    cy, cx = H // 2, W // 2
    half = want_px // 2
    y1 = max(0, cy - half)
    y2 = min(H, cy + half)
    x1 = max(0, cx - half)
    x2 = min(W, cx + half)

    cropped_vol = vol_norm[:, y1:y2, x1:x2]

    # 4. Form 2.5D groups and normalize
    groups = []
    for idx in sampled_indices:
        # Triplet grouping: (3, H_crop, W_crop)
        grp = cropped_vol[idx - 1 : idx + 2]
        grp_t = torch.from_numpy(grp)

        # Spatial resizing to target resolution (e.g. 224x224)
        if grp_t.shape[1:] != target_resolution:
            grp_t = TF.resize(grp_t, list(target_resolution), antialias=True)

        # Apply augmentation if provided
        if transform is not None:
            grp_t = transform(grp_t)

        # ImageNet normalization
        grp_t = TF.normalize(grp_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        groups.append(grp_t)

    # Stack: (n_groups, 3, H, W)
    return torch.stack(groups, dim=0)


class PhaseD1KneeDataset(Dataset):
    """Study-level dataset for Phase D1 2.5D DINOv2 / SlotHead architecture."""

    def __init__(
        self,
        study_uids: List[str],
        slot_manifest_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        processed_dir: Optional[str] = None,
        gold_uids: Optional[set] = None,
        n_groups: int = 12,
        slice_band: Tuple[float, float] = (0.2, 0.8),
        target_resolution: Tuple[int, int] = (224, 224),
        target_fov_mm: float = 130.0,
        transform=None,
    ):
        """
        Args:
            study_uids: List of StudyInstanceUID strings for this split.
            slot_manifest_df: DataFrame generated by slot_mapping_d1.py.
            labels_df: Canonical DataFrame from phase_b_step3_remediated_training_corpus.csv.
            processed_dir: Path to processed .npz volumes.
            gold_uids: Set of Gold-58 StudyInstanceUIDs for runtime leakage assertion.
            n_groups: Number of 2.5D slice groups per slot (default 12).
            slice_band: Relative depth interval to sample from (default (0.2, 0.8)).
            target_resolution: (H, W) resolution (default (224, 224)).
            target_fov_mm: Desired physical FOV (default 130.0 mm).
            transform: Optional dynamic data augmentation.
        """
        super().__init__()
        self.processed_dir = processed_dir or os.environ.get("RSNA_PROCESSED_DIR", os.path.join(os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))), "phase_c/c1/processed"))
        self.n_groups = n_groups
        self.slice_band = slice_band
        self.target_resolution = target_resolution
        self.target_fov_mm = target_fov_mm
        self.transform = transform
        self.gold_uids = gold_uids or set()

        # Strict Gold-58 runtime leakage assertion
        overlap = set(study_uids) & self.gold_uids
        assert len(overlap) == 0, f"CRITICAL GOVERNANCE VIOLATION: {len(overlap)} Gold-58 studies detected in Phase D dataset!"

        # Index slot manifest by study
        manifest_indexed = slot_manifest_df.set_index("StudyInstanceUID")

        # Build labels map: continuous soft targets Y and sample weights W
        self.labels_map = self._build_labels_map(labels_df, study_uids)

        # Retain studies that have labels and manifest entries
        self.study_uids = [
            s for s in study_uids if s in self.labels_map and s in manifest_indexed.index
        ]
        assert len(self.study_uids) > 0, "No valid studies found in Phase D dataset."

        self.slot_manifest_indexed = manifest_indexed

    def _build_labels_map(
        self, labels_df: pd.DataFrame, study_uids: List[str]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract continuous soft targets Y and sample weights W per study."""
        labels_map = {}
        filtered = labels_df[labels_df["StudyInstanceUID"].isin(study_uids)]

        for study_uid, group in filtered.groupby("StudyInstanceUID"):
            group_targets = set(group["target"].tolist())
            missing = set(TARGETS) - group_targets
            assert len(missing) == 0, f"Study {study_uid} missing targets: {missing}"

            indexed = group.set_index("target").reindex(TARGETS)
            y = indexed["label_probability"].values.astype(np.float32)
            assert np.all((y >= 0.0) & (y <= 1.0)), f"Study {study_uid} has invalid Y values"

            w = indexed["sample_weight"].values.astype(np.float32)
            abstain = indexed["abstain_flag"].values.astype(np.int32)
            w[abstain == 1] = 0.0
            assert np.all(w >= 0.0), f"Study {study_uid} has negative sample weights"

            labels_map[study_uid] = (
                torch.from_numpy(y),
                torch.from_numpy(w),
            )
        return labels_map

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        study_uid = self.study_uids[idx]
        manifest_row = self.slot_manifest_indexed.loc[study_uid]

        # Targets and weights
        targets, weights = self.labels_map[study_uid]

        # Allocate slot tensors
        # Shape per slot: (n_groups, 3, H, W)
        # Full study tensor: (6, n_groups, 3, H, W)
        H, W = self.target_resolution
        slots_tensor = torch.zeros(
            (NUM_SLOTS_D1, self.n_groups, 3, H, W), dtype=torch.float32
        )
        slot_mask = torch.zeros(NUM_SLOTS_D1, dtype=torch.float32)

        for slot_idx, slot_name in enumerate(CANONICAL_SLOTS_D1):
            mask_val = int(manifest_row[f"mask_slot_{slot_idx}"])
            ser_uid = str(manifest_row[f"slot_{slot_idx}_{slot_name}"])

            if mask_val == 1 and ser_uid:
                npz_path = os.path.join(self.processed_dir, study_uid, f"{ser_uid}.npz")
                if os.path.exists(npz_path):
                    try:
                        data = np.load(npz_path)
                        vol = data["volume"]
                        spacing = data["spacing"]
                        slot_groups = extract_2p5d_groups_from_volume(
                            vol,
                            spacing,
                            n_groups=self.n_groups,
                            slice_band=self.slice_band,
                            target_resolution=self.target_resolution,
                            target_fov_mm=self.target_fov_mm,
                            transform=self.transform
                        )
                        slots_tensor[slot_idx] = slot_groups
                        slot_mask[slot_idx] = 1.0
                    except Exception as e:
                        # Corrupted or unreadable npz: keep slot as zero with slot_mask = 0
                        slot_mask[slot_idx] = 0.0
                else:
                    slot_mask[slot_idx] = 0.0
            else:
                # Missing slot: strictly zero tensor with slot_mask = 0 (Section 13)
                slot_mask[slot_idx] = 0.0

        return {
            "study_uid": study_uid,
            "images": slots_tensor,        # (6, 12, 3, 224, 224)
            "slot_mask": slot_mask,        # (6,)
            "targets": targets,            # (12,)
            "weights": weights,            # (12,)
        }
