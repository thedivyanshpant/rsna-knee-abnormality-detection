#!/usr/bin/env python3
"""
Phase D7: Dataset Engine Supporting Standard and Multi-Window Depth Coverage.
RSNA Knee Abnormality Detection

Strict Governance:
- Zero Gold-58 leakage assertion against csv/gold_studies.csv.
- Uses Phase D6 resolved slot mapping manifest.
- Uses Phase D3 candidate supervision corpus.
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

NUM_SLOTS = 6
NUM_TARGETS = 12

CANONICAL_SLOTS_D6 = [
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


def extract_standard_2p5d_groups(
    volume: np.ndarray,
    spacing: np.ndarray,
    n_groups: int = 12,
    slice_band: Tuple[float, float] = (0.2, 0.8),
    target_resolution: Tuple[int, int] = (224, 224),
    target_fov_mm: float = 130.0,
) -> torch.Tensor:
    """Standard D6 slice sampling: n_groups linearly spaced centers across slice_band."""
    D, H, W = volume.shape
    vol_f32 = volume.astype(np.float32)

    p1 = float(np.percentile(vol_f32, 1.0))
    p99 = float(np.percentile(vol_f32, 99.0))
    denom = max(p99 - p1, 1e-6)
    vol_norm = np.clip((vol_f32 - p1) / denom, 0.0, 1.0)

    min_slice = max(1, int(np.floor(D * slice_band[0])))
    max_slice = min(D - 2, int(np.ceil(D * slice_band[1])))
    if min_slice > max_slice:
        min_slice, max_slice = 1, max(1, D - 2)

    sampled_indices = np.linspace(min_slice, max_slice, n_groups, dtype=int)

    px = float(spacing[1]) if len(spacing) > 1 and spacing[1] > 0 else 0.5
    want_px = int(round(target_fov_mm / px))
    cy, cx = H // 2, W // 2
    half = want_px // 2
    y1 = max(0, cy - half)
    y2 = min(H, cy + half)
    x1 = max(0, cx - half)
    x2 = min(W, cx + half)
    cropped_vol = vol_norm[:, y1:y2, x1:x2]

    groups = []
    for idx in sampled_indices:
        grp = cropped_vol[idx - 1 : idx + 2]
        grp_t = torch.from_numpy(grp)
        if grp_t.shape[1:] != target_resolution:
            grp_t = TF.resize(grp_t, list(target_resolution), antialias=True)
        grp_t = TF.normalize(grp_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        groups.append(grp_t)

    return torch.stack(groups, dim=0)


def extract_multiwindow_depth_groups(
    volume: np.ndarray,
    spacing: np.ndarray,
    target_resolution: Tuple[int, int] = (224, 224),
    target_fov_mm: float = 130.0,
) -> torch.Tensor:
    """
    Multi-Window Depth Coverage sampling:
    Divides the volume into 3 overlapping anatomical depth windows across [0.08, 0.92]:
    - Win 0 (Peripheral 1 / Medial / Anterior): [0.08, 0.40] -> 4 groups
    - Win 1 (Central / Notch / Articular):       [0.35, 0.65] -> 4 groups
    - Win 2 (Peripheral 2 / Lateral / Posterior):[0.60, 0.92] -> 4 groups
    Total: 12 groups covering 87.5-96.9% of the full anatomical depth.
    """
    D, H, W = volume.shape
    vol_f32 = volume.astype(np.float32)

    p1 = float(np.percentile(vol_f32, 1.0))
    p99 = float(np.percentile(vol_f32, 99.0))
    denom = max(p99 - p1, 1e-6)
    vol_norm = np.clip((vol_f32 - p1) / denom, 0.0, 1.0)

    windows = [
        (0.08, 0.40, 4),
        (0.35, 0.65, 4),
        (0.60, 0.92, 4)
    ]
    sampled_indices = []
    for w_start, w_end, n_grp in windows:
        min_s = max(1, int(np.floor(D * w_start)))
        max_s = min(D - 2, int(np.ceil(D * w_end)))
        if min_s > max_s:
            min_s, max_s = 1, max(1, D - 2)
        idx = np.linspace(min_s, max_s, n_grp, dtype=int)
        sampled_indices.extend(idx.tolist())

    px = float(spacing[1]) if len(spacing) > 1 and spacing[1] > 0 else 0.5
    want_px = int(round(target_fov_mm / px))
    cy, cx = H // 2, W // 2
    half = want_px // 2
    y1 = max(0, cy - half)
    y2 = min(H, cy + half)
    x1 = max(0, cx - half)
    x2 = min(W, cx + half)
    cropped_vol = vol_norm[:, y1:y2, x1:x2]

    groups = []
    for idx in sampled_indices:
        grp = cropped_vol[idx - 1 : idx + 2]
        grp_t = torch.from_numpy(grp)
        if grp_t.shape[1:] != target_resolution:
            grp_t = TF.resize(grp_t, list(target_resolution), antialias=True)
        grp_t = TF.normalize(grp_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        groups.append(grp_t)

    return torch.stack(groups, dim=0)


class PhaseD7KneeDataset(Dataset):
    """Dataset supporting standard D6 sampling and Multi-Window Depth Coverage."""

    def __init__(
        self,
        study_uids: List[str],
        slot_manifest_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        processed_dir: str,
        gold_uids: set,
        multiwindow: bool = False,
        target_resolution: Tuple[int, int] = (224, 224),
        target_fov_mm: float = 130.0,
    ):
        super().__init__()
        self.study_uids = list(study_uids)
        self.processed_dir = processed_dir
        self.multiwindow = multiwindow
        self.target_resolution = target_resolution
        self.target_fov_mm = target_fov_mm

        # Zero Gold leakage assertion
        leakage = set(self.study_uids) & gold_uids
        assert len(leakage) == 0, f"CRITICAL GOVERNANCE LEAKAGE: {leakage}"

        # Index slot manifest
        self.slot_manifest_indexed = slot_manifest_df.set_index("StudyInstanceUID")

        # Index labels
        labels_sub = labels_df[labels_df["StudyInstanceUID"].isin(self.study_uids)].copy()
        self.labels_map = {}
        for uid, grp in labels_sub.groupby("StudyInstanceUID"):
            grp_indexed = grp.set_index("target").reindex(TARGETS)
            y = grp_indexed["label_probability"].values.astype(np.float32)
            w = grp_indexed["sample_weight"].values.astype(np.float32)
            if "abstain_flag" in grp_indexed.columns:
                abstain = grp_indexed["abstain_flag"].values.astype(np.int32)
                w[abstain == 1] = 0.0
            self.labels_map[uid] = (torch.from_numpy(y), torch.from_numpy(w))

        # Retain studies that have labels and manifest entries
        self.study_uids = [
            s for s in self.study_uids if s in self.labels_map and s in self.slot_manifest_indexed.index
        ]
        assert len(self.study_uids) > 0, "No valid studies found in Phase D7 dataset."

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        study_uid = self.study_uids[idx]
        manifest_row = self.slot_manifest_indexed.loc[study_uid]
        targets, weights = self.labels_map[study_uid]

        H, W = self.target_resolution
        n_groups = 12
        slots_tensor = torch.zeros((NUM_SLOTS, n_groups, 3, H, W), dtype=torch.float32)
        slot_mask = torch.zeros(NUM_SLOTS, dtype=torch.float32)

        for slot_idx, slot_name in enumerate(CANONICAL_SLOTS_D6):
            mask_val = int(manifest_row[f"mask_slot_{slot_idx}"])
            col_name = f"slot_{slot_idx}_{slot_name}"
            ser_uid = str(manifest_row[col_name]) if col_name in manifest_row and pd.notna(manifest_row[col_name]) else None

            if mask_val == 1 and ser_uid and ser_uid != "nan":
                npz_path = os.path.join(self.processed_dir, study_uid, f"{ser_uid}.npz")
                if os.path.exists(npz_path):
                    try:
                        data = np.load(npz_path)
                        vol = data["volume"]
                        spacing = data["spacing"]

                        if self.multiwindow:
                            grp_tensor = extract_multiwindow_depth_groups(
                                vol, spacing,
                                target_resolution=self.target_resolution,
                                target_fov_mm=self.target_fov_mm
                            )
                        else:
                            grp_tensor = extract_standard_2p5d_groups(
                                vol, spacing,
                                n_groups=12,
                                slice_band=(0.2, 0.8),
                                target_resolution=self.target_resolution,
                                target_fov_mm=self.target_fov_mm
                            )

                        slots_tensor[slot_idx] = grp_tensor
                        slot_mask[slot_idx] = 1.0
                    except Exception:
                        slot_mask[slot_idx] = 0.0

        return {
            "study_uid": study_uid,
            "images": slots_tensor,
            "slot_mask": slot_mask,
            "targets": targets,
            "weights": weights,
        }
