"""
Phase C3: Study-Level Multi-Series PyTorch Dataset.

Maps:
StudyInstanceUID
  -> Discovers valid preprocessed .npz series in phase_c/c1/processed/<StudyInstanceUID>/
  -> Maps series into 5 canonical anatomical/contrast slots:
       Slot 0: Sagittal Fluid-Sensitive (FS)
       Slot 1: Coronal Fluid-Sensitive (FS)
       Slot 2: Axial Fluid-Sensitive (FS)
       Slot 3: Sagittal Non-FS / T1
       Slot 4: Coronal Non-FS / T1
  -> Enforces strict deterministic slot assignment from series metadata
     (HALTS if metadata is absent; strictly forbids arbitrary filename classification)
  -> Deterministic collision resolution policy with provenance logging
  -> Missing slots: zero-filled tensors with slot_mask[s] = 0.0
  -> Phase B soft targets: 12 continuous Y values, 12 sample weights W (W=0 for abstentions)
"""
import os
import glob
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_SLOTS = 5
NUM_TARGETS = 12
TENSOR_SHAPE = (32, 256, 256)

CANONICAL_SLOTS = [
    "sagittal_fluid_sensitive",  # Slot 0
    "coronal_fluid_sensitive",   # Slot 1
    "axial_fluid_sensitive",     # Slot 2
    "sagittal_non_fs_t1",        # Slot 3
    "coronal_non_fs_t1",         # Slot 4
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


class InsufficientMetadataError(RuntimeError):
    """Raised when series plane/contrast metadata is insufficient for deterministic classification."""
    pass


def classify_series_slot(plane: str, fluid_sensitive: int) -> Optional[int]:
    """Deterministically map Anatomical_Plane and Fluid_Sensitive tag to slot index 0..4.

    Slot 0: Sagittal FS (Plane=Sagittal, Fluid=1)
    Slot 1: Coronal FS  (Plane=Coronal, Fluid=1)
    Slot 2: Axial FS    (Plane=Axial, Fluid=1)
    Slot 3: Sagittal T1 (Plane=Sagittal, Fluid=0)
    Slot 4: Coronal T1  (Plane=Coronal, Fluid=0)
    Axial T1 (Plane=Axial, Fluid=0) is unmapped to the 5 primary slots (returns None).
    """
    plane_str = str(plane).strip().capitalize()
    is_fluid = int(fluid_sensitive) == 1

    if plane_str == "Sagittal":
        return 0 if is_fluid else 3
    elif plane_str == "Coronal":
        return 1 if is_fluid else 4
    elif plane_str == "Axial":
        return 2 if is_fluid else None
    return None


class MultiSeriesKneeDataset(Dataset):
    """Study-level dataset for Phase C3 multi-series 3D vision modeling."""

    def __init__(
        self,
        study_uids: List[str],
        processed_dir: str,
        labels_df: pd.DataFrame,
        series_meta_df: Optional[pd.DataFrame] = None,
        gold_uids: Optional[set] = None,
        transform=None,
    ):
        """
        Args:
            study_uids: List of StudyInstanceUID strings for this split.
            processed_dir: Path to phase_c/c1/processed/.
            labels_df: Canonical DataFrame from phase_b_step3_remediated_training_corpus.csv.
            series_meta_df: DataFrame containing ['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive'].
                            If None or empty, dataset strictly halts with InsufficientMetadataError.
            gold_uids: Set of Gold-58 StudyInstanceUIDs for runtime leakage assertion.
            transform: Optional spatial/intensity augmentations.
        """
        super().__init__()
        self.processed_dir = processed_dir
        self.transform = transform
        self.gold_uids = gold_uids or set()

        # Strict Gold-58 isolation check
        overlap = set(study_uids) & self.gold_uids
        assert len(overlap) == 0, f"CRITICAL LEAKAGE: {len(overlap)} Gold-58 studies detected in dataset!"

        # Strict series metadata validation
        if series_meta_df is None or len(series_meta_df) == 0:
            raise InsufficientMetadataError(
                "Series metadata (train_series.csv containing Anatomical_Plane and Fluid_Sensitive) "
                "is absent or empty. C3.2 explicitly forbids arbitrary filename classification into canonical slots. "
                "Deterministic classification requires series metadata mapping SeriesInstanceUID to Anatomical_Plane and Fluid_Sensitive."
            )

        required_meta_cols = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"}
        missing_cols = required_meta_cols - set(series_meta_df.columns)
        if missing_cols:
            raise InsufficientMetadataError(
                f"Series metadata is missing required columns: {missing_cols}. "
                "Deterministic classification requires Anatomical_Plane and Fluid_Sensitive."
            )

        # Pre-index series metadata by StudyInstanceUID
        self.series_meta_by_study = dict(tuple(series_meta_df.groupby("StudyInstanceUID")))

        # Pre-index labels: study_uid -> (targets_tensor, weights_tensor)
        self.labels_map = self._build_labels_map(labels_df, study_uids)

        # Filter study_uids to those with labels and valid directory
        self.study_uids = [s for s in study_uids if s in self.labels_map]
        assert len(self.study_uids) > 0, "No valid studies found in dataset."

        # Collision & provenance tracking
        self.collision_log: List[Dict[str, Any]] = []

    def _build_labels_map(self, labels_df: pd.DataFrame, study_uids: List[str]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract continuous Y and W tensors per study in canonical target order."""
        labels_map = {}
        filtered_df = labels_df[labels_df["StudyInstanceUID"].isin(study_uids)]

        for study_uid, group in filtered_df.groupby("StudyInstanceUID"):
            # Ensure all 12 targets exist
            group_targets = set(group["target"].tolist())
            missing = set(TARGETS) - group_targets
            assert len(missing) == 0, f"Study {study_uid} missing targets: {missing}"

            # Order by canonical TARGETS list
            indexed_group = group.set_index("target").reindex(TARGETS)

            # Continuous Y (label_probability)
            y = indexed_group["label_probability"].values.astype(np.float32)
            assert np.all((y >= 0.0) & (y <= 1.0)), f"Study {study_uid} has invalid Y values"

            # Sample weight W (forced to 0 if abstain_flag == 1)
            w = indexed_group["sample_weight"].values.astype(np.float32)
            abstain = indexed_group["abstain_flag"].values.astype(np.int32)
            w[abstain == 1] = 0.0
            assert np.all(w >= 0.0), f"Study {study_uid} has negative sample weights"

            labels_map[study_uid] = (
                torch.from_numpy(y),
                torch.from_numpy(w),
            )
        return labels_map

    def _resolve_slot_assignments(self, study_uid: str, available_series_files: Dict[str, str]) -> Tuple[Dict[int, str], List[Dict[str, Any]]]:
        """Map available on-disk series files into canonical slots 0..4 using series metadata.

        Collision policy:
        If multiple series map to the same slot, select the series deterministically
        by lexicographically lowest SeriesInstanceUID, and log the collision for provenance.
        """
        slot_assignments: Dict[int, str] = {}
        collisions: List[Dict[str, Any]] = []

        if study_uid not in self.series_meta_by_study:
            return slot_assignments, collisions

        meta_rows = self.series_meta_by_study[study_uid]

        for _, row in meta_rows.iterrows():
            ser_uid = row["SeriesInstanceUID"]
            if ser_uid not in available_series_files:
                continue

            plane = row["Anatomical_Plane"]
            fluid = row["Fluid_Sensitive"]
            slot_idx = classify_series_slot(plane, fluid)

            if slot_idx is not None:
                if slot_idx in slot_assignments:
                    # Collision detected: deterministic tie-breaker
                    existing_ser = slot_assignments[slot_idx]
                    chosen_ser = min(existing_ser, ser_uid)
                    discarded_ser = max(existing_ser, ser_uid)
                    slot_assignments[slot_idx] = chosen_ser

                    collision_record = {
                        "study_uid": study_uid,
                        "slot_idx": slot_idx,
                        "slot_name": CANONICAL_SLOTS[slot_idx],
                        "chosen_series": chosen_ser,
                        "discarded_series": discarded_ser,
                        "plane": plane,
                        "fluid_sensitive": fluid,
                    }
                    collisions.append(collision_record)
                    self.collision_log.append(collision_record)
                else:
                    slot_assignments[slot_idx] = ser_uid

        return slot_assignments, collisions

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        study_uid = self.study_uids[idx]
        targets, weights = self.labels_map[study_uid]

        # Scan available .npz files on disk
        study_dir = os.path.join(self.processed_dir, study_uid)
        available_npz = {}
        if os.path.isdir(study_dir):
            for fn in os.listdir(study_dir):
                if fn.endswith(".npz"):
                    ser_uid = fn[:-4]
                    available_npz[ser_uid] = os.path.join(study_dir, fn)

        # Map to canonical slots
        slot_assignments, _ = self._resolve_slot_assignments(study_uid, available_npz)

        # Allocate slot volume tensor (5, 1, 32, 256, 256) and slot_mask (5,)
        volumes = torch.zeros((NUM_SLOTS, 1, *TENSOR_SHAPE), dtype=torch.float32)
        slot_mask = torch.zeros((NUM_SLOTS,), dtype=torch.float32)
        provenance: Dict[int, str] = {i: "" for i in range(NUM_SLOTS)}

        for slot_idx in range(NUM_SLOTS):
            if slot_idx in slot_assignments:
                ser_uid = slot_assignments[slot_idx]
                npz_path = available_npz[ser_uid]
                try:
                    with np.load(npz_path) as data:
                        vol_np = data["volume"].astype(np.float32)  # (32, 256, 256)
                    vol_tensor = torch.from_numpy(vol_np).unsqueeze(0)  # (1, 32, 256, 256)
                    if self.transform is not None:
                        vol_tensor = self.transform(vol_tensor)
                    volumes[slot_idx] = vol_tensor
                    slot_mask[slot_idx] = 1.0
                    provenance[slot_idx] = ser_uid
                except Exception:
                    # In case of load failure, leave zero-padded and mask=0
                    volumes[slot_idx].zero_()
                    slot_mask[slot_idx] = 0.0

        return {
            "study_uid": study_uid,
            "volumes": volumes,          # (5, 1, 32, 256, 256)
            "slot_mask": slot_mask,      # (5,)
            "targets": targets,          # (12,) continuous soft BCE targets
            "weights": weights,          # (12,) continuous sample weights
            "provenance": provenance,    # Dict mapping slot_idx -> SeriesInstanceUID
        }


def multi_series_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for MultiSeriesKneeDataset.

    Stacks tensors cleanly and preserves list of study UIDs and slot provenance.
    """
    return {
        "study_uid": [d["study_uid"] for d in batch],
        "volumes": torch.stack([d["volumes"] for d in batch], dim=0),
        "slot_mask": torch.stack([d["slot_mask"] for d in batch], dim=0),
        "targets": torch.stack([d["targets"] for d in batch], dim=0),
        "weights": torch.stack([d["weights"] for d in batch], dim=0),
        "provenance": [d["provenance"] for d in batch],
    }

