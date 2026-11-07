"""
Phase C2: Unified dataset interface for RAW / Pipeline A / Pipeline B / Pipeline C.

Each study has multiple series. For training, we select ONE representative series
per study (the fluid-sensitive sagittal series, falling back to any sagittal, then
any series) to keep the experiment controlled and avoid variable-length inputs.

The dataset returns:
    volume: torch.Tensor of shape (1, D, H, W) — single-channel 3D volume
    targets: torch.Tensor of shape (12,) — soft label probabilities (Y)
    weights: torch.Tensor of shape (12,) — sample weights (W)
    study_uid: str — for provenance tracking
"""
import os
import glob
import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage
import torch
from torch.utils.data import Dataset

from config import (
    DATA_ROOT, TRAIN_SERIES_DIR, TRAIN_SERIES_CSV,
    C0_PROCESSED, LABELS_CSV, TARGETS, MODEL_INPUT_SHAPE,
)


def _select_representative_series(study_uid: str, series_meta: pd.DataFrame) -> str:
    """Select one representative series per study.
    Priority: fluid-sensitive sagittal > any sagittal > first available.
    """
    study_series = series_meta[series_meta["StudyInstanceUID"] == study_uid]
    if study_series.empty:
        # Fallback: pick first series directory from disk
        study_path = os.path.join(TRAIN_SERIES_DIR, study_uid)
        if os.path.isdir(study_path):
            dirs = sorted(os.listdir(study_path))
            return dirs[0] if dirs else None
        return None

    # Prefer fluid-sensitive sagittal
    fs_sag = study_series[
        (study_series["Fluid_Sensitive"] == 1) &
        (study_series["Anatomical_Plane"] == "Sagittal")
    ]
    if len(fs_sag) > 0:
        return fs_sag.iloc[0]["SeriesInstanceUID"]

    # Fallback: any sagittal
    sag = study_series[study_series["Anatomical_Plane"] == "Sagittal"]
    if len(sag) > 0:
        return sag.iloc[0]["SeriesInstanceUID"]

    # Fallback: first series
    return study_series.iloc[0]["SeriesInstanceUID"]


def _load_raw_dicom_volume(study_uid: str, series_uid: str) -> np.ndarray:
    """Load raw DICOM series and return normalized float32 volume."""
    series_path = os.path.join(TRAIN_SERIES_DIR, study_uid, series_uid)
    dcm_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files in {series_path}")

    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f)
            slices.append(ds)
        except Exception:
            pass

    if not slices:
        raise ValueError(f"Could not read any DICOM from {series_path}")

    # Sort by position along slice normal
    has_pos = all(
        hasattr(s, "ImagePositionPatient") and hasattr(s, "ImageOrientationPatient")
        for s in slices
    )
    if has_pos:
        iop = slices[0].ImageOrientationPatient
        r = np.array(iop[:3])
        c = np.array(iop[3:])
        normal = np.cross(r, c)
        slices.sort(key=lambda s: np.dot(np.array(s.ImagePositionPatient), normal))
    elif all(hasattr(s, "InstanceNumber") for s in slices):
        slices.sort(key=lambda s: int(s.InstanceNumber))

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)

    # Apply rescale
    slope = getattr(slices[0], "RescaleSlope", 1.0)
    intercept = getattr(slices[0], "RescaleIntercept", 0.0)
    if slope != 1.0 or intercept != 0.0:
        volume = volume * float(slope) + float(intercept)

    # Robust percentile normalization (same as Pipeline B/C)
    p_low, p_high = np.percentile(volume, (0.5, 99.5))
    if p_high > p_low:
        volume = np.clip((volume - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        volume = np.zeros_like(volume)

    return volume.astype(np.float32)


def _resize_volume_to_target(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize a 3D volume to target_shape using trilinear interpolation."""
    td, th, tw = target_shape
    d, h, w = volume.shape
    if (d, h, w) == (td, th, tw):
        return volume
    zoom_factors = (td / d, th / h, tw / w)
    resized = scipy.ndimage.zoom(volume, zoom_factors, order=1)
    # Ensure exact shape (rounding can cause ±1)
    resized = resized[:td, :th, :tw]
    # Pad if needed
    if resized.shape != (td, th, tw):
        padded = np.zeros((td, th, tw), dtype=resized.dtype)
        rd, rh, rw = resized.shape
        padded[:rd, :rh, :rw] = resized
        resized = padded
    return resized


def _load_preprocessed_volume(pipeline: str, study_uid: str, series_uid: str) -> np.ndarray:
    """Load a preprocessed .npz volume from C0 output."""
    npz_path = os.path.join(C0_PROCESSED, pipeline, study_uid, f"{series_uid}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing preprocessed volume: {npz_path}")
    data = np.load(npz_path)
    volume = data["volume"].astype(np.float32)
    return volume


class KneeC2Dataset(Dataset):
    """Unified dataset for the C2 four-way experiment.

    Args:
        study_uids: List of StudyInstanceUIDs for this split.
        pipeline: One of 'raw', 'pipeline_a', 'pipeline_b', 'pipeline_c'.
        labels_df: DataFrame with columns [StudyInstanceUID, target, label_probability,
                   confidence_score, sample_weight, abstain_flag, final_predicted_label].
        series_meta: DataFrame from train_series.csv for series selection.
        target_shape: Fixed 3D shape for model input.
    """

    def __init__(
        self,
        study_uids: list,
        pipeline: str,
        labels_df: pd.DataFrame,
        series_meta: pd.DataFrame,
        target_shape: tuple = MODEL_INPUT_SHAPE,
    ):
        self.study_uids = list(study_uids)
        self.pipeline = pipeline
        self.target_shape = target_shape

        # Pre-compute labels: StudyUID -> {target: (Y, W)}
        self.label_map = {}
        for uid in self.study_uids:
            study_labels = labels_df[labels_df["StudyInstanceUID"] == uid]
            y = np.zeros(len(TARGETS), dtype=np.float32)
            w = np.zeros(len(TARGETS), dtype=np.float32)
            for _, row in study_labels.iterrows():
                if row["target"] in TARGETS:
                    idx = TARGETS.index(row["target"])
                    y[idx] = float(row["label_probability"])
                    w[idx] = float(row["sample_weight"])
                    if int(row.get("abstain_flag", 0)) == 1:
                        w[idx] = 0.0
            self.label_map[uid] = (y, w)

        # Pre-select representative series for each study
        self.series_map = {}
        for uid in self.study_uids:
            self.series_map[uid] = _select_representative_series(uid, series_meta)

    def __len__(self):
        return len(self.study_uids)

    def __getitem__(self, idx):
        study_uid = self.study_uids[idx]
        series_uid = self.series_map[study_uid]

        # Load volume based on pipeline
        if self.pipeline == "raw":
            volume = _load_raw_dicom_volume(study_uid, series_uid)
        else:
            volume = _load_preprocessed_volume(self.pipeline, study_uid, series_uid)

        # Normalize preprocessed Pipeline A volumes (not already normalized)
        if self.pipeline == "pipeline_a":
            p_low, p_high = np.percentile(volume, (0.5, 99.5))
            if p_high > p_low:
                volume = np.clip((volume - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
            else:
                volume = np.zeros_like(volume, dtype=np.float32)

        # Resize to fixed target shape
        volume = _resize_volume_to_target(volume, self.target_shape)

        # Convert to tensor: (1, D, H, W) — single channel
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()

        # Labels
        y, w = self.label_map[study_uid]
        y_tensor = torch.from_numpy(y).float()
        w_tensor = torch.from_numpy(w).float()

        return volume_tensor, y_tensor, w_tensor, study_uid
