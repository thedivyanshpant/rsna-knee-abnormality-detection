"""
Phase D1: DICOM Sequence Metadata Resolver & Physical Slice Sorter.

Implements Sections 10, 11, and 14:
- DICOM sequence weighting resolution (T1, T2, PD) with deterministic TR/TE fallbacks.
- Fat-suppression identification using canonical clinical acronyms (FS, FATSAT, TIRM, STIR, SPAIR, SPIR).
- Physical slice sorting along the acquisition plane normal:
    Primary: IPP · (IOP_row × IOP_column)
    Fallback: InstanceNumber
    Tertiary: Natural filename ordering
- Complete auditability: Zero silent guessing; all resolution paths and fallbacks are logged.
"""

import os
import re
import glob
import json
import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import pydicom

FAT_SUPPRESSION_KEYWORDS = ["FS", "FATSAT", "TIRM", "STIR", "SPAIR", "SPIR", "FAT SAT", "FAT_SAT"]


def resolve_sequence_properties(
    series_desc: Optional[str],
    sequence_name: Optional[str] = None,
    scan_options: Optional[Any] = None,
    repetition_time: Optional[float] = None,
    echo_time: Optional[float] = None
) -> Dict[str, Any]:
    """Deterministically resolve MR sequence weighting and fat suppression status."""
    desc_str = str(series_desc or "").strip().upper()
    seq_str = str(sequence_name or "").strip().upper()
    
    # Handle ScanOptions which may be a list/MultiValue or string
    if isinstance(scan_options, (list, tuple, pydicom.multival.MultiValue)):
        scan_str = " ".join(str(s) for s in scan_options).upper()
    else:
        scan_str = str(scan_options or "").strip().upper()

    combined_text = f"{desc_str} {seq_str} {scan_str}"

    # 1. Fat suppression detection
    has_fat_sat = any(kw in combined_text for kw in FAT_SUPPRESSION_KEYWORDS)

    # 2. Sequence weighting detection (Section 10)
    # Rule 1: T1 if contains T1 and not T2/PD
    is_t1 = ("T1" in desc_str) and not ("T2" in desc_str or "PD" in desc_str or "PROTON" in desc_str)
    # Rule 2: T2 if contains T2 and not PD/Proton
    is_t2 = ("T2" in desc_str) and not ("PD" in desc_str or "PROTON" in desc_str)
    # Rule 3: PD if contains PD / Proton
    is_pd = ("PD" in desc_str) or ("PROTON" in desc_str)

    classification_path = "description_heuristics"
    tr = float(repetition_time) if repetition_time is not None else -1.0
    te = float(echo_time) if echo_time is not None else -1.0

    # Fallback to TR / TE (Section 10)
    if not (is_t1 or is_t2 or is_pd):
        classification_path = "tr_te_fallback"
        if 0.0 < tr < 800.0:
            is_t1 = True
        elif te > 60.0:
            is_t2 = True
        elif tr >= 800.0:
            is_pd = True
        else:
            classification_path = "unresolved"

    if is_t1:
        weighting = "T1"
    elif is_t2:
        weighting = "T2"
    elif is_pd:
        weighting = "PD"
    else:
        weighting = "UNKNOWN"

    # Fluid sensitivity: T2 or PD sequences are fluid sensitive
    is_fluid = is_t2 or is_pd

    return {
        "weighting": weighting,
        "is_fluid": is_fluid,
        "has_fat_sat": has_fat_sat,
        "classification_path": classification_path,
        "repetition_time": tr,
        "echo_time": te,
        "series_description": desc_str
    }


def sort_dicom_slices_physically(
    dicom_file_paths: List[str]
) -> Tuple[List[str], str]:
    """Sort DICOM slices physically along the acquisition normal (Section 14).
    
    Primary: IPP · (IOP_row × IOP_column)
    Secondary fallback: InstanceNumber
    Tertiary fallback: Natural filename ordering
    """
    if len(dicom_file_paths) <= 1:
        return dicom_file_paths, "trivial_single_slice"

    # Attempt primary physical sorting
    try:
        slice_coords = []
        for path in dicom_file_paths:
            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            ipp = np.array(dcm.ImagePositionPatient, dtype=np.float64)
            iop = np.array(dcm.ImageOrientationPatient, dtype=np.float64)
            row_vec = iop[:3]
            col_vec = iop[3:]
            normal_vec = np.cross(row_vec, col_vec)
            dist = np.dot(ipp, normal_vec)
            slice_coords.append((dist, path))
        
        slice_coords.sort(key=lambda x: x[0])
        sorted_paths = [x[1] for x in slice_coords]
        return sorted_paths, "physical_normal_projection"
    except Exception:
        pass

    # Secondary fallback: InstanceNumber
    try:
        instance_coords = []
        for path in dicom_file_paths:
            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            inst = int(dcm.InstanceNumber)
            instance_coords.append((inst, path))
        instance_coords.sort(key=lambda x: x[0])
        sorted_paths = [x[1] for x in instance_coords]
        return sorted_paths, "instance_number_fallback"
    except Exception:
        pass

    # Tertiary fallback: Natural filename sorting
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    sorted_paths = sorted(dicom_file_paths, key=natural_keys)
    return sorted_paths, "natural_filename_fallback"


PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

def run_dicom_metadata_audit(
    raw_dicom_dir: Optional[str] = None,
    train_series_csv: Optional[str] = None,
    output_audit_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Audit DICOM metadata across available raw DICOM series."""
    raw_dicom_dir = raw_dicom_dir or os.environ.get("RSNA_RAW_DICOM_DIR", os.path.join(PROJECT_ROOT, "data/train_series"))
    train_series_csv = train_series_csv or os.path.join(PROJECT_ROOT, "csv/train_series.csv")
    output_audit_dir = output_audit_dir or os.path.join(PROJECT_ROOT, "phase_d/d1/audits")
    os.makedirs(output_audit_dir, exist_ok=True)
    meta_df = pd.read_csv(train_series_csv)
    meta_by_series = dict(tuple(meta_df.groupby("SeriesInstanceUID")))

    series_dirs = glob.glob(os.path.join(raw_dicom_dir, "*", "*"))
    print(f"Auditing {len(series_dirs)} raw DICOM series in {raw_dicom_dir}...")

    records = []
    sorting_methods = {}
    slice_counts = []
    pixel_spacings = []

    for sdir in series_dirs:
        parts = sdir.split("/")
        study_uid = parts[-2]
        ser_uid = parts[-1]
        dcm_files = glob.glob(os.path.join(sdir, "*.dcm"))
        if not dcm_files:
            continue

        sorted_files, sort_method = sort_dicom_slices_physically(dcm_files)
        sorting_methods[sort_method] = sorting_methods.get(sort_method, 0) + 1
        num_slices = len(sorted_files)
        slice_counts.append(num_slices)

        # Read header of first sorted slice
        try:
            dcm = pydicom.dcmread(sorted_files[0], stop_before_pixels=True)
            desc = getattr(dcm, "SeriesDescription", "")
            seq_name = getattr(dcm, "SequenceName", "")
            scan_opts = getattr(dcm, "ScanOptions", None)
            tr = float(getattr(dcm, "RepetitionTime", -1.0))
            te = float(getattr(dcm, "EchoTime", -1.0))
            px_val = getattr(dcm, "PixelSpacing", None)
            px = float(px_val[0]) if px_val is not None else -1.0
            if px > 0:
                pixel_spacings.append(px)
        except Exception as e:
            desc, seq_name, scan_opts, tr, te, px = "", "", None, -1.0, -1.0, -1.0

        res = resolve_sequence_properties(desc, seq_name, scan_opts, tr, te)

        # Merge with train_series.csv for Anatomical_Plane
        plane = "UNKNOWN"
        if ser_uid in meta_by_series:
            plane = str(meta_by_series[ser_uid]["Anatomical_Plane"].values[0])

        records.append({
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": ser_uid,
            "Anatomical_Plane": plane,
            "SeriesDescription": desc,
            "TR": tr,
            "TE": te,
            "PixelSpacing": px,
            "num_slices": num_slices,
            "sorting_method": sort_method,
            "weighting": res["weighting"],
            "is_fluid": res["is_fluid"],
            "has_fat_sat": res["has_fat_sat"],
            "classification_path": res["classification_path"]
        })

    audit_df = pd.DataFrame(records)
    audit_csv = os.path.join(output_audit_dir, "d1_raw_dicom_metadata_audit.csv")
    audit_df.to_csv(audit_csv, index=False)

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_studies_inspected": audit_df["StudyInstanceUID"].nunique() if len(audit_df) > 0 else 0,
        "total_series_inspected": len(audit_df),
        "series_by_weighting": audit_df["weighting"].value_counts().to_dict(),
        "fluid_sensitive_count": int(audit_df["is_fluid"].sum()),
        "fat_suppressed_count": int(audit_df["has_fat_sat"].sum()),
        "per_plane_counts": audit_df["Anatomical_Plane"].value_counts().to_dict(),
        "classification_paths": audit_df["classification_path"].value_counts().to_dict(),
        "unresolved_sequences": int((audit_df["weighting"] == "UNKNOWN").sum()),
        "slice_sorting_methods": sorting_methods,
        "slice_count_distribution": {
            "min": int(np.min(slice_counts)) if slice_counts else 0,
            "median": float(np.median(slice_counts)) if slice_counts else 0,
            "mean": round(float(np.mean(slice_counts)), 2) if slice_counts else 0,
            "max": int(np.max(slice_counts)) if slice_counts else 0
        },
        "pixel_spacing_distribution": {
            "min": round(float(np.min(pixel_spacings)), 4) if pixel_spacings else 0,
            "mean": round(float(np.mean(pixel_spacings)), 4) if pixel_spacings else 0,
            "max": round(float(np.max(pixel_spacings)), 4) if pixel_spacings else 0
        },
        "pass_status": bool((audit_df["weighting"] == "UNKNOWN").sum() == 0)
    }

    audit_json = os.path.join(output_audit_dir, "d1_metadata_audit.json")
    with open(audit_json, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_dicom_metadata_audit()
    print("[SUCCESS] DICOM Metadata Audit Complete:")
    print(f"Total Series Inspected: {summary['total_series_inspected']}")
    print(f"Weightings: {summary['series_by_weighting']}")
    print(f"Fat-Suppressed: {summary['fat_suppressed_count']}")
    print(f"Unresolved: {summary['unresolved_sequences']}")
    print(f"Sorting methods: {summary['slice_sorting_methods']}")
