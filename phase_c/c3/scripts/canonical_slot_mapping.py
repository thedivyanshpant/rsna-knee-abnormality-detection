#!/usr/bin/env python3
"""
Phase C3: Canonical Series Slot Mapping & Audit Engine
Deterministic 5-slot multi-series allocation for RSNA Knee Abnormality Detection.

Canonical Slots:
  0: sagittal_fluid_sensitive  (Plane == 'Sagittal' and Fluid_Sensitive == 1)
  1: coronal_fluid_sensitive   (Plane == 'Coronal' and Fluid_Sensitive == 1)
  2: axial_fluid_sensitive     (Plane == 'Axial' and Fluid_Sensitive == 1)
  3: sagittal_non_fs_t1        (Plane == 'Sagittal' and Fluid_Sensitive == 0)
  4: coronal_non_fs_t1         (Plane == 'Coronal' and Fluid_Sensitive == 0)
"""

import os
import sys
import json
import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Canonical paths
REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
PROCESSED_DIR = os.path.join(REPO_ROOT, "phase_c", "c1", "processed")
META_PATH = os.path.join(REPO_ROOT, "csv", "train_series.csv")
GOLD_PATH = os.path.join(REPO_ROOT, "csv", "gold_studies.csv")
MANIFEST_DIR = os.path.join(REPO_ROOT, "phase_c", "c3", "manifests")
RESULTS_DIR = os.path.join(REPO_ROOT, "phase_c", "c3", "results")

CANONICAL_SLOTS = [
    "sagittal_fluid_sensitive",
    "coronal_fluid_sensitive",
    "axial_fluid_sensitive",
    "sagittal_non_fs_t1",
    "coronal_non_fs_t1",
]
NUM_SLOTS = len(CANONICAL_SLOTS)

KNOWN_PARTIAL_STUDIES = {
    "1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035": "1.2.826.0.1.3680043.8.498.39396636671446532796538574315802928348",
    "1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680": "1.2.826.0.1.3680043.8.498.31160785238781353848727311763596115703",
}


def classify_series_slot(plane: str, fluid_sensitive: int) -> Optional[int]:
    plane_str = str(plane).strip().capitalize()
    is_fluid = int(fluid_sensitive) == 1

    if plane_str == "Sagittal":
        return 0 if is_fluid else 3
    elif plane_str == "Coronal":
        return 1 if is_fluid else 4
    elif plane_str == "Axial":
        return 2 if is_fluid else None
    return None


def run_slot_mapping_audit():
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("PHASE C3: CANONICAL SERIES SLOT MAPPING & AUDIT")
    print("=" * 80)

    # 1. Load inputs
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing series metadata at {META_PATH}")
    meta_df = pd.read_csv(META_PATH)

    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(f"Missing Gold-58 list at {GOLD_PATH}")
    gold_df = pd.read_csv(GOLD_PATH)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())

    # Validate metadata schema
    required_cols = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"}
    missing_cols = required_cols - set(meta_df.columns)
    if missing_cols:
        raise ValueError(f"Series metadata missing columns: {missing_cols}")

    # Check for unexpected metadata values
    valid_planes = {"Sagittal", "Coronal", "Axial"}
    found_planes = set(meta_df["Anatomical_Plane"].dropna().unique())
    unexpected_planes = found_planes - valid_planes
    if unexpected_planes:
        raise ValueError(f"CRITICAL: Unexpected Anatomical_Plane values: {unexpected_planes}")

    valid_fluids = {0, 1}
    found_fluids = set(meta_df["Fluid_Sensitive"].dropna().unique())
    unexpected_fluids = found_fluids - valid_fluids
    if unexpected_fluids:
        raise ValueError(f"CRITICAL: Unexpected Fluid_Sensitive values: {unexpected_fluids}")

    # 2. Inspect on-disk processed volumes
    study_dirs = sorted([d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))])
    total_studies = len(study_dirs)

    # Assert 0 Gold leakage on disk
    overlap_gold = set(study_dirs) & gold_uids
    if overlap_gold:
        raise AssertionError(f"CRITICAL LEAKAGE: {len(overlap_gold)} Gold studies found in {PROCESSED_DIR}")

    on_disk_series_by_study: Dict[str, Dict[str, str]] = {}
    all_on_disk_series = set()
    for s in study_dirs:
        s_path = os.path.join(PROCESSED_DIR, s)
        s_files = {f[:-4]: os.path.join(s_path, f) for f in os.listdir(s_path) if f.endswith(".npz")}
        on_disk_series_by_study[s] = s_files
        all_on_disk_series.update(s_files.keys())

    total_processed_npz = len(all_on_disk_series)

    # 3. Filter metadata for non-Gold cohort
    non_gold_meta = meta_df[~meta_df["StudyInstanceUID"].isin(gold_uids)].copy()
    total_non_gold_meta_rows = len(non_gold_meta)
    non_gold_meta_series = set(non_gold_meta["SeriesInstanceUID"].tolist())

    # 4. Join validation
    unmatched_processed = all_on_disk_series - non_gold_meta_series
    if unmatched_processed:
        raise AssertionError(f"CRITICAL JOIN ANOMALY: {len(unmatched_processed)} processed .npz have no metadata! Example: {list(unmatched_processed)[:5]}")

    meta_no_npz = non_gold_meta_series - all_on_disk_series
    # Verify meta_no_npz exactly equals known partial failed series
    expected_missing = set(KNOWN_PARTIAL_STUDIES.values())
    unexplained_missing = meta_no_npz - expected_missing
    if unexplained_missing:
        raise AssertionError(f"CRITICAL JOIN ANOMALY: Unexplained missing series from disk: {unexplained_missing}")

    # Pre-index non-Gold metadata by study
    meta_by_study = dict(tuple(non_gold_meta.groupby("StudyInstanceUID")))

    # 5. Execute canonical slot mapping across all 4,349 studies
    mapping_records = []
    collision_records = []
    slot_counts = {i: 0 for i in range(NUM_SLOTS)}
    slots_populated_hist = {k: 0 for k in range(NUM_SLOTS + 1)}

    for study_uid in study_dirs:
        available_npz = on_disk_series_by_study[study_uid]
        assigned_slots: Dict[int, str] = {}
        assigned_provenance: Dict[int, Dict[str, Any]] = {}

        if study_uid in meta_by_study:
            study_meta = meta_by_study[study_uid]
            for _, row in study_meta.iterrows():
                ser_uid = row["SeriesInstanceUID"]
                if ser_uid not in available_npz:
                    continue  # E.g. Corrupt partial series

                plane = row["Anatomical_Plane"]
                fluid = row["Fluid_Sensitive"]
                fat_sup = row.get("Fat_Suppression", None)
                slot_idx = classify_series_slot(plane, fluid)

                if slot_idx is not None:
                    if slot_idx in assigned_slots:
                        # Collision detected
                        existing_ser = assigned_slots[slot_idx]
                        chosen_ser = min(existing_ser, ser_uid)
                        discarded_ser = max(existing_ser, ser_uid)
                        assigned_slots[slot_idx] = chosen_ser

                        collision_rec = {
                            "StudyInstanceUID": study_uid,
                            "slot_idx": slot_idx,
                            "slot_name": CANONICAL_SLOTS[slot_idx],
                            "chosen_series": chosen_ser,
                            "discarded_series": discarded_ser,
                            "plane": plane,
                            "fluid_sensitive": fluid,
                        }
                        collision_records.append(collision_rec)
                    else:
                        assigned_slots[slot_idx] = ser_uid
                        assigned_provenance[slot_idx] = {
                            "plane": plane,
                            "fluid_sensitive": fluid,
                            "fat_suppression": fat_sup,
                        }

        # Compute slot mask and record
        mask = [1 if i in assigned_slots else 0 for i in range(NUM_SLOTS)]
        num_populated = sum(mask)
        slots_populated_hist[num_populated] += 1

        for i in range(NUM_SLOTS):
            if mask[i] == 1:
                slot_counts[i] += 1

        record = {
            "StudyInstanceUID": study_uid,
            "num_available_series": len(available_npz),
            "num_populated_slots": num_populated,
            "slot_0_sagittal_fs": assigned_slots.get(0, ""),
            "slot_1_coronal_fs": assigned_slots.get(1, ""),
            "slot_2_axial_fs": assigned_slots.get(2, ""),
            "slot_3_sagittal_t1": assigned_slots.get(3, ""),
            "slot_4_coronal_t1": assigned_slots.get(4, ""),
            "mask_slot_0": mask[0],
            "mask_slot_1": mask[1],
            "mask_slot_2": mask[2],
            "mask_slot_3": mask[3],
            "mask_slot_4": mask[4],
            "is_known_partial": study_uid in KNOWN_PARTIAL_STUDIES,
        }
        mapping_records.append(record)

    mapping_df = pd.DataFrame(mapping_records)
    collisions_df = pd.DataFrame(collision_records)

    # 6. Audit known partial studies
    partial_study_audit = {}
    for p_uid, missing_ser in KNOWN_PARTIAL_STUDIES.items():
        row = mapping_df[mapping_df["StudyInstanceUID"] == p_uid].iloc[0]
        row_series_uids = [row["slot_0_sagittal_fs"], row["slot_1_coronal_fs"], row["slot_2_axial_fs"], row["slot_3_sagittal_t1"], row["slot_4_coronal_t1"]]
        partial_study_audit[p_uid] = {
            "num_available_series": int(row["num_available_series"]),
            "num_populated_slots": int(row["num_populated_slots"]),
            "mask": [int(row[f"mask_slot_{i}"]) for i in range(NUM_SLOTS)],
            "missing_failed_series": missing_ser,
            "safely_handled": int(row["num_populated_slots"]) >= 1 and missing_ser not in row_series_uids,
        }

    # 7. Save outputs
    manifest_csv_path = os.path.join(MANIFEST_DIR, "c3_canonical_slot_mapping_manifest.csv")
    mapping_df.to_csv(manifest_csv_path, index=False)

    collisions_csv_path = os.path.join(MANIFEST_DIR, "c3_canonical_slot_mapping_collisions.csv")
    collisions_df.to_csv(collisions_csv_path, index=False)

    audit_summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_studies_evaluated": total_studies,
        "total_processed_npz_series": total_processed_npz,
        "total_metadata_rows": len(meta_df),
        "total_non_gold_metadata_rows": total_non_gold_meta_rows,
        "gold_series_in_metadata": len(meta_df) - total_non_gold_meta_rows,
        "matched_series": len(all_on_disk_series & non_gold_meta_series),
        "unmatched_processed_series": len(unmatched_processed),
        "metadata_rows_with_no_processed_npz": len(meta_no_npz),
        "missing_series_explanation": {
            "expected_partial_failures": list(expected_missing),
            "unexplained_missing": list(unexplained_missing),
        },
        "studies_by_populated_slot_count": slots_populated_hist,
        "per_slot_population_counts": {CANONICAL_SLOTS[i]: slot_counts[i] for i in range(NUM_SLOTS)},
        "per_slot_missing_counts": {CANONICAL_SLOTS[i]: total_studies - slot_counts[i] for i in range(NUM_SLOTS)},
        "total_collisions": len(collisions_df),
        "colliding_studies_count": collisions_df["StudyInstanceUID"].nunique() if len(collisions_df) > 0 else 0,
        "known_partial_studies_audit": partial_study_audit,
        "unexpected_metadata_values": {
            "unexpected_planes": list(unexpected_planes),
            "unexpected_fluids": list(unexpected_fluids),
        },
        "pass_status": True,
    }

    audit_json_path = os.path.join(MANIFEST_DIR, "c3_canonical_slot_mapping_audit.json")
    with open(audit_json_path, "w") as f:
        json.dump(audit_summary, f, indent=2)

    # 8. Generate Markdown report
    md_path = os.path.join(RESULTS_DIR, "c3_canonical_slot_mapping_audit.md")
    with open(md_path, "w") as f:
        f.write("# Phase C3: Canonical Series Slot Mapping & Audit Report\n\n")
        f.write(f"- **Audit Date**: {audit_summary['timestamp']}\n")
        f.write(f"- **Total Non-Gold Cohort Studies**: {total_studies}\n")
        f.write(f"- **Total Processed Volumes on Disk**: {total_processed_npz}\n")
        f.write(f"- **Total Matched Series**: {audit_summary['matched_series']}\n")
        f.write(f"- **Unmatched Processed Series**: {audit_summary['unmatched_processed_series']} (FAIL criteria = > 0)\n")
        f.write(f"- **Metadata Rows with No Processed NPZ**: {audit_summary['metadata_rows_with_no_processed_npz']} (Both verified as C1 corrupt raw series)\n")
        f.write(f"- **Status**: **PASS (Zero Join Anomalies)**\n\n")

        f.write("## 1. Studies by Populated Slot Count\n\n")
        f.write("| Populated Slots | Number of Studies | Percentage |\n| :---: | :---: | :---: |\n")
        for k in range(NUM_SLOTS + 1):
            cnt = slots_populated_hist[k]
            pct = (cnt / total_studies) * 100.0
            f.write(f"| **{k} / 5 slots** | {cnt:,} | {pct:.2f}% |\n")

        f.write("\n## 2. Per-Slot Allocation Counts\n\n")
        f.write("| Slot Index | Canonical Slot Name | Populated Studies | Missing Studies | Coverage (%) |\n| :---: | :--- | :---: | :---: | :---: |\n")
        for i in range(NUM_SLOTS):
            pop = slot_counts[i]
            mis = total_studies - pop
            pct = (pop / total_studies) * 100.0
            f.write(f"| {i} | `{CANONICAL_SLOTS[i]}` | {pop:,} | {mis:,} | {pct:.2f}% |\n")

        f.write("\n## 3. Collision Resolution Audit\n\n")
        f.write(f"- **Total Collisions Detected**: {len(collisions_df)}\n")
        f.write(f"- **Studies with Collisions**: {audit_summary['colliding_studies_count']}\n")
        f.write("- **Collision Policy**: Deterministic selection of lexicographically lowest `SeriesInstanceUID` (`min(series_uids)`). Both chosen and discarded series are permanently logged to manifest.\n\n")

        f.write("## 4. Known C1 Partial Studies Audit\n\n")
        for p_uid, p_data in partial_study_audit.items():
            f.write(f"### Study `{p_uid}`\n")
            f.write(f"- Available Processed Series: {p_data['num_available_series']}\n")
            f.write(f"- Populated Canonical Slots: {p_data['num_populated_slots']}\n")
            f.write(f"- Slot Mask: `{p_data['mask']}`\n")
            f.write(f"- Unprocessed / Corrupted Raw Series: `{p_data['missing_failed_series']}`\n")
            f.write(f"- Safely Handled (No fabricated volumes): **{p_data['safely_handled']}**\n\n")

    print("\n[SUCCESS] Canonical slot mapping and audit completed successfully.")
    print(f"  - Manifest CSV: {manifest_csv_path}")
    print(f"  - Collisions CSV: {collisions_csv_path}")
    print(f"  - Audit JSON: {audit_json_path}")
    print(f"  - Markdown Report: {md_path}")
    print(f"  - Total studies: {total_studies}")
    print(f"  - Total processed series: {total_processed_npz}")
    print(f"  - Collisions: {len(collisions_df)}")
    print(f"  - Studies with all 5 slots populated: {slots_populated_hist[5]} ({slots_populated_hist[5]/total_studies*100:.2f}%)")
    print(f"  - Studies with >= 4 slots populated: {slots_populated_hist[4]+slots_populated_hist[5]} ({(slots_populated_hist[4]+slots_populated_hist[5])/total_studies*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_slot_mapping_audit()
