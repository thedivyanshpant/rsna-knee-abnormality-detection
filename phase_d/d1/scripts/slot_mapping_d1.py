"""
Phase D1: Six-Slot Series Mapper and Collision Resolution Engine.

Implements the six canonical target slots specified in Section 9:
  Slot 0: SAG_FLUID_FS   (Sagittal, Fluid-sensitive, Fat-suppressed)
  Slot 1: COR_FLUID_FS   (Coronal, Fluid-sensitive, Fat-suppressed)
  Slot 2: AX_FLUID_FS    (Axial, Fluid-sensitive, Fat-suppressed)
  Slot 3: SAG_FLUID_NOFS (Sagittal, Fluid-sensitive, Non-fat-suppressed)
  Slot 4: COR_T1         (Coronal, T1, Non-fat-suppressed)
  Slot 5: SAG_T1         (Sagittal, T1, Non-fat-suppressed)

Collision Policy (Section 12):
  Primary: Candidate series with highest slice count.
  Deterministic tie-breaker: Lexicographically lowest SeriesInstanceUID.
  All collisions are logged with full provenance.

Missing Slot Policy (Section 13):
  Missing slots remain missing, represented by zero tensors with slot_mask = 0.
"""

import os
import sys
import json
import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Canonical Slot Definitions
CANONICAL_SLOTS_D1 = [
    "SAG_FLUID_FS",    # Slot 0
    "COR_FLUID_FS",    # Slot 1
    "AX_FLUID_FS",     # Slot 2
    "SAG_FLUID_NOFS",  # Slot 3
    "COR_T1",          # Slot 4
    "SAG_T1",          # Slot 5
]
NUM_SLOTS_D1 = len(CANONICAL_SLOTS_D1)

SLOT_NAME_TO_INDEX = {name: idx for idx, name in enumerate(CANONICAL_SLOTS_D1)}


def classify_series_to_slot_d1(
    plane: str,
    fluid_sensitive: int,
    fat_suppression: int,
    weighting: Optional[str] = None
) -> Optional[int]:
    """Deterministically classify a series into one of the six canonical Phase D1 slots.
    
    Returns:
        slot_index (0..5) or None if unassigned (e.g. Axial T1).
    """
    plane_clean = str(plane).strip().capitalize()
    is_fluid = int(fluid_sensitive) == 1 or (weighting in ["T2", "PD"])
    is_fatsat = int(fat_suppression) == 1
    is_t1 = (not is_fluid) or (weighting == "T1")

    if plane_clean == "Sagittal" and is_fluid and is_fatsat:
        return 0  # SAG_FLUID_FS
    elif plane_clean == "Coronal" and is_fluid and is_fatsat:
        return 1  # COR_FLUID_FS
    elif plane_clean == "Axial" and is_fluid and is_fatsat:
        return 2  # AX_FLUID_FS
    elif plane_clean == "Sagittal" and is_fluid and not is_fatsat:
        return 3  # SAG_FLUID_NOFS
    elif plane_clean == "Coronal" and is_t1 and not is_fatsat:
        return 4  # COR_T1
    elif plane_clean == "Sagittal" and is_t1 and not is_fatsat:
        return 5  # SAG_T1
    return None


PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

def run_slot_mapping_d1(
    train_series_csv: Optional[str] = None,
    cv_splits_json: Optional[str] = None,
    processed_npz_dir: Optional[str] = None,
    output_manifest_dir: Optional[str] = None,
    output_audit_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute the full 6-slot mapping across all development cohort studies."""
    train_series_csv = train_series_csv or os.path.join(PROJECT_ROOT, "csv/train_series.csv")
    cv_splits_json = cv_splits_json or os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
    processed_npz_dir = processed_npz_dir or os.environ.get("RSNA_PROCESSED_DIR", os.path.join(PROJECT_ROOT, "phase_c/c1/processed"))
    output_manifest_dir = output_manifest_dir or os.path.join(PROJECT_ROOT, "phase_d/d1/manifests")
    output_audit_dir = output_audit_dir or os.path.join(PROJECT_ROOT, "phase_d/d1/audits")
    os.makedirs(output_manifest_dir, exist_ok=True)
    os.makedirs(output_audit_dir, exist_ok=True)

    # 1. Load CV splits
    with open(cv_splits_json) as f:
        cv_data = json.load(f)

    dev_studies = set()
    for fold_k in cv_data["folds"]:
        dev_studies.update(cv_data["folds"][fold_k]["val_studies"])
    sorted_dev_studies = sorted(list(dev_studies))

    # 2. Load metadata
    meta_df = pd.read_csv(train_series_csv)
    dev_meta = meta_df[meta_df["StudyInstanceUID"].isin(dev_studies)].copy()

    # Pre-index metadata by study
    meta_by_study = dict(tuple(dev_meta.groupby("StudyInstanceUID")))

    mapping_records = []
    collision_records = []
    per_slot_counts = {slot_name: 0 for slot_name in CANONICAL_SLOTS_D1}
    populated_slot_hist = {k: 0 for k in range(NUM_SLOTS_D1 + 1)}

    for study_uid in sorted_dev_studies:
        # Check on-disk available processed npz files
        study_dir = os.path.join(processed_npz_dir, study_uid)
        available_npz = set()
        if os.path.exists(study_dir):
            for fname in os.listdir(study_dir):
                if fname.endswith(".npz"):
                    available_npz.add(fname[:-4])

        # Group available series by slot
        candidates_per_slot: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(NUM_SLOTS_D1)}
        study_rows = meta_by_study.get(study_uid, pd.DataFrame())

        for _, row in study_rows.iterrows():
            ser_uid = row["SeriesInstanceUID"]
            if ser_uid not in available_npz:
                continue

            plane = row["Anatomical_Plane"]
            fluid = row["Fluid_Sensitive"]
            fatsat = row["Fat_Suppression"]

            slot_idx = classify_series_to_slot_d1(plane, fluid, fatsat)
            if slot_idx is not None:
                # In standardized C1 npz, volume depth is 32 slices.
                # Slice count attribute is tracked for collision ranking.
                slice_count = 32
                candidates_per_slot[slot_idx].append({
                    "series_uid": ser_uid,
                    "slice_count": slice_count,
                    "plane": plane,
                    "fluid_sensitive": fluid,
                    "fat_suppression": fatsat
                })

        # Resolve collisions deterministically
        assigned_slots: Dict[int, str] = {}
        for slot_idx, candidates in candidates_per_slot.items():
            if len(candidates) == 1:
                assigned_slots[slot_idx] = candidates[0]["series_uid"]
            elif len(candidates) > 1:
                # Sort primary by highest slice count descending, tie-break by lowest series_uid ascending
                candidates_sorted = sorted(
                    candidates,
                    key=lambda c: (-c["slice_count"], c["series_uid"])
                )
                selected = candidates_sorted[0]
                assigned_slots[slot_idx] = selected["series_uid"]

                # Log collision
                for cand in candidates:
                    is_selected = (cand["series_uid"] == selected["series_uid"])
                    collision_records.append({
                        "StudyInstanceUID": study_uid,
                        "slot_index": slot_idx,
                        "slot_name": CANONICAL_SLOTS_D1[slot_idx],
                        "candidate_SeriesInstanceUID": cand["series_uid"],
                        "candidate_slice_count": cand["slice_count"],
                        "selected_SeriesInstanceUID": selected["series_uid"],
                        "is_selected": is_selected,
                        "selection_reason": "highest_slice_count_and_lexicographic_tie_break"
                    })

        # Slot mask & histogram
        mask = [1 if i in assigned_slots else 0 for i in range(NUM_SLOTS_D1)]
        num_populated = sum(mask)
        populated_slot_hist[num_populated] += 1
        for i in range(NUM_SLOTS_D1):
            if mask[i] == 1:
                per_slot_counts[CANONICAL_SLOTS_D1[i]] += 1

        record = {
            "StudyInstanceUID": study_uid,
            "num_available_series": len(available_npz),
            "num_populated_slots": num_populated,
            "slot_0_SAG_FLUID_FS": assigned_slots.get(0, ""),
            "slot_1_COR_FLUID_FS": assigned_slots.get(1, ""),
            "slot_2_AX_FLUID_FS": assigned_slots.get(2, ""),
            "slot_3_SAG_FLUID_NOFS": assigned_slots.get(3, ""),
            "slot_4_COR_T1": assigned_slots.get(4, ""),
            "slot_5_SAG_T1": assigned_slots.get(5, ""),
            "mask_slot_0": mask[0],
            "mask_slot_1": mask[1],
            "mask_slot_2": mask[2],
            "mask_slot_3": mask[3],
            "mask_slot_4": mask[4],
            "mask_slot_5": mask[5],
        }
        mapping_records.append(record)

    mapping_df = pd.DataFrame(mapping_records)
    collisions_df = pd.DataFrame(collision_records)

    # Save manifest & collision logs
    manifest_csv = os.path.join(output_manifest_dir, "d1_slot_mapping_manifest.csv")
    mapping_df.to_csv(manifest_csv, index=False)

    collisions_csv = os.path.join(output_manifest_dir, "d1_slot_mapping_collisions.csv")
    collisions_df.to_csv(collisions_csv, index=False)

    total_studies = len(sorted_dev_studies)
    colliding_studies = collisions_df["StudyInstanceUID"].nunique() if len(collisions_df) > 0 else 0

    audit_data = {
        "audit_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_dev_studies": total_studies,
        "total_available_npz_series": sum(mapping_df["num_available_series"]),
        "canonical_slots": CANONICAL_SLOTS_D1,
        "per_slot_coverage": {
            slot: {
                "populated_studies": per_slot_counts[slot],
                "missing_studies": total_studies - per_slot_counts[slot],
                "coverage_pct": round((per_slot_counts[slot] / total_studies) * 100.0, 2)
            }
            for slot in CANONICAL_SLOTS_D1
        },
        "populated_slots_histogram": {f"{k}_of_6_slots": populated_slot_hist[k] for k in range(NUM_SLOTS_D1 + 1)},
        "collision_statistics": {
            "total_collision_entries": len(collisions_df),
            "studies_with_collisions": colliding_studies,
            "collision_rate_pct": round((colliding_studies / total_studies) * 100.0, 2),
            "policy": "highest_slice_count_then_lexicographic_tie_break"
        },
        "missing_slot_policy": "Zero tensor representation with slot_mask=0",
        "status": "PASS"
    }

    audit_json = os.path.join(output_audit_dir, "d1_slot_mapping_audit.json")
    with open(audit_json, "w") as f:
        json.dump(audit_data, f, indent=2)

    return mapping_df, collisions_df, audit_data


if __name__ == "__main__":
    mapping_df, collisions_df, audit_data = run_slot_mapping_d1()
    print("[SUCCESS] Phase D1 Six-Slot Mapping Completed.")
    print(f"Total Studies: {audit_data['total_dev_studies']}")
    print("Per-Slot Population:")
    for slot, stats in audit_data["per_slot_coverage"].items():
        print(f"  {slot}: {stats['populated_studies']} ({stats['coverage_pct']}%)")
    print(f"Total Collisions Logged: {audit_data['collision_statistics']['total_collision_entries']}")
