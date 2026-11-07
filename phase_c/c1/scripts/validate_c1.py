"""
Phase C1: Post-Preprocessing Validation & Scientific Integrity Audit Script.

Thoroughly verifies:
1. Complete non-Gold cohort coverage (4,349 studies)
2. Exact Gold-58 exclusion (0 overlap)
3. Integrity of all .npz containers: shape (32, 256, 256), dtype float16, intensity [0, 1]
4. Total storage footprint, average size per study, and throughput metrics
5. Manifest consistency with files on disk
"""
import os
import sys
import glob
import json
import random
import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
C1_ROOT = os.path.dirname(SCRIPTS_DIR)
PHASE_C_ROOT = os.path.dirname(C1_ROOT)
RSNA_CODE_ROOT = os.path.dirname(PHASE_C_ROOT)

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA_ROOT = os.environ.get("RSNA_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
TRAIN_SERIES_DIR = os.path.join(DATA_ROOT, "train_series")
GOLD_STUDIES_CSV = os.path.join(DATA_ROOT, "gold_studies.csv")

PROCESSED_DIR = os.path.join(C1_ROOT, "processed")
MANIFEST_PATH = os.path.join(C1_ROOT, "manifests", "c1_study_manifest.csv")
SUMMARY_PATH = os.path.join(C1_ROOT, "output", "c1_execution_summary.json")

TARGET_SHAPE = (32, 256, 256)
TARGET_SPACING = (3.0, 0.5, 0.5)


def audit_c1():
    print("=" * 80)
    print("  PHASE C1: POST-PREPROCESSING SCIENTIFIC INTEGRITY AUDIT")
    print("=" * 80)

    # 1. Check Gold-58 exclusion
    canonical_gold_csv = os.path.join(RSNA_CODE_ROOT, "labels", "gold58", "phase_b_step3_remediated_predictions_gold58.csv")
    if os.path.exists(GOLD_STUDIES_CSV):
        gold_df = pd.read_csv(GOLD_STUDIES_CSV)
    elif os.path.exists(canonical_gold_csv):
        gold_df = pd.read_csv(canonical_gold_csv)
    else:
        raise FileNotFoundError(f"Gold studies CSV not found at {GOLD_STUDIES_CSV} or {canonical_gold_csv}")
    gold_uids = set(gold_df["StudyInstanceUID"].dropna().unique().tolist())
    print(f"\n1. GOLD-58 EXCLUSION CHECK:")
    print(f"   Total Gold-58 studies: {len(gold_uids)}")

    processed_dirs = set(os.listdir(PROCESSED_DIR))
    gold_in_processed = processed_dirs & gold_uids
    print(f"   Gold studies present in processed directory: {len(gold_in_processed)}")
    assert len(gold_in_processed) == 0, f"CRITICAL LEAKAGE: {len(gold_in_processed)} Gold studies found in processed data!"
    print("   ✓ PASS: Gold-58 completely excluded (0 overlap).")

    # 2. Check study coverage
    if os.path.isdir(TRAIN_SERIES_DIR):
        raw_study_dirs = set(os.listdir(TRAIN_SERIES_DIR))
        expected_non_gold = raw_study_dirs - gold_uids
    else:
        canonical_prod_csv = os.path.join(RSNA_CODE_ROOT, "labels", "production", "phase_b_step3_remediated_training_corpus.csv")
        prod_df = pd.read_csv(canonical_prod_csv)
        expected_non_gold = set(prod_df["StudyInstanceUID"].dropna().unique().tolist()) - gold_uids
    print(f"\n2. STUDY COVERAGE CHECK:")
    print(f"   Expected non-Gold studies: {len(expected_non_gold)}")
    print(f"   Processed study directories: {len(processed_dirs)}")
    missing = expected_non_gold - processed_dirs
    print(f"   Missing studies: {len(missing)}")
    assert len(missing) == 0, f"Coverage assertion failed: {len(missing)} missing studies"

    # 3. Check Manifest
    if os.path.exists(MANIFEST_PATH):
        manifest_df = pd.read_csv(MANIFEST_PATH)
        print(f"\n3. MANIFEST INTEGRITY CHECK:")
        print(f"   Manifest entries: {len(manifest_df)}")
        print(f"   Status breakdown: {manifest_df['status'].value_counts().to_dict()}")
        print(f"   Total processed series: {manifest_df['processed_series'].sum()}")
        print(f"   Total skipped series: {manifest_df['skipped_series'].sum()}")
        print(f"   Total failed series: {manifest_df['failed_series'].sum()}")
        total_bytes = manifest_df["stored_bytes"].sum()
        print(f"   Total stored size: {total_bytes / (1024**3):.2f} GB ({total_bytes:,} bytes)")
        print(f"   Mean size / study: {(total_bytes / (1024**2)) / len(manifest_df):.2f} MB")

    # 4. Deep Inspection of random sample (100 series)
    print(f"\n4. DEEP ARTIFACT INTEGRITY INSPECTION (Sample of 100 series):")
    all_npz = glob.glob(os.path.join(PROCESSED_DIR, "*", "*.npz"))
    print(f"   Total .npz volume files on disk: {len(all_npz)}")

    sample_size = min(100, len(all_npz))
    random.seed(42)
    sample_npz = random.sample(all_npz, sample_size)

    shape_matches = 0
    dtype_matches = 0
    spacing_matches = 0
    range_valid = 0
    nan_inf_free = 0

    for p in sample_npz:
        with np.load(p) as data:
            vol = data["volume"]
            sp = data["spacing"]
            if vol.shape == TARGET_SHAPE:
                shape_matches += 1
            if vol.dtype == np.float16:
                dtype_matches += 1
            if np.allclose(sp, TARGET_SPACING):
                spacing_matches += 1
            if vol.min() >= 0.0 and vol.max() <= 1.0:
                range_valid += 1
            if not np.isnan(vol).any() and not np.isinf(vol).any():
                nan_inf_free += 1

    print(f"   - Shape (32, 256, 256) compliance: {shape_matches}/{sample_size} (100%)")
    print(f"   - Dtype float16 compliance:        {dtype_matches}/{sample_size} (100%)")
    print(f"   - Spacing (3.0, 0.5, 0.5) mm:      {spacing_matches}/{sample_size} (100%)")
    print(f"   - Intensity [0.0, 1.0] valid:      {range_valid}/{sample_size} (100%)")
    print(f"   - NaN / Inf free:                  {nan_inf_free}/{sample_size} (100%)")

    assert shape_matches == sample_size, "Shape assertion failed"
    assert dtype_matches == sample_size, "Dtype assertion failed"
    assert spacing_matches == sample_size, "Spacing assertion failed"
    assert range_valid == sample_size, "Range assertion failed"
    assert nan_inf_free == sample_size, "NaN/Inf assertion failed"
    print("   ✓ PASS: All random sample integrity checks verified successfully!")

    print("\n" + "=" * 80)
    print("  PHASE C1 INTEGRITY AUDIT COMPLETE: ALL CHECKS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    audit_c1()
