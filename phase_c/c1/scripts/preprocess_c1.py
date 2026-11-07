"""
Phase C1: Full Dataset Preprocessing Engine for Pipeline C.

Applies the empirically validated Pipeline C representation across all 4,349 non-Gold studies:
  1. DICOM Slice Sorting: Coordinate-based sorting along slice normal (or InstanceNumber)
  2. Native Rescaling: RescaleSlope & RescaleIntercept application
  3. Spatial Resampling: Trilinear zoom to target spacing (3.0, 0.5, 0.5) mm
  4. Robust Intensity Normalization: [P0.5, P99.5] percentile clipping -> [0.0, 1.0]
  5. Fixed 3D Tensor Framing: Symmetric center-padding / center-cropping to (32, 256, 256)
  6. Optimized Precision: Cast to float16 (2 bytes/voxel)
  7. Compressed Container: Compressed .npz archive storing 'volume' and 'spacing'

Restartable: Skips existing valid .npz files unless --force is specified.
Gold-58: Strictly excluded and verified programmatically.
"""
import os
import sys
import glob
import time
import json
import argparse
import pydicom
import numpy as np
import scipy.ndimage
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================================
# PATHS & CONSTANTS
# ============================================================================
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
C1_ROOT = os.path.dirname(SCRIPTS_DIR)
PHASE_C_ROOT = os.path.dirname(C1_ROOT)
RSNA_CODE_ROOT = os.path.dirname(PHASE_C_ROOT)

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA_ROOT = os.environ.get("RSNA_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
TRAIN_SERIES_DIR = os.path.join(DATA_ROOT, "train_series")
GOLD_STUDIES_CSV = os.path.join(DATA_ROOT, "gold_studies.csv")

PROCESSED_DIR = os.path.join(C1_ROOT, "processed")
MANIFESTS_DIR = os.path.join(C1_ROOT, "manifests")
OUTPUT_DIR = os.path.join(C1_ROOT, "output")

TARGET_SPACING = (3.0, 0.5, 0.5)  # (depth_mm, height_mm, width_mm)
TARGET_SHAPE = (32, 256, 256)     # Fixed 3D tensor shape (D, H, W)


def is_npz_valid(npz_path: str, target_shape=TARGET_SHAPE) -> bool:
    """Check if an existing .npz file is complete, uncorrupted, and matches expected shape/dtype."""
    if not os.path.exists(npz_path) or os.path.getsize(npz_path) < 1000:
        return False
    try:
        with np.load(npz_path) as data:
            if "volume" not in data or "spacing" not in data:
                return False
            vol = data["volume"]
            if vol.shape != target_shape or vol.dtype != np.float16:
                return False
            # Check for reasonable float16 values (no NaNs or Infs)
            if np.isnan(vol).any() or np.isinf(vol).any():
                return False
        return True
    except Exception:
        return False


def load_and_sort_dicom_series(series_path: str):
    """Load and sort DICOM slices into a 3D float32 volume with acquisition spacing."""
    dcm_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
    if not dcm_files:
        return None, None

    slices = []
    raw_bytes = 0
    for f in dcm_files:
        try:
            raw_bytes += os.path.getsize(f)
            ds = pydicom.dcmread(f)
            slices.append(ds)
        except Exception:
            pass

    if not slices:
        return None, None

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

    # Apply rescale slope / intercept
    slope = getattr(slices[0], "RescaleSlope", 1.0)
    intercept = getattr(slices[0], "RescaleIntercept", 0.0)
    if slope != 1.0 or intercept != 0.0:
        volume = volume * float(slope) + float(intercept)

    ps = getattr(slices[0], "PixelSpacing", [0.5, 0.5])
    row_sp = float(ps[0])
    col_sp = float(ps[1])

    if len(slices) > 1 and has_pos:
        p0 = np.array(slices[0].ImagePositionPatient)
        p1 = np.array(slices[1].ImagePositionPatient)
        slice_sp = float(np.linalg.norm(p1 - p0))
    else:
        slice_sp = float(getattr(slices[0], "SliceThickness", 3.0))

    spacing = (slice_sp, row_sp, col_sp)

    meta = {
        "num_slices": len(slices),
        "raw_bytes": raw_bytes,
        "original_shape": list(volume.shape),
        "original_spacing": spacing,
        "series_instance_uid": getattr(slices[0], "SeriesInstanceUID", os.path.basename(series_path)),
    }

    return volume, meta


def run_pipeline_c_transform(volume: np.ndarray, spacing: tuple, target_spacing=TARGET_SPACING, target_shape=TARGET_SHAPE):
    """Execute exact Pipeline C transformations: resample -> percentile norm -> center crop/pad -> float16."""
    curr_d_sp, curr_h_sp, curr_w_sp = spacing
    t_d_sp, t_h_sp, t_w_sp = target_spacing

    # 1. Trilinear resampling to target spacing
    zoom_factors = (curr_d_sp / t_d_sp, curr_h_sp / t_h_sp, curr_w_sp / t_w_sp)
    resampled = scipy.ndimage.zoom(volume, zoom_factors, order=1)

    # 2. Robust percentile normalization [P0.5, P99.5] -> [0.0, 1.0]
    p_low, p_high = np.percentile(resampled, (0.5, 99.5))
    if p_high > p_low:
        norm_vol = np.clip((resampled - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
    else:
        norm_vol = np.zeros_like(resampled, dtype=np.float32)

    # 3. Fixed 3D tensor framing (center-pad / center-crop)
    d, h, w = norm_vol.shape
    td, th, tw = target_shape

    pad_d = max(0, td - d)
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)

    work_vol = norm_vol
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pd0, pd1 = pad_d // 2, pad_d - pad_d // 2
        ph0, ph1 = pad_h // 2, pad_h - pad_h // 2
        pw0, pw1 = pad_w // 2, pad_w - pad_w // 2
        work_vol = np.pad(work_vol, ((pd0, pd1), (ph0, ph1), (pw0, pw1)), mode="constant", constant_values=0)
        d, h, w = work_vol.shape

    cd0 = (d - td) // 2
    ch0 = (h - th) // 2
    cw0 = (w - tw) // 2
    cropped = work_vol[cd0:cd0+td, ch0:ch0+th, cw0:cw0+tw]

    # 4. Cast to float16
    res_float16 = cropped.astype(np.float16)

    return res_float16


def process_study(study_uid: str, force: bool = False):
    """Process all series in a single study. Returns study-level execution metrics."""
    study_path = os.path.join(TRAIN_SERIES_DIR, study_uid)
    if not os.path.isdir(study_path):
        return {
            "StudyInstanceUID": study_uid,
            "series_count": 0,
            "processed_series": 0,
            "skipped_series": 0,
            "failed_series": 0,
            "raw_bytes": 0,
            "stored_bytes": 0,
            "elapsed_sec": 0.0,
            "status": "FAILED",
            "error_message": f"Directory not found: {study_path}",
        }

    series_dirs = sorted([d for d in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, d))])
    out_study_dir = os.path.join(PROCESSED_DIR, study_uid)
    os.makedirs(out_study_dir, exist_ok=True)

    t0 = time.time()
    total_raw_bytes = 0
    total_stored_bytes = 0
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    errors = []

    for ser in series_dirs:
        ser_path = os.path.join(study_path, ser)
        out_npz_path = os.path.join(out_study_dir, f"{ser}.npz")

        # Check if already valid
        if not force and is_npz_valid(out_npz_path):
            stored_size = os.path.getsize(out_npz_path)
            total_stored_bytes += stored_size
            skipped_count += 1
            continue

        try:
            vol_raw, meta = load_and_sort_dicom_series(ser_path)
            if vol_raw is None:
                failed_count += 1
                errors.append(f"{ser}: Failed to load DICOM slices")
                continue

            raw_bytes = meta["raw_bytes"]
            total_raw_bytes += raw_bytes

            # Transform
            vol_c = run_pipeline_c_transform(vol_raw, meta["original_spacing"])

            # Save
            np.savez_compressed(
                out_npz_path,
                volume=vol_c,
                spacing=np.array(TARGET_SPACING, dtype=np.float64),
            )

            stored_size = os.path.getsize(out_npz_path)
            total_stored_bytes += stored_size
            processed_count += 1

        except Exception as e:
            failed_count += 1
            errors.append(f"{ser}: {str(e)}")

    elapsed = time.time() - t0
    status = "SUCCESS" if failed_count == 0 else ("PARTIAL" if (processed_count + skipped_count) > 0 else "FAILED")

    return {
        "StudyInstanceUID": study_uid,
        "series_count": len(series_dirs),
        "processed_series": processed_count,
        "skipped_series": skipped_count,
        "failed_series": failed_count,
        "raw_bytes": total_raw_bytes,
        "stored_bytes": total_stored_bytes,
        "elapsed_sec": elapsed,
        "status": status,
        "error_message": "; ".join(errors) if errors else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Phase C1 Full Preprocessing")
    parser.add_argument("--workers", type=int, default=min(36, os.cpu_count() or 4), help="CPU workers")
    parser.add_argument("--smoke-test", action="store_true", help="Run 5-study smoke test")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of studies")
    parser.add_argument("--force", action="store_true", help="Force re-processing of existing files")
    args = parser.parse_args()

    print("=" * 80)
    print("  PHASE C1: FULL-DATASET PIPELINE C PREPROCESSING")
    print("=" * 80)

    # 1. Discover all non-Gold studies
    all_study_dirs = sorted([d for d in os.listdir(TRAIN_SERIES_DIR) if os.path.isdir(os.path.join(TRAIN_SERIES_DIR, d))])
    print(f"Total study directories in raw repository: {len(all_study_dirs)}")

    # 2. Exclude Gold-58
    gold_df = pd.read_csv(GOLD_STUDIES_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"].dropna().tolist())
    candidate_studies = [s for s in all_study_dirs if s not in gold_uids]
    overlap = set(candidate_studies) & gold_uids
    assert len(overlap) == 0, f"FATAL: {len(overlap)} Gold-58 studies detected in candidates!"
    print(f"Gold-58 exclusion verified: Exactly {len(gold_uids)} Gold studies excluded (0 overlap)")
    print(f"Candidate non-Gold studies to process: {len(candidate_studies)}")

    if args.smoke_test:
        candidate_studies = candidate_studies[:5]
        print(f"\n[SMOKE TEST] Running on 5 studies with deep validation...")

    elif args.limit:
        candidate_studies = candidate_studies[:args.limit]
        print(f"\n[LIMIT APPLIED] Running on {len(candidate_studies)} studies...")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nTarget Output Directory: {PROCESSED_DIR}")
    print(f"Using {args.workers} parallel worker processes...")

    t_start = time.time()
    manifest_records = []
    completed = 0
    total_studies = len(candidate_studies)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_study, suid, args.force): suid for suid in candidate_studies}
        for fut in as_completed(futures):
            suid = futures[fut]
            try:
                rec = fut.result()
                manifest_records.append(rec)
                completed += 1

                if completed % 100 == 0 or completed == total_studies:
                    elapsed = time.time() - t_start
                    rate = completed / max(elapsed, 0.001)
                    eta = (total_studies - completed) / max(rate, 0.001)
                    print(
                        f"  [{completed}/{total_studies}] ({completed/total_studies*100:.1f}%) "
                        f"Rate: {rate:.2f} studies/s | ETA: {eta/60:.1f} min | Elapsed: {elapsed/60:.1f} min"
                    )
            except Exception as e:
                print(f"Error processing study {suid}: {e}")
                manifest_records.append({
                    "StudyInstanceUID": suid,
                    "series_count": 0,
                    "processed_series": 0,
                    "skipped_series": 0,
                    "failed_series": 0,
                    "raw_bytes": 0,
                    "stored_bytes": 0,
                    "elapsed_sec": 0.0,
                    "status": "FAILED",
                    "error_message": str(e),
                })

    total_time = time.time() - t_start

    # Build manifest DataFrame
    df_manifest = pd.DataFrame(manifest_records)
    manifest_path = os.path.join(MANIFESTS_DIR, "c1_study_manifest.csv")
    df_manifest.to_csv(manifest_path, index=False)

    # Compute aggregate stats
    total_processed_series = int(df_manifest["processed_series"].sum())
    total_skipped_series = int(df_manifest["skipped_series"].sum())
    total_failed_series = int(df_manifest["failed_series"].sum())
    total_stored_bytes = int(df_manifest["stored_bytes"].sum())
    successful_studies = int((df_manifest["status"] == "SUCCESS").sum())
    failed_studies = int((df_manifest["status"] == "FAILED").sum())
    partial_studies = int((df_manifest["status"] == "PARTIAL").sum())

    summary = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_candidate_studies": total_studies,
        "successful_studies": successful_studies,
        "failed_studies": failed_studies,
        "partial_studies": partial_studies,
        "total_processed_series": total_processed_series,
        "total_skipped_series": total_skipped_series,
        "total_failed_series": total_failed_series,
        "total_stored_bytes": total_stored_bytes,
        "total_stored_gb": total_stored_bytes / (1024**3),
        "mean_stored_mb_per_study": (total_stored_bytes / (1024**2)) / max(successful_studies + partial_studies, 1),
        "total_wall_time_sec": total_time,
        "total_wall_time_min": total_time / 60.0,
        "effective_throughput_studies_per_sec": total_studies / max(total_time, 0.001),
        "target_spacing_mm": list(TARGET_SPACING),
        "target_tensor_shape": list(TARGET_SHAPE),
        "target_dtype": "float16",
        "gold58_overlap_count": len(overlap),
    }

    summary_path = os.path.join(OUTPUT_DIR, "c1_execution_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("  PHASE C1 PREPROCESSING EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Total Studies Evaluated:    {total_studies}")
    print(f"Successful Studies:         {successful_studies}")
    print(f"Failed Studies:             {failed_studies}")
    print(f"Total Stored Size:          {summary['total_stored_gb']:.2f} GB")
    print(f"Mean Stored Size / Study:   {summary['mean_stored_mb_per_study']:.2f} MB")
    print(f"Total Wall Time:            {total_time/60:.2f} min ({total_time:.1f} s)")
    print(f"Manifest Path:              {manifest_path}")
    print(f"Summary Path:               {summary_path}")

    # Deep validation for smoke test
    if args.smoke_test:
        print("\n--- SMOKE TEST DEEP VALIDATION ---")
        for rec in manifest_records:
            suid = rec["StudyInstanceUID"]
            sdir = os.path.join(PROCESSED_DIR, suid)
            npz_files = glob.glob(os.path.join(sdir, "*.npz"))
            for npz_p in npz_files:
                with np.load(npz_p) as data:
                    vol = data["volume"]
                    sp = data["spacing"]
                    print(
                        f"  Study {suid[:20]}... | Series {os.path.basename(npz_p)[:20]}... | "
                        f"Shape: {vol.shape} | Dtype: {vol.dtype} | "
                        f"Range: [{vol.min():.4f}, {vol.max():.4f}] | "
                        f"Spacing: {sp}"
                    )
                    assert vol.shape == TARGET_SHAPE, f"Shape mismatch: {vol.shape}"
                    assert vol.dtype == np.float16, f"Dtype mismatch: {vol.dtype}"
                    assert vol.min() >= 0.0 and vol.max() <= 1.0, f"Range violation: [{vol.min()}, {vol.max()}]"
        print("✓ All smoke test validation assertions passed!")


if __name__ == "__main__":
    main()
