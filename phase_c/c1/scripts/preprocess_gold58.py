"""
Phase C1: Preprocessing Engine for Gold-58 Dataset into Pipeline C Format.

Preprocesses all 58 Gold studies (336 series) from raw DICOMs into
the standardized Pipeline C format:
  1. Coordinate-based slice sorting along acquisition normal
  2. RescaleSlope / RescaleIntercept correction
  3. Trilinear resampling to (3.0, 0.5, 0.5) mm
  4. Robust [P0.5, P99.5] percentile normalization -> [0.0, 1.0]
  5. Center pad/crop to fixed 3D tensor shape (32, 256, 256)
  6. float16 precision stored in compressed .npz ('volume', 'spacing')

Outputs are saved in phase_c/c1/processed_gold/ to guarantee strict isolation
from the 4,349 non-Gold development studies in phase_c/c1/processed/.
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

TARGET_SPACING = (3.0, 0.5, 0.5)  # (depth_mm, height_mm, width_mm)
TARGET_SHAPE = (32, 256, 256)     # (D, H, W)


def is_npz_valid(npz_path: str, target_shape=TARGET_SHAPE) -> bool:
    if not os.path.exists(npz_path) or os.path.getsize(npz_path) < 1000:
        return False
    try:
        with np.load(npz_path) as data:
            if "volume" not in data or "spacing" not in data:
                return False
            vol = data["volume"]
            if vol.shape != target_shape or vol.dtype != np.float16:
                return False
            if np.isnan(vol).any() or np.isinf(vol).any():
                return False
        return True
    except Exception:
        return False


def load_and_sort_dicom_series(series_path: str):
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

    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        volume = volume * slope + intercept

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
    curr_d_sp, curr_h_sp, curr_w_sp = spacing
    t_d_sp, t_h_sp, t_w_sp = target_spacing

    # 1. Trilinear resampling
    zoom_factors = (curr_d_sp / t_d_sp, curr_h_sp / t_h_sp, curr_w_sp / t_w_sp)
    resampled = scipy.ndimage.zoom(volume, zoom_factors, order=1)

    # 2. Percentile normalization [P0.5, P99.5] -> [0.0, 1.0]
    p_low, p_high = np.percentile(resampled, (0.5, 99.5))
    if p_high > p_low:
        norm_vol = np.clip((resampled - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
    else:
        norm_vol = np.zeros_like(resampled, dtype=np.float32)

    # 3. Center crop / pad
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

    return cropped.astype(np.float16)


def process_single_series(args_tuple):
    study_uid, ser_uid, raw_ser_path, out_dir, force = args_tuple
    out_file = os.path.join(out_dir, f"{ser_uid}.npz")

    if not force and is_npz_valid(out_file):
        size = os.path.getsize(out_file)
        return {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": ser_uid,
            "status": "SKIPPED_EXISTING",
            "stored_bytes": size,
            "error": None,
        }

    try:
        t0 = time.time()
        vol_raw, meta = load_and_sort_dicom_series(raw_ser_path)
        if vol_raw is None:
            return {
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": ser_uid,
                "status": "FAILED_NO_SLICES",
                "stored_bytes": 0,
                "error": "No slices loaded",
            }

        vol_c = run_pipeline_c_transform(vol_raw, meta["original_spacing"])
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(out_file, volume=vol_c, spacing=np.array(TARGET_SPACING, dtype=np.float32))
        elapsed = time.time() - t0
        stored_bytes = os.path.getsize(out_file)

        return {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": ser_uid,
            "status": "SUCCESS",
            "stored_bytes": stored_bytes,
            "elapsed_sec": elapsed,
            "error": None,
        }
    except Exception as e:
        return {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": ser_uid,
            "status": "FAILED_EXCEPTION",
            "stored_bytes": 0,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Gold-58 DICOMs to Pipeline C .npz")
    repo_root = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    parser.add_argument("--raw_dir", type=str, default=os.environ.get("RSNA_RAW_DICOM_DIR", os.path.join(repo_root, "data", "train_series")))
    parser.add_argument("--gold_csv", type=str, default=os.path.join(repo_root, "csv", "gold_studies.csv"))
    parser.add_argument("--out_dir", type=str, default=os.environ.get("RSNA_GOLD_OUT_DIR", os.path.join(repo_root, "phase_c", "c1", "processed_gold")))
    parser.add_argument("--manifest_out", type=str, default=os.path.join(repo_root, "phase_c", "c1", "manifests", "c1_gold58_manifest.csv"))
    parser.add_argument("--summary_out", type=str, default=os.path.join(repo_root, "phase_c", "c1", "output", "c1_gold58_execution_summary.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"Loading Gold-58 studies list from: {args.gold_csv}")
    gold_df = pd.read_csv(args.gold_csv)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())
    print(f"Loaded {len(gold_uids)} Gold study UIDs.")
    assert len(gold_uids) == 58, f"Expected exactly 58 Gold studies, got {len(gold_uids)}"

    # Discover series
    tasks = []
    for s_uid in sorted(gold_uids):
        s_raw_path = os.path.join(args.raw_dir, s_uid)
        if not os.path.isdir(s_raw_path):
            print(f"WARNING: Study raw directory missing: {s_raw_path}")
            continue
        ser_dirs = sorted([d for d in os.listdir(s_raw_path) if os.path.isdir(os.path.join(s_raw_path, d))])
        s_out_dir = os.path.join(args.out_dir, s_uid)
        for ser_uid in ser_dirs:
            raw_ser_path = os.path.join(s_raw_path, ser_uid)
            tasks.append((s_uid, ser_uid, raw_ser_path, s_out_dir, args.force))

    print(f"Total series tasks to process: {len(tasks)}")
    assert len(tasks) == 336, f"Expected 336 series tasks, got {len(tasks)}"

    os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    t_start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_series, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), start=1):
            res = future.result()
            results.append(res)
            if i % 50 == 0 or i == len(tasks):
                print(f"Progress: [{i}/{len(tasks)}] series processed...")

    total_wall_time = time.time() - t_start

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.manifest_out, index=False)
    print(f"Saved execution manifest to: {args.manifest_out}")

    # Summary
    success_count = (res_df["status"].isin(["SUCCESS", "SKIPPED_EXISTING"])).sum()
    failed_count = (res_df["status"].str.startswith("FAILED")).sum()
    total_stored_bytes = res_df["stored_bytes"].sum()

    # Validate output files on disk
    validation_failures = 0
    for res in results:
        s_uid = res["StudyInstanceUID"]
        ser_uid = res["SeriesInstanceUID"]
        f_path = os.path.join(args.out_dir, s_uid, f"{ser_uid}.npz")
        if not is_npz_valid(f_path):
            print(f"Validation failure for: {f_path}")
            validation_failures += 1

    summary = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_gold_studies": len(gold_uids),
        "total_series": len(tasks),
        "successful_series": int(success_count),
        "failed_series": int(failed_count),
        "validation_failures": int(validation_failures),
        "total_stored_bytes": int(total_stored_bytes),
        "total_stored_mb": float(total_stored_bytes / (1024 * 1024)),
        "wall_time_sec": float(total_wall_time),
        "target_spacing_mm": list(TARGET_SPACING),
        "target_shape": list(TARGET_SHAPE),
        "dtype": "float16",
        "output_directory": args.out_dir,
    }

    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {args.summary_out}")
    print(f"Summary: {summary}")

    if failed_count > 0 or validation_failures > 0:
        print(f"CRITICAL ERROR: {failed_count} failures, {validation_failures} invalid files.")
        sys.exit(1)
    else:
        print("ALL 336 GOLD-58 SERIES PREPROCESSED AND VALIDATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
