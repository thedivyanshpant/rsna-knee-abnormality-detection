import os
import glob
import time
import json
import resource
import pydicom
import numpy as np
import scipy.ndimage
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.environ.get('RSNA_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
DATA_DIR = os.environ.get('RSNA_DATA_ROOT', os.path.join(REPO_ROOT, 'data'))
TRAIN_SERIES_DIR = os.path.join(DATA_DIR, 'train_series')
REPO_ROOT = os.environ.get('RSNA_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
C0_ROOT = os.environ.get('RSNA_C0_ROOT', os.path.join(REPO_ROOT, 'phase_c', 'c0'))
OUTPUT_DIR = os.path.join(C0_ROOT, 'output')
PROCESSED_DIR = os.path.join(C0_ROOT, 'processed')

TARGET_SPACING = (3.0, 0.5, 0.5) # (depth_mm, height_mm, width_mm)
TARGET_SHAPE = (32, 256, 256)    # Fixed 3D tensor shape for Pipeline C

def load_and_sort_dicom_series(series_path):
    dcm_files = sorted(glob.glob(os.path.join(series_path, '*.dcm')))
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
        
    has_pos = all(hasattr(s, 'ImagePositionPatient') and hasattr(s, 'ImageOrientationPatient') for s in slices)
    
    if has_pos:
        iop = slices[0].ImageOrientationPatient
        r = np.array(iop[:3])
        c = np.array(iop[3:])
        normal = np.cross(r, c)
        slices.sort(key=lambda s: np.dot(np.array(s.ImagePositionPatient), normal))
    elif all(hasattr(s, 'InstanceNumber') for s in slices):
        slices.sort(key=lambda s: int(s.InstanceNumber))
        
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0) # (D, H, W)
    
    slope = getattr(slices[0], 'RescaleSlope', 1.0)
    intercept = getattr(slices[0], 'RescaleIntercept', 0.0)
    if slope != 1.0 or intercept != 0.0:
        volume = volume * slope + intercept
        
    ps = getattr(slices[0], 'PixelSpacing', [0.5, 0.5])
    row_sp = float(ps[0])
    col_sp = float(ps[1])
    
    if len(slices) > 1 and has_pos:
        p0 = np.array(slices[0].ImagePositionPatient)
        p1 = np.array(slices[1].ImagePositionPatient)
        slice_sp = float(np.linalg.norm(p1 - p0))
    else:
        slice_sp = float(getattr(slices[0], 'SliceThickness', 3.0))
        
    spacing = (slice_sp, row_sp, col_sp)
    
    meta = {
        'num_slices': len(slices),
        'raw_bytes': raw_bytes,
        'original_shape': list(volume.shape),
        'original_spacing': spacing,
        'series_instance_uid': getattr(slices[0], 'SeriesInstanceUID', os.path.basename(series_path)),
        'image_orientation': getattr(slices[0], 'ImageOrientationPatient', None)
    }
    
    return volume, meta

def run_pipeline_a(volume, spacing, meta):
    """Pipeline A: Minimal volume construction"""
    t0 = time.time()
    vol_a = volume.astype(np.float32)
    elapsed = time.time() - t0
    return vol_a, spacing, elapsed

def run_pipeline_b(volume, spacing, meta, target_spacing=TARGET_SPACING):
    """Pipeline B: Standardized Spacing & Robust Intensity Normalization"""
    t0 = time.time()
    curr_d_sp, curr_h_sp, curr_w_sp = spacing
    t_d_sp, t_h_sp, t_w_sp = target_spacing
    
    zoom_factors = (curr_d_sp / t_d_sp, curr_h_sp / t_h_sp, curr_w_sp / t_w_sp)
    resampled = scipy.ndimage.zoom(volume, zoom_factors, order=1)
    
    p_low, p_high = np.percentile(resampled, (0.5, 99.5))
    if p_high > p_low:
        norm_vol = np.clip((resampled - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
    else:
        norm_vol = np.zeros_like(resampled, dtype=np.float32)
        
    elapsed = time.time() - t0
    return norm_vol, target_spacing, elapsed

def run_pipeline_c(norm_vol_b, spacing, meta, target_shape=TARGET_SHAPE):
    """Pipeline C: Training-Ready Fixed 3D Tensor & float16 Precision"""
    t0 = time.time()
    d, h, w = norm_vol_b.shape
    td, th, tw = target_shape
    
    pad_d = max(0, td - d)
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    
    work_vol = norm_vol_b
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pd0, pd1 = pad_d // 2, pad_d - pad_d // 2
        ph0, ph1 = pad_h // 2, pad_h - pad_h // 2
        pw0, pw1 = pad_w // 2, pad_w - pad_w // 2
        work_vol = np.pad(work_vol, ((pd0, pd1), (ph0, ph1), (pw0, pw1)), mode='constant', constant_values=0)
        d, h, w = work_vol.shape
        
    cd0 = (d - td) // 2
    ch0 = (h - th) // 2
    cw0 = (w - tw) // 2
    cropped = work_vol[cd0:cd0+td, ch0:ch0+th, cw0:cw0+tw]
    
    res_float16 = cropped.astype(np.float16)
    elapsed = time.time() - t0
    return res_float16, spacing, elapsed

def process_single_study(study_uid):
    study_path = os.path.join(TRAIN_SERIES_DIR, study_uid)
    series_dirs = sorted([d for d in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, d))])
    
    study_records = []
    
    # Target directories
    dir_a = os.path.join(PROCESSED_DIR, 'pipeline_a', study_uid)
    dir_b = os.path.join(PROCESSED_DIR, 'pipeline_b', study_uid)
    dir_c = os.path.join(PROCESSED_DIR, 'pipeline_c', study_uid)
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)
    os.makedirs(dir_c, exist_ok=True)
    
    for ser in series_dirs:
        ser_path = os.path.join(study_path, ser)
        t_load_0 = time.time()
        vol_raw, meta = load_and_sort_dicom_series(ser_path)
        load_time = time.time() - t_load_0
        
        if vol_raw is None:
            continue
            
        raw_bytes = meta['raw_bytes']
        
        # 1. Pipeline A
        vol_a, sp_a, t_a = run_pipeline_a(vol_raw, meta['original_spacing'], meta)
        file_a = os.path.join(dir_a, f"{ser}.npz")
        np.savez_compressed(file_a, volume=vol_a, spacing=np.array(sp_a))
        size_a = os.path.getsize(file_a)
        ram_a = vol_a.nbytes
        
        # 2. Pipeline B
        vol_b, sp_b, t_b = run_pipeline_b(vol_raw, meta['original_spacing'], meta)
        file_b = os.path.join(dir_b, f"{ser}.npz")
        np.savez_compressed(file_b, volume=vol_b, spacing=np.array(sp_b))
        size_b = os.path.getsize(file_b)
        ram_b = vol_b.nbytes
        
        # 3. Pipeline C
        vol_c, sp_c, t_c = run_pipeline_c(vol_b, sp_b, meta)
        file_c = os.path.join(dir_c, f"{ser}.npz")
        np.savez_compressed(file_c, volume=vol_c, spacing=np.array(sp_c))
        size_c = os.path.getsize(file_c)
        ram_c = vol_c.nbytes
        
        study_records.append({
            'StudyInstanceUID': study_uid,
            'SeriesInstanceUID': ser,
            'raw_slices': meta['num_slices'],
            'raw_bytes': raw_bytes,
            'load_time_sec': load_time,
            
            # Pipeline A
            'shape_a': str(vol_a.shape),
            'dtype_a': str(vol_a.dtype),
            'stored_bytes_a': size_a,
            'ram_bytes_a': ram_a,
            'time_sec_a': load_time + t_a,
            'reduction_ratio_a': raw_bytes / size_a if size_a > 0 else 0,
            
            # Pipeline B
            'shape_b': str(vol_b.shape),
            'dtype_b': str(vol_b.dtype),
            'stored_bytes_b': size_b,
            'ram_bytes_b': ram_b,
            'time_sec_b': load_time + t_b,
            'reduction_ratio_b': raw_bytes / size_b if size_b > 0 else 0,
            
            # Pipeline C
            'shape_c': str(vol_c.shape),
            'dtype_c': str(vol_c.dtype),
            'stored_bytes_c': size_c,
            'ram_bytes_c': ram_c,
            'time_sec_c': load_time + t_b + t_c,
            'reduction_ratio_c': raw_bytes / size_c if size_c > 0 else 0,
        })
        
    return study_records

def main():
    print("=== C0.3: EXECUTING PREPROCESSING PIPELINES OVER ~10 GB PILOT ===")
    t_start = time.time()
    
    pilot_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'pilot_selection.csv'))
    selected_studies = pilot_df[pilot_df['selected']]['StudyInstanceUID'].tolist()
    print(f"Total Selected Pilot Studies: {len(selected_studies)}")
    
    num_workers = min(32, os.cpu_count() or 4)
    print(f"Using {num_workers} CPU worker processes...")
    
    all_metrics = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_study, suid): suid for suid in selected_studies}
        for fut in as_completed(futures):
            suid = futures[fut]
            try:
                records = fut.result()
                all_metrics.extend(records)
                completed += 1
                if completed % 10 == 0 or completed == len(selected_studies):
                    print(f"Processed {completed}/{len(selected_studies)} studies ({len(all_metrics)} series completed)... [Elapsed: {time.time() - t_start:.1f}s]")
            except Exception as e:
                print(f"Error processing study {suid}: {e}")
                
    df_metrics = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(OUTPUT_DIR, 'preprocessing_metrics.csv')
    df_metrics.to_csv(metrics_path, index=False)
    
    total_time = time.time() - t_start
    print(f"\nAll 80 pilot studies successfully processed in {total_time:.2f} s ({total_time / 60:.2f} min).")
    print(f"Saved: {metrics_path} ({len(df_metrics)} series records)")

if __name__ == '__main__':
    main()
