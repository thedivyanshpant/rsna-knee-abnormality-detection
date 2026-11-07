import os
import glob
import json
import time
import pandas as pd
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.environ.get('RSNA_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
DATA_DIR = os.environ.get('RSNA_DATA_ROOT', os.path.join(REPO_ROOT, 'data'))
TRAIN_SERIES_DIR = os.path.join(DATA_DIR, 'train_series')
REPO_ROOT = os.environ.get('RSNA_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
OUTPUT_DIR = os.environ.get('RSNA_C0_OUTPUT', os.path.join(REPO_ROOT, 'phase_c', 'c0', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def discover():
    t0 = time.time()
    print("=== C0.1: MRI DATASET DISCOVERY & INVENTORY ===")
    
    # 1. Load CSV metadata
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    train_series_df = pd.read_csv(os.path.join(DATA_DIR, 'train_series.csv'))
    gold_df = pd.read_csv(os.path.join(DATA_DIR, 'gold_studies.csv'))
    gold_uids = set(gold_df['StudyInstanceUID'].dropna().tolist())
    
    print(f"Total records in train.csv: {len(train_df)}")
    print(f"Total series in train_series.csv: {len(train_series_df)}")
    print(f"Total gold studies to exclude from pilot/training tuning: {len(gold_uids)}")
    
    # 2. Inspect study directories
    available_studies = sorted(os.listdir(TRAIN_SERIES_DIR))
    print(f"Discovered study directories in train_series/: {len(available_studies)}")
    
    study_sizes = []
    study_series_counts = []
    total_raw_bytes = 0
    total_dicom_files = 0
    
    # Let's compute study stats
    study_records = []
    
    for s_idx, study_uid in enumerate(available_studies):
        study_path = os.path.join(TRAIN_SERIES_DIR, study_uid)
        series_dirs = [d for d in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, d))]
        
        study_bytes = 0
        study_slices = 0
        for ser in series_dirs:
            ser_path = os.path.join(study_path, ser)
            files = [f for f in os.listdir(ser_path) if f.endswith('.dcm') or not os.path.isdir(os.path.join(ser_path, f))]
            study_slices += len(files)
            for f in files:
                study_bytes += os.path.getsize(os.path.join(ser_path, f))
                
        total_raw_bytes += study_bytes
        total_dicom_files += study_slices
        study_sizes.append(study_bytes)
        study_series_counts.append(len(series_dirs))
        
        is_gold = study_uid in gold_uids
        study_records.append({
            'StudyInstanceUID': study_uid,
            'series_count': len(series_dirs),
            'slice_count': study_slices,
            'raw_bytes': study_bytes,
            'raw_mb': study_bytes / (1024**2),
            'is_gold': is_gold
        })
        
        if (s_idx + 1) % 1000 == 0 or s_idx == len(available_studies) - 1:
            print(f"Scanned {s_idx + 1}/{len(available_studies)} studies... ({total_raw_bytes / (1024**3):.2f} GB accumulated)")
            
    df_studies = pd.DataFrame(study_records)
    
    # Plane distribution from train_series_df
    plane_counts = train_series_df['Anatomical_Plane'].value_counts().to_dict()
    fluid_counts = train_series_df['Fluid_Sensitive'].value_counts().to_dict()
    fat_counts = train_series_df['Fat_Suppression'].value_counts().to_dict()
    
    # 3. Sample DICOM properties (sample 100 random studies)
    import pydicom
    np.random.seed(42)
    sample_indices = np.random.choice(len(available_studies), size=min(100, len(available_studies)), replace=False)
    
    dimensions_counter = Counter()
    spacing_samples = []
    thickness_samples = []
    bit_depth_counter = Counter()
    photometric_counter = Counter()
    
    print("Sampling DICOM metadata across 100 studies...")
    for idx in sample_indices:
        suid = available_studies[idx]
        spath = os.path.join(TRAIN_SERIES_DIR, suid)
        for ser in os.listdir(spath):
            ser_path = os.path.join(spath, ser)
            dcm_files = glob.glob(os.path.join(ser_path, '*.dcm'))
            if dcm_files:
                try:
                    ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
                    rows = getattr(ds, 'Rows', None)
                    cols = getattr(ds, 'Columns', None)
                    if rows and cols:
                        dimensions_counter[f"{rows}x{cols}"] += 1
                    ps = getattr(ds, 'PixelSpacing', None)
                    if ps:
                        spacing_samples.append([float(ps[0]), float(ps[1])])
                    th = getattr(ds, 'SliceThickness', None)
                    if th:
                        thickness_samples.append(float(th))
                    ba = getattr(ds, 'BitsAllocated', None)
                    bs = getattr(ds, 'BitsStored', None)
                    if ba and bs:
                        bit_depth_counter[f"{ba}/{bs}"] += 1
                    pm = getattr(ds, 'PhotometricInterpretation', 'UNKNOWN')
                    photometric_counter[str(pm)] += 1
                except Exception as e:
                    pass

    spacing_arr = np.array(spacing_samples) if spacing_samples else np.array([[0.5, 0.5]])
    thickness_arr = np.array(thickness_samples) if thickness_samples else np.array([3.0])
    
    inventory = {
        'dataset_root': DATA_DIR,
        'train_series_path': TRAIN_SERIES_DIR,
        'discovery_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'total_study_directories': len(available_studies),
        'total_non_gold_studies': int((~df_studies['is_gold']).sum()),
        'total_gold_studies_in_dir': int(df_studies['is_gold'].sum()),
        'total_series_count': int(df_studies['series_count'].sum()),
        'total_dicom_files': int(total_dicom_files),
        'total_raw_storage_bytes': int(total_raw_bytes),
        'total_raw_storage_gb': float(round(total_raw_bytes / (1024**3), 3)),
        'per_study_statistics': {
            'mean_raw_bytes': float(df_studies['raw_bytes'].mean()),
            'median_raw_bytes': float(df_studies['raw_bytes'].median()),
            'min_raw_bytes': int(df_studies['raw_bytes'].min()),
            'max_raw_bytes': int(df_studies['raw_bytes'].max()),
            'mean_raw_mb': float(df_studies['raw_mb'].mean()),
            'median_raw_mb': float(df_studies['raw_mb'].median()),
            'mean_series_count': float(df_studies['series_count'].mean()),
            'median_series_count': float(df_studies['series_count'].median()),
            'mean_slices_per_study': float(df_studies['slice_count'].mean()),
            'median_slices_per_study': float(df_studies['slice_count'].median())
        },
        'plane_distribution': plane_counts,
        'fluid_sensitive_distribution': {str(k): int(v) for k, v in fluid_counts.items()},
        'fat_suppression_distribution': {str(k): int(v) for k, v in fat_counts.items()},
        'sampled_image_dimensions': dict(dimensions_counter.most_common(10)),
        'sampled_pixel_spacing': {
            'mean_row_col_spacing_mm': [float(round(spacing_arr[:, 0].mean(), 4)), float(round(spacing_arr[:, 1].mean(), 4))],
            'min_spacing_mm': [float(round(spacing_arr[:, 0].min(), 4)), float(round(spacing_arr[:, 1].min(), 4))],
            'max_spacing_mm': [float(round(spacing_arr[:, 0].max(), 4)), float(round(spacing_arr[:, 1].max(), 4))]
        },
        'sampled_slice_thickness': {
            'mean_thickness_mm': float(round(thickness_arr.mean(), 2)),
            'median_thickness_mm': float(round(float(np.median(thickness_arr)), 2)),
            'min_thickness_mm': float(round(thickness_arr.min(), 2)),
            'max_thickness_mm': float(round(thickness_arr.max(), 2))
        },
        'sampled_bit_depths': dict(bit_depth_counter),
        'sampled_photometric_interpretations': dict(photometric_counter)
    }
    
    out_json = os.path.join(OUTPUT_DIR, 'dataset_inventory.json')
    with open(out_json, 'w') as f:
        json.dump(inventory, f, indent=2)
        
    print(f"\nInventory saved to: {out_json}")
    print(f"Total raw dataset size: {inventory['total_raw_storage_gb']} GB across {inventory['total_study_directories']} studies.")
    print(f"Mean study size: {inventory['per_study_statistics']['mean_raw_mb']:.2f} MB, Median: {inventory['per_study_statistics']['median_raw_mb']:.2f} MB")
    print(f"Elapsed time: {time.time() - t0:.2f} s")

if __name__ == '__main__':
    discover()
