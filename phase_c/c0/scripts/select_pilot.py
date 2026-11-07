import os
import random
import json
import pandas as pd
import numpy as np

REPO_ROOT = os.environ.get('RSNA_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
DATA_DIR = os.environ.get('RSNA_DATA_ROOT', os.path.join(REPO_ROOT, 'data'))
TRAIN_SERIES_DIR = os.path.join(DATA_DIR, 'train_series')
REPO_ROOT = os.environ.get('RSNA_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
OUTPUT_DIR = os.environ.get('RSNA_C0_OUTPUT', os.path.join(REPO_ROOT, 'phase_c', 'c0', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
TARGET_PILOT_BYTES = 10 * (1024**3) # 10.0 GiB (10,737,418,240 bytes)

def select_pilot():
    print(f"=== C0.2: PILOT STUDY SELECTION (Target: ~10.0 GB, Seed: {SEED}) ===")
    
    # Load gold studies to exclude
    gold_df = pd.read_csv(os.path.join(DATA_DIR, 'gold_studies.csv'))
    gold_uids = set(gold_df['StudyInstanceUID'].dropna().tolist())
    
    available_studies = sorted(os.listdir(TRAIN_SERIES_DIR))
    
    study_data = []
    for study_uid in available_studies:
        if study_uid in gold_uids:
            continue
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
                
        study_data.append({
            'StudyInstanceUID': study_uid,
            'raw_size_bytes': study_bytes,
            'raw_size_mb': study_bytes / (1024**2),
            'raw_size_gb': study_bytes / (1024**3),
            'series_count': len(series_dirs),
            'slice_count': study_slices
        })
        
    df_all = pd.DataFrame(study_data)
    
    # Deterministic shuffle
    rng = np.random.RandomState(SEED)
    shuffled_indices = rng.permutation(len(df_all))
    df_shuffled = df_all.iloc[shuffled_indices].copy().reset_index(drop=True)
    
    accumulated_bytes = 0
    selected_flags = []
    accum_gb_list = []
    
    for idx, row in df_shuffled.iterrows():
        b = row['raw_size_bytes']
        if accumulated_bytes < TARGET_PILOT_BYTES:
            accumulated_bytes += b
            selected_flags.append(True)
            accum_gb_list.append(accumulated_bytes / (1024**3))
        else:
            selected_flags.append(False)
            accum_gb_list.append(accumulated_bytes / (1024**3))
            
    df_shuffled['selected'] = selected_flags
    df_shuffled['accumulated_gb'] = accum_gb_list
    df_shuffled['selection_seed'] = SEED
    
    # Output complete pilot selection CSV
    out_csv = os.path.join(OUTPUT_DIR, 'pilot_selection.csv')
    df_shuffled.to_csv(out_csv, index=False)
    
    selected_df = df_shuffled[df_shuffled['selected']].copy()
    print(f"\nPilot Selection Complete:")
    print(f"Total Candidate Non-Gold Studies: {len(df_shuffled)}")
    print(f"Selected Pilot Studies: {len(selected_df)}")
    print(f"Total Pilot Raw Size: {selected_df['raw_size_bytes'].sum():,} bytes ({selected_df['raw_size_bytes'].sum() / (1024**3):.4f} GB)")
    print(f"Total Pilot Series Count: {selected_df['series_count'].sum()}")
    print(f"Total Pilot DICOM Slices: {selected_df['slice_count'].sum():,}")
    print(f"Mean Study Size: {selected_df['raw_size_mb'].mean():.2f} MB (Median: {selected_df['raw_size_mb'].median():.2f} MB)")
    print(f"Min Study Size: {selected_df['raw_size_mb'].min():.2f} MB, Max: {selected_df['raw_size_mb'].max():.2f} MB")
    print(f"Saved: {out_csv}")

if __name__ == '__main__':
    select_pilot()
