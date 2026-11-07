import os
import glob
import json
import numpy as np
import pandas as pd

REPO_ROOT = os.environ.get('RSNA_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
C0_ROOT = os.environ.get('RSNA_C0_ROOT', os.path.join(REPO_ROOT, 'phase_c', 'c0'))
OUTPUT_DIR = os.path.join(C0_ROOT, 'output')
PROCESSED_DIR = os.path.join(C0_ROOT, 'processed')

def analyze():
    print("=== C0.4 - C0.6: STATISTICAL ANALYSIS & EXTRAPOLATION ===")
    
    # 1. Load inventory & pilot metrics
    with open(os.path.join(OUTPUT_DIR, 'dataset_inventory.json')) as f:
        inv = json.load(f)
        
    df_metrics = pd.read_csv(os.path.join(OUTPUT_DIR, 'preprocessing_metrics.csv'))
    pilot_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'pilot_selection.csv'))
    selected_uids = set(pilot_df[pilot_df['selected']]['StudyInstanceUID'])
    
    print(f"Total metrics records: {len(df_metrics)} across {df_metrics['StudyInstanceUID'].nunique()} studies.")
    
    # Group by StudyInstanceUID
    study_grouped = df_metrics.groupby('StudyInstanceUID').agg({
        'raw_bytes': 'sum',
        'stored_bytes_a': 'sum',
        'stored_bytes_b': 'sum',
        'stored_bytes_c': 'sum',
        'ram_bytes_a': 'sum',
        'ram_bytes_b': 'sum',
        'ram_bytes_c': 'sum',
        'time_sec_a': 'sum',
        'time_sec_b': 'sum',
        'time_sec_c': 'sum',
        'SeriesInstanceUID': 'count'
    }).reset_index()
    
    raw_pilot_bytes = study_grouped['raw_bytes'].sum()
    raw_pilot_gb = raw_pilot_bytes / (1024**3)
    
    full_dataset_raw_bytes = inv['total_raw_storage_bytes']
    full_dataset_raw_gb = inv['total_raw_storage_gb']
    full_studies_count = inv['total_study_directories']
    
    # Per-pipeline summary
    pipelines = ['Pipeline A (Minimal)', 'Pipeline B (Standardized)', 'Pipeline C (Training-Ready)']
    keys = ['a', 'b', 'c']
    
    comp_rows = []
    extrapolations = {}
    
    for name, k in zip(pipelines, keys):
        stored_bytes_total = study_grouped[f'stored_bytes_{k}'].sum()
        stored_gb_total = stored_bytes_total / (1024**3)
        ram_bytes_total = study_grouped[f'ram_bytes_{k}'].sum()
        ram_gb_total = ram_bytes_total / (1024**3)
        
        reduction_ratio = raw_pilot_bytes / stored_bytes_total if stored_bytes_total > 0 else 0
        space_saving_pct = (1.0 - (stored_bytes_total / raw_pilot_bytes)) * 100.0
        
        # Per-study stored MB
        study_stored_mb = study_grouped[f'stored_bytes_{k}'] / (1024**2)
        study_ram_mb = study_grouped[f'ram_bytes_{k}'] / (1024**2)
        study_time = study_grouped[f'time_sec_{k}']
        
        # Extrapolate full dataset
        extrapolated_stored_gb = full_dataset_raw_gb * (stored_bytes_total / raw_pilot_bytes)
        extrapolated_ram_gb = (full_dataset_raw_bytes / raw_pilot_bytes) * ram_bytes_total / (1024**3)
        
        # Shape stats
        sample_shapes = df_metrics[f'shape_{k}'].value_counts().head(3).to_dict()
        sample_dtypes = df_metrics[f'dtype_{k}'].value_counts().to_dict()
        
        comp_rows.append({
            'Pipeline': name,
            'Pilot_Stored_GB': round(stored_gb_total, 4),
            'Reduction_Ratio': round(reduction_ratio, 2),
            'Space_Saving_Pct': round(space_saving_pct, 2),
            'Mean_Study_Stored_MB': round(study_stored_mb.mean(), 2),
            'Median_Study_Stored_MB': round(study_stored_mb.median(), 2),
            'Min_Study_Stored_MB': round(study_stored_mb.min(), 2),
            'Max_Study_Stored_MB': round(study_stored_mb.max(), 2),
            'Mean_Decoded_RAM_MB_per_Study': round(study_ram_mb.mean(), 2),
            'Median_Decoded_RAM_MB_per_Study': round(study_ram_mb.median(), 2),
            'Total_Decoded_RAM_Pilot_GB': round(ram_gb_total, 2),
            'Mean_Proc_Time_sec_per_Study': round(study_time.mean(), 2),
            'Median_Proc_Time_sec_per_Study': round(study_time.median(), 2),
            'Extrapolated_Full_Dataset_GB': round(extrapolated_stored_gb, 2),
            'Dominant_Dtype': list(sample_dtypes.keys())[0],
            'Fixed_or_Dynamic_Dimensions': 'Fixed (32, 256, 256)' if k == 'c' else 'Dynamic (Subject-Specific)'
        })
        
        extrapolations[name] = {
            'pilot_raw_gb': round(raw_pilot_gb, 4),
            'pilot_processed_stored_gb': round(stored_gb_total, 4),
            'reduction_ratio': round(reduction_ratio, 4),
            'space_saving_pct': round(space_saving_pct, 2),
            'mean_study_stored_mb': round(study_stored_mb.mean(), 2),
            'median_study_stored_mb': round(study_stored_mb.median(), 2),
            'mean_decoded_ram_mb_per_study': round(study_ram_mb.mean(), 2),
            'extrapolated_full_dataset_stored_gb': round(extrapolated_stored_gb, 2),
            'extrapolated_full_dataset_ram_gb': round(extrapolated_ram_gb, 2),
            'dominant_dtype': list(sample_dtypes.keys())[0],
            'dominant_shapes': sample_shapes
        }
        
    df_comp = pd.DataFrame(comp_rows)
    comp_csv = os.path.join(OUTPUT_DIR, 'pipeline_comparison.csv')
    df_comp.to_csv(comp_csv, index=False)
    
    extrap_json = os.path.join(OUTPUT_DIR, 'extrapolated_storage.json')
    with open(extrap_json, 'w') as f:
        json.dump(extrapolations, f, indent=2)
        
    print(f"\nSaved Comparison Table: {comp_csv}")
    print(f"Saved Extrapolations: {extrap_json}")
    print("\n--- COMPARISON RESULTS ---")
    print(df_comp[['Pipeline', 'Pilot_Stored_GB', 'Reduction_Ratio', 'Mean_Study_Stored_MB', 'Mean_Decoded_RAM_MB_per_Study', 'Extrapolated_Full_Dataset_GB']].to_string())

if __name__ == '__main__':
    analyze()
