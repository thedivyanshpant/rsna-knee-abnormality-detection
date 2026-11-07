import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pydicom

from preprocess import load_and_sort_dicom_series, TARGET_SPACING, TARGET_SHAPE

REPO_ROOT = os.environ.get('RSNA_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
DATA_DIR = os.environ.get('RSNA_DATA_ROOT', os.path.join(REPO_ROOT, 'data'))
TRAIN_SERIES_DIR = os.path.join(DATA_DIR, 'train_series')
REPO_ROOT = os.environ.get('RSNA_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
C0_ROOT = os.environ.get('RSNA_C0_ROOT', os.path.join(REPO_ROOT, 'phase_c', 'c0'))
PROCESSED_DIR = os.path.join(C0_ROOT, 'processed')
VIS_DIR = os.path.join(C0_ROOT, 'visual_samples')
os.makedirs(VIS_DIR, exist_ok=True)

# Select 3 representative pilot studies covering Sagittal (ACL/Meniscus), Coronal (MCL/OA/Meniscus), Axial (PF/Cartilage/Effusion)
df_metrics = pd.read_csv(os.path.join(C0_ROOT, 'output', 'preprocessing_metrics.csv'))
train_series_csv = pd.read_csv(os.path.join(DATA_DIR, 'train_series.csv'))

merged = df_metrics.merge(train_series_csv, on=['StudyInstanceUID', 'SeriesInstanceUID'], how='inner')
print(f"Merged metrics with plane metadata: {len(merged)} series.")

# Pick 3 studies that have all 3 planes
studies_with_all_planes = []
for suid, group in merged.groupby('StudyInstanceUID'):
    planes = set(group['Anatomical_Plane'])
    if {'Sagittal', 'Coronal', 'Axial'}.issubset(planes):
        studies_with_all_planes.append(suid)

print(f"Found {len(studies_with_all_planes)} pilot studies containing all 3 planes.")
sample_study_uids = studies_with_all_planes[:3]

def render_comparison_panel(suid, sample_idx):
    study_path = os.path.join(TRAIN_SERIES_DIR, suid)
    study_series_info = merged[merged['StudyInstanceUID'] == suid]
    
    # Select 1 Sagittal, 1 Coronal, 1 Axial series
    selected_series = []
    for plane in ['Sagittal', 'Coronal', 'Axial']:
        matching = study_series_info[study_series_info['Anatomical_Plane'] == plane]
        if not matching.empty:
            selected_series.append(matching.iloc[0])
            
    fig, axes = plt.subplots(len(selected_series), 4, figsize=(20, 5.5 * len(selected_series)))
    if len(selected_series) == 1:
        axes = np.expand_dims(axes, 0)
        
    for row_idx, ser_info in enumerate(selected_series):
        ser_uid = ser_info['SeriesInstanceUID']
        plane = ser_info['Anatomical_Plane']
        ser_path = os.path.join(study_path, ser_uid)
        
        # 1. Raw volume
        vol_raw, meta = load_and_sort_dicom_series(ser_path)
        # 2. Pipeline A
        file_a = os.path.join(PROCESSED_DIR, 'pipeline_a', suid, f"{ser_uid}.npz")
        vol_a = np.load(file_a)['volume']
        # 3. Pipeline B
        file_b = os.path.join(PROCESSED_DIR, 'pipeline_b', suid, f"{ser_uid}.npz")
        vol_b = np.load(file_b)['volume']
        # 4. Pipeline C
        file_c = os.path.join(PROCESSED_DIR, 'pipeline_c', suid, f"{ser_uid}.npz")
        vol_c = np.load(file_c)['volume']
        
        # Select central slice
        slice_raw_idx = vol_raw.shape[0] // 2
        slice_a_idx = vol_a.shape[0] // 2
        slice_b_idx = vol_b.shape[0] // 2
        slice_c_idx = vol_c.shape[0] // 2
        
        img_raw = vol_raw[slice_raw_idx]
        img_a = vol_a[slice_a_idx]
        img_b = vol_b[slice_b_idx]
        img_c = vol_c[slice_c_idx]
        
        # Column 0: RAW
        axes[row_idx, 0].imshow(img_raw, cmap='gray')
        axes[row_idx, 0].set_title(f"RAW: {plane}\nShape: {vol_raw.shape} | Slices: {len(vol_raw)}\nNative Range: [{img_raw.min():.0f}, {img_raw.max():.0f}]", fontsize=11, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Column 1: Pipeline A
        axes[row_idx, 1].imshow(img_a, cmap='gray')
        axes[row_idx, 1].set_title(f"Pipeline A (Minimal)\nShape: {vol_a.shape} | float32\nStorage: {os.path.getsize(file_a)/(1024**2):.2f} MB", fontsize=11, fontweight='bold', color='navy')
        axes[row_idx, 1].axis('off')
        
        # Column 2: Pipeline B
        axes[row_idx, 2].imshow(img_b, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 2].set_title(f"Pipeline B (Standardized)\nShape: {vol_b.shape} | Spacing: {TARGET_SPACING}\nNorm: [0, 1] | Storage: {os.path.getsize(file_b)/(1024**2):.2f} MB", fontsize=11, fontweight='bold', color='darkgreen')
        axes[row_idx, 2].axis('off')
        
        # Column 3: Pipeline C
        axes[row_idx, 3].imshow(img_c, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 3].set_title(f"Pipeline C (Training-Ready)\nFixed Shape: {vol_c.shape} | float16\nStorage: {os.path.getsize(file_c)/(1024**2):.2f} MB", fontsize=11, fontweight='bold', color='darkred')
        axes[row_idx, 3].axis('off')
        
    plt.suptitle(f"Phase C0 Visual Information Preservation Audit — Study UID: ...{suid[-16:]}\nComparison across Anatomical Planes (Sagittal / Coronal / Axial)", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_png = os.path.join(VIS_DIR, f"visual_preservation_study_{sample_idx+1}_{suid[-10:]}.png")
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved visual sample: {out_png}")

print("Rendering visual comparison panels for 3 representative pilot studies...")
for i, suid in enumerate(sample_study_uids):
    render_comparison_panel(suid, i)

print("Visual preservation check complete!")
