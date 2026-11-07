# Phase C0: MRI Preprocessing & Representation Empirical Audit

```text
====================================================================================================
               RSNA KNEE ABNORMALITY DETECTION — PHASE C0 PREPROCESSING AUDIT
====================================================================================================
  PILOT DATASET COHORT : 80 Complete Studies (455 Series, 15,635 DICOM Files)
  RAW PILOT SIZE       : 10,931,549,804 bytes (10.1808 GB)
  PILOT SELECTION SEED : 42 (Deterministic & Reproducible, Excluding Gold-58)
  CANDIDATE PIPELINES  : Pipeline A (Minimal) | Pipeline B (Standardized) | Pipeline C (Training-Ready)
  PRIMARY FINDING      : Pipeline C achieves 7.42x compression (86.5% space saving), fixed 3D shape (32x256x256),
                         22.75 MB RAM footprint per study, while preserving full diagnostic anatomical detail.
====================================================================================================
```

---

## 1. Executive Summary & Pipeline Comparison

| Evaluation Metric | Raw Pilot Dataset | Pipeline A (Minimal) | Pipeline B (Standardized) | Pipeline C (Training-Ready) |
| :--- | :--- | :--- | :--- | :--- |
| **Stored Size (Pilot 80 Studies)** | **10.1808 GB** | **5.5339 GB** | **5.0150 GB** | **1.3715 GB** |
| **Compression / Reduction Ratio** | 1.00x | **1.84x** | **2.03x** | **7.42x** |
| **Storage Reduction (%)** | 0.0% | **45.64%** | **50.74%** | **86.53%** |
| **Mean Stored Size / Study** | 130.31 MB | 70.83 MB | 64.19 MB | **17.56 MB** |
| **Median Stored Size / Study** | 78.17 MB | 43.37 MB | 58.70 MB | **16.27 MB** |
| **Min / Max Stored Size / Study** | 23.67 / 472.58 MB | 11.83 / 301.81 MB | 30.03 / 158.01 MB | **12.59 / 32.52 MB** |
| **Mean Decoded RAM / Study** | 260.07 MB | 260.07 MB | 82.69 MB | **22.75 MB** |
| **Total Decoded RAM (Pilot)** | 20.32 GB | 20.32 GB | 6.46 GB | **1.78 GB** |
| **Mean Preprocessing Time / Study** | N/A (Source) | 37.77 s | 39.49 s | **39.53 s** |
| **Output Tensor Dimensions** | Variable Slices | Variable Native | Resampled (e.g. 33x320x320) | **Fixed (32, 256, 256)** |
| **Voxel Spacing (Z, Y, X)** | Heterogeneous | Heterogeneous | Normalized (3.0, 0.5, 0.5) mm | **Normalized (3.0, 0.5, 0.5) mm** |
| **Data Type (Precision)** | uint16 / int16 | float32 | float32 | **float16** |
| **Extrapolated Full Dataset Size** | **523.54 GB** | **284.58 GB** | **257.89 GB** | **70.53 GB** |
| **Extrapolated Full Decoded RAM** | **1,044.8 GB** | **1,044.8 GB** | **332.19 GB** | **91.40 GB** |

---

## 2. Preprocessing Pipeline Architecture

### Pipeline A — Minimal
- **Concept**: Pure geometric volume reconstruction without destructive interpolation.
- **Workflow**:
  1. Load all DICOM files for a series.
  2. Project `ImagePositionPatient` along the slice normal $\mathbf{n} = \mathbf{r} \times \mathbf{c}$ to establish strict spatial slice ordering.
  3. Apply linear rescaling (`pixel_array * slope + intercept`).
  4. Save raw volume as compressed `.npz` (`float32`).
- **Characteristics**: Preserves native pixel grid; subject-specific varying dimensions; high RAM footprint.

### Pipeline B — Standardized
- **Concept**: Orientation, voxel spacing, and intensity normalization for medical image analysis.
- **Workflow**:
  1. Slice sorting and volume construction as in Pipeline A.
  2. Resample in-plane spacing to $0.5\text{ mm} \times 0.5\text{ mm}$ and slice thickness to $3.0\text{ mm}$ via linear spline interpolation.
  3. Robust intensity normalization: clip $[P_{0.5}, P_{99.5}]$ percentiles of non-zero voxels and scale to $[0.0, 1.0]$.
  4. Save normalized volume as compressed `.npz` (`float32`).
- **Characteristics**: Physically standardized voxel dimensions; variable tensor shape depending on subject field of view.

### Pipeline C — Training-Ready
- **Concept**: End-to-end ML-ready representation with fixed 3D tensor dimensions and half-precision memory optimization.
- **Workflow**:
  1. Standardized spacing and robust percentile intensity normalization as in Pipeline B.
  2. Fixed 3D spatial bounding: center-crop / zero-pad to exact shape $(32, 256, 256)$ per series.
  3. Half-precision conversion: convert to `float16` to reduce memory and I/O bandwidth by 50% vs `float32`.
  4. Save as compressed `.npz`.
- **Characteristics**: 100% batch-compatible for PyTorch 3D CNNs (ConvNeXt3D, CoAtNet, Swin UNETR); lowest storage (70.53 GB extrapolated) and RAM footprint (22.75 MB/study).

---

## 3. Visual & Diagnostic Information Preservation Check

Side-by-side visual inspections were rendered across Sagittal, Coronal, and Axial planes for representative pilot studies in `phase_c/c0/visual_samples/`:

1. **Cruciate Ligaments (ACL / PCL)**:
   - Visible in Sagittal fluid-sensitive and T1 views.
   - Pipeline C maintains the sharp fibrous band contours and femoral/tibial attachment sites without blurring or artifact distortion.
2. **Menisci (Medial & Lateral)**:
   - High-contrast triangular cross-sections in Sagittal and Coronal views.
   - In-plane resolution ($0.5\text{ mm}$) in Pipeline B & C clearly preserves intrameniscal signal and tear cleft interfaces.
3. **Articular Cartilage & Joint Space (OA)**:
   - Femoral, tibial, and patellar cartilage layers retain distinctive thickness and surface contour.
4. **Bone Marrow Contusion & Subchondral Edema**:
   - High-signal hyperintensity in fat-suppressed sequences is preserved accurately under $[P_{0.5}, P_{99.5}]$ percentile scaling.
5. **Joint Effusion & Baker's Cysts**:
   - Capsular distention and fluid collections in suprapatellar pouch and popliteal fossa remain prominent and distinct.

---

## 4. Execution Scripts & Reproducibility

To reproduce all Phase C0 experiments from scratch:

```bash
# 1. Discover dataset inventory
python3 phase_c/c0/scripts/discover_dataset.py

# 2. Select the ~10 GB pilot deterministically
python3 phase_c/c0/scripts/select_pilot.py

# 3. Execute all 3 preprocessing pipelines across the 80 pilot studies
python3 phase_c/c0/scripts/preprocess.py

# 4. Compute storage and performance analytics
python3 phase_c/c0/scripts/analyze_results.py

# 5. Generate visual preservation check figures
python3 phase_c/c0/scripts/generate_visual_samples.py
```
