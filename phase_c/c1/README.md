# Phase C1 — Full-Dataset Pipeline C Preprocessing & Sharding

This directory contains the production-grade preprocessed Pipeline C dataset, execution engine, verification tools, manifests, and execution summaries for all 4,349 non-Gold training studies.

---

## 1. Directory Structure

```
phase_c/c1/
├── README.md                      # This documentation
├── scripts/                       # Execution and audit scripts
│   ├── preprocess_c1.py           # Multi-threaded production preprocessing engine
│   └── validate_c1.py             # Scientific integrity & artifact verification audit
├── manifests/                     # Provenance & tracking
│   └── c1_study_manifest.csv      # Complete study-level execution manifest (4,349 rows)
├── output/                        # Execution metadata & analytics
│   └── c1_execution_summary.json  # Aggregated metrics, timing, throughput, and storage
└── processed/                     # Preprocessed Pipeline C dataset (73.66 GB)
    └── <StudyInstanceUID>/        # 4,349 study directories
        └── <SeriesInstanceUID>.npz # 24,033 preprocessed 3D volume files
```

---

## 2. Technical Specification of Preprocessed Volumes

All `.npz` files follow the standardized Pipeline C specification:
* **Tensor Dimensions**: Exactly `(32, 256, 256)` $(D, H, W)$
* **Voxel Spacing**: Resampled to `(3.0, 0.5, 0.5)` mm
* **Intensity Range**: Robust $[P_{0.5}, P_{99.5}]$ percentile clipped and normalized to $[0.0, 1.0]$
* **Data Precision**: `float16` (2 bytes/voxel, exactly 4.19 MB decoded RAM per series)
* **Archive Content**:
  - `volume`: 3D numpy array of shape `(32, 256, 256)` with dtype `float16`
  - `spacing`: 1D numpy array of shape `(3,)` with values `[3.0, 0.5, 0.5]`

---

## 3. Dataset Summary Statistics

| Dataset Dimension | Measured Metric |
| :--- | :--- |
| **Total Non-Gold Candidate Studies** | **4,349 studies** (100% evaluated) |
| **Fully Successful Studies** | **4,347 studies** (99.95%) |
| **Partial Studies** | **2 studies** (4/5 series processed, 1 raw corrupt DICOM slice) |
| **Failed Studies** | **0 studies** |
| **Total Processed Series** | **24,033 `.npz` files** |
| **Total Preprocessed Storage Size** | **73.66 GB (79,096,374,846 bytes)** |
| **Average Stored Size per Study** | **17.34 MB** |
| **Original Raw Footprint** | **523.54 GB** |
| **Net Storage Space Reduction** | **85.93% (7.11x reduction)** |
| **Gold-58 Separation** | **0 overlap** (58 gold studies strictly excluded) |
| **Total Preprocessing Wall Time** | **129.34 min (2.15 hours)** on 36 CPU worker threads |
| **Effective Preprocessing Throughput** | **0.56 studies/sec (33.6 studies/min)** |

---

## 4. Verification & Integrity Assertions

Running `python3 scripts/validate_c1.py` confirms:
1. **Gold-58 Exclusion**: 0 Gold-58 studies are present in `processed/`.
2. **Cohort Coverage**: Exactly 4,349 study directories exist with 24,033 valid `.npz` series.
3. **Random Sample Audit (100 Series)**:
   - Shape compliance `(32, 256, 256)`: **100/100 (100%)**
   - Dtype compliance `float16`: **100/100 (100%)**
   - Spacing compliance `(3.0, 0.5, 0.5)` mm: **100/100 (100%)**
   - Intensity range valid $[0.0, 1.0]$: **100/100 (100%)**
   - NaN / Inf free: **100/100 (100%)**
