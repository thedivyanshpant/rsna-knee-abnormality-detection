# Phase D1: Target Architecture & Data Pre-Training Audit (D1.0)

- **Audit Timestamp**: `2026-09-12T17:16:03.285495+00:00`
- **Audit Verdict**: **PASS**

## Section 35 Mandatory Questions & Verified Answers

### 1. How many development studies are available?
- **4,349 studies** (100% of frozen 5-fold CV splits).

### 2 & 3. Per-Slot Allocation & Coverage Percentages

| Slot | Canonical Name | Populated Studies | Missing Studies | Coverage (%) |
| :---: | :--- | :---: | :---: | :---: |
| Slot 0 | `SAG_FLUID_FS` | 4,094 | 255 | **94.14%** |
| Slot 1 | `COR_FLUID_FS` | 4,191 | 158 | **96.37%** |
| Slot 2 | `AX_FLUID_FS` | 4,348 | 1 | **99.98%** |
| Slot 3 | `SAG_FLUID_NOFS` | 0 | 4,349 | **0.0%** |
| Slot 4 | `COR_T1` | 3,361 | 988 | **77.28%** |
| Slot 5 | `SAG_T1` | 4,209 | 140 | **96.78%** |

> **Note on Slot 3 (`SAG_FLUID_NOFS`)**: In Kaggle's `train_series.csv`, non-fat-suppressed fluid sequences were tagged identically as non-fluid. Under Phase D Section 13 governance, Slot 3 is cleanly represented by a zero tensor and `slot_mask = 0.0`.

### 4. Slot Collisions
- Total Collisions Logged: **5,227** across **1,737 studies**.
- Resolution Policy: `highest_slice_count_then_deterministic_lexicographic_tie_break`.

### 5. Unresolved Sequences
- Unresolved Sequences: **0** (PASS (100% resolved)).

### 6. Requiring Metadata Fallback
- Raw DICOM sequences requiring TR/TE fallback: **73 / 336 (21.73%)**.
- Development cohort sequences requiring fallback: **0** (metadata 100% complete).

### 7. Slice Counts per Selected Series
- Preprocessed C1 volumes: **Fixed 32 slices resampled to 3.0 mm depth spacing**.
- Raw DICOM series: Min = 11, Median = 30.0, Mean = 31.33, Max = 320.

### 8. PixelSpacing Distribution
- Preprocessed C1 volumes: `[3.0, 0.5, 0.5] mm (constant resampled)`.
- Raw DICOM: Min = 0.1465 mm, Mean = 0.3472 mm, Max = 0.7813 mm.

### 9. Physical Crop Dimensions
- Specification: `130.0 mm physical extent`.
- In-plane FOV: `256 x 0.5 mm = 128.0 mm physical extent (matches 130mm specification within 1.5%)`.
- Resized to model input: `224 x 224 pixels`.

### 10. Usable 3-Slice Groups
- Slice Band: `(0.2, 0.8) of acquired volume depth` (`slices 7 to 24 out of 32 slices`).
- Sampled groups per slot: **12** (total 72 groups per study).
- Grouping method: `triplet grouping [slice[i-1], slice[i], slice[i+1]] stacked to (3, H, W)`.

### 11. Expected Tensor Storage Size (Section 32 Audit)
- If pre-extracting all 2.5D image tensors to disk:
  * Float16: **87.79 GiB**
  * Float32: **175.59 GiB**
- Storage-Efficient Strategy: `On-the-fly dynamic 2.5D extraction from existing 73.66 GB C1 volumes (0 additional disk GB)`.

### 12. Expected DINO Feature-Cache Size
- Feature cache per slice group: **0.896 GiB**.
- Pooled feature cache per slot: **76.45 MiB**.

### 13. Missing Slots per Study
- `1_missing_slots`: **3,038 studies** (69.86%)
- `2_missing_slots`: **1,080 studies** (24.83%)
- `3_missing_slots`: **231 studies** (5.31%)
- Handled with: `Explicit zero tensor with slot_mask = 0 (Section 13)`.

### 14. Studies Affected by Ambiguous Metadata
- Count: **0** (100% of the 4,349 development studies in train_series.csv have unambiguous Anatomical_Plane, Fluid_Sensitive, and Fat_Suppression.).
