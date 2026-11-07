# Phase C3: Canonical Series Slot Mapping & Audit Report

- **Audit Date**: 2026-09-12T08:31:08.355376+00:00
- **Total Non-Gold Cohort Studies**: 4349
- **Total Processed Volumes on Disk**: 24033
- **Total Matched Series**: 24033
- **Unmatched Processed Series**: 0 (FAIL criteria = > 0)
- **Metadata Rows with No Processed NPZ**: 2 (Both verified as C1 corrupt raw series)
- **Status**: **PASS (Zero Join Anomalies)**

## 1. Studies by Populated Slot Count

| Populated Slots | Number of Studies | Percentage |
| :---: | :---: | :---: |
| **0 / 5 slots** | 0 | 0.00% |
| **1 / 5 slots** | 0 | 0.00% |
| **2 / 5 slots** | 0 | 0.00% |
| **3 / 5 slots** | 231 | 5.31% |
| **4 / 5 slots** | 1,080 | 24.83% |
| **5 / 5 slots** | 3,038 | 69.86% |

## 2. Per-Slot Allocation Counts

| Slot Index | Canonical Slot Name | Populated Studies | Missing Studies | Coverage (%) |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `sagittal_fluid_sensitive` | 4,094 | 255 | 94.14% |
| 1 | `coronal_fluid_sensitive` | 4,191 | 158 | 96.37% |
| 2 | `axial_fluid_sensitive` | 4,348 | 1 | 99.98% |
| 3 | `sagittal_non_fs_t1` | 4,209 | 140 | 96.78% |
| 4 | `coronal_non_fs_t1` | 3,361 | 988 | 77.28% |

## 3. Collision Resolution Audit

- **Total Collisions Detected**: 2669
- **Studies with Collisions**: 1737
- **Collision Policy**: Deterministic selection of lexicographically lowest `SeriesInstanceUID` (`min(series_uids)`). Both chosen and discarded series are permanently logged to manifest.

## 4. Known C1 Partial Studies Audit

### Study `1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035`
- Available Processed Series: 4
- Populated Canonical Slots: 4
- Slot Mask: `[1, 1, 0, 1, 1]`
- Unprocessed / Corrupted Raw Series: `1.2.826.0.1.3680043.8.498.39396636671446532796538574315802928348`
- Safely Handled (No fabricated volumes): **True**

### Study `1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680`
- Available Processed Series: 4
- Populated Canonical Slots: 4
- Slot Mask: `[1, 0, 1, 1, 1]`
- Unprocessed / Corrupted Raw Series: `1.2.826.0.1.3680043.8.498.31160785238781353848727311763596115703`
- Safely Handled (No fabricated volumes): **True**

