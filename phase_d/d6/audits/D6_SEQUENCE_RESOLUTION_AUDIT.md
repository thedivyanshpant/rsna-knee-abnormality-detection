# Phase D6: Sequence Resolution & Multi-Sequence Representation Audit

- **Audit Timestamp**: 2026-09-14T14:47:26.906459+00:00
- **Development Cohort**: 4349 non-Gold studies (100% Gold-isolated)
- **Total Processed Series**: 24033 3D NPZ volumes

## 1. Sequence Slot Population Comparison: D1/D3 Baseline vs Phase D6

| Slot Index | Canonical Slot Name | Plane | Fluid | FS | D1/D3 Baseline Coverage | D6 Coverage | Delta Studies | Delta (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Slot 0 | `SAG_FLUID_FS` | SAG | 1 | 1 | 4094 (94.1%) | **4094 (94.1%)** | **+0** | **+0.0%** |
| Slot 1 | `COR_FLUID_FS` | COR | 1 | 1 | 4191 (96.4%) | **4191 (96.4%)** | **+0** | **+0.0%** |
| Slot 2 | `AX_FLUID_FS` | AX | 1 | 1 | 4348 (100.0%) | **4348 (100.0%)** | **+0** | **+0.0%** |
| Slot 3 | `SAG_FLUID_NOFS` | SAG | 1 | 0 | 0 (0.0%) | **852 (19.6%)** | **+852** | **+19.59%** |
| Slot 4 | `COR_T1` | COR | 0 | 0 | 3361 (77.3%) | **3361 (77.3%)** | **+0** | **+0.0%** |
| Slot 5 | `SAG_T1` | SAG | 0 | 0 | 4209 (96.8%) | **4209 (96.8%)** | **+0** | **+0.0%** |

## 2. Cohort Completeness Histogram (Populated Slots per Study)

| Populated Slots | D1/D3 Baseline Studies | D1/D3 Pct (%) | Phase D6 Studies | Phase D6 Pct (%) | Movement |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0 of 6 slots** | 0 | 0.00% | **0** | **0.00%** | +0 |
| **1 of 6 slots** | 0 | 0.00% | **0** | **0.00%** | +0 |
| **2 of 6 slots** | 0 | 0.00% | **0** | **0.00%** | +0 |
| **3 of 6 slots** | 231 | 5.31% | **2** | **0.05%** | -229 |
| **4 of 6 slots** | 1080 | 24.83% | **1260** | **28.97%** | +180 |
| **5 of 6 slots** | 3038 | 69.86% | **2513** | **57.78%** | -525 |
| **6 of 6 slots** | 0 | 0.00% | **574** | **13.20%** | +574 |

## 3. Slot 3 (SAG_FLUID_NOFS) Activation Analysis

- In Phase D1/D3/D5, `SAG_FLUID_NOFS` was **0.00% populated** (0 studies) because non-FS series in `train_series.csv` were categorized as `(0, 0)`.
- In Phase D6, **852 studies (19.59%)** that acquired $\ge 2$ Sagittal non-FS acquisitions are now resolved using tissue intensity bifurcation.
- The higher marrow/subcutaneous fat intensity sequence is routed to `SAG_T1` (Slot 5), while the lower-intensity fluid/cartilage-contrast sequence is routed to `SAG_FLUID_NOFS` (Slot 3).
- As a direct consequence, **574 studies now achieve 6/6 complete multi-sequence coverage** (up from 0 in D1/D3).

## 4. Collision Resolution Policy & Metrics

- **Total Collision Entries**: 5227
- **Affected Studies**: 1737 (39.94%)
- **Methodology**: Replaced blind alphabetical SeriesInstanceUID sorting with physical volume tissue quality score ($f_{\text{tissue}} \times (P_{90} - P_{10})$).
