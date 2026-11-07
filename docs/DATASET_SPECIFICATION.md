# RSNA Knee Abnormality Detection — Dataset Specification & Data Layout

This document specifies the dataset directory structure, series classification conventions, anatomical slot mapping, and preprocessing standards required to train and evaluate models in this repository.

---

## 1. Dataset Origin & Acquisition

The dataset originates from the **RSNA Knee Abnormality Detection Challenge**. It comprises multi-sequence volumetric knee magnetic resonance imaging (MRI) studies paired with multilingual free-text radiology reports and ground-truth clinical annotations.

- **Primary Source**: RSNA / Kaggle Knee Abnormality Detection Challenge.
- **Modality**: Multi-sequence Magnetic Resonance Imaging (MRI).
- **Format**: Raw DICOM (`.dcm`) series organized hierarchically by Study UID and Series UID.
- **De-Identification**: All studies are fully de-identified and anonymized in compliance with HIPAA Safe Harbor standards.

---

## 2. Expected Raw Dataset Directory Layout

External raw datasets should be stored outside the code repository (or linked via `data/`). The repository expects the following directory structure:

```text
/path/to/RSNA_Knee_Data/                   <-- Root configured via $RSNA_DATA_ROOT
├── train.csv                              <-- StudyInstanceUID, Report, pathology targets
├── train_series.csv                       <-- Series metadata (Planes, Fluid, Fat-Suppression)
├── test.csv                               <-- Test cohort StudyInstanceUIDs
├── test_series.csv                        <-- Test cohort series metadata
├── sample_submission.csv                  <-- Baseline submission format
├── train_series/                          <-- Raw training DICOM series
│   ├── <StudyInstanceUID_1>/
│   │   ├── <SeriesInstanceUID_A>/
│   │   │   ├── 1.dcm
│   │   │   ├── 2.dcm
│   │   │   └── ...
│   │   └── <SeriesInstanceUID_B>/
│   │       ├── 1.dcm
│   │       └── ...
│   └── <StudyInstanceUID_2>/
│       └── ...
└── test_series/                           <-- Raw test DICOM series
    └── <StudyInstanceUID_N>/
        └── ...
```

### Metadata Columns

#### `train_series.csv`
- `StudyInstanceUID`: Unique identifier for the patient examination.
- `SeriesInstanceUID`: Unique identifier for the individual MRI sequence.
- `Fluid_Sensitive`: Binary indicator (`1` = fluid-sensitive / T2-weighted / PD-weighted, `0` = anatomical / T1-weighted).
- `Fat_Suppression`: Binary indicator (`1` = fat-suppressed, `0` = non-fat-suppressed).
- `Anatomical_Plane`: Acquisition orientation (`Sagittal`, `Coronal`, or `Axial`).

---

## 3. The 12 Clinical Abnormality Targets

The challenge requires simultaneous multi-label detection across 12 distinct knee pathologies spanning ligaments, menisci, cartilage/osteoarthritis, fluid collections, and osseous trauma:

| Target ID | Abnormality Name | Category | Primary Diagnostic Series / View |
| :---: | :--- | :--- | :--- |
| `0` | **ACL** | Ligamentous Tear | Sagittal Fluid-Sensitive (FS & Non-FS), Coronal |
| `1` | **MCL** | Ligamentous Tear | Coronal Fluid-Sensitive FS |
| `2` | **Medial Meniscus** | Meniscal Tear | Sagittal FS / Non-FS, Coronal FS |
| `3` | **Lateral Meniscus** | Meniscal Tear | Sagittal FS / Non-FS, Coronal FS |
| `4` | **Medial OA** | Joint Space Narrowing / Cartilage Loss | Coronal T1, Sagittal |
| `5` | **Lateral OA** | Joint Space Narrowing / Cartilage Loss | Coronal T1, Sagittal |
| `6` | **PF OA** (Patellofemoral) | Patellofemoral Osteoarthritis | Axial Fluid-Sensitive FS |
| `7` | **Effusion** | Fluid Collection | Axial Fluid-Sensitive FS, Sagittal FS |
| `8` | **Synovitis** | Synovial Inflammation | Axial Fluid-Sensitive FS, Sagittal FS |
| `9` | **Baker's** (Baker's Cyst) | Popliteal Fluid Collection | Axial Fluid-Sensitive FS, Sagittal FS (Posterior) |
| `10` | **Contusion** | Bone Marrow Edema | Coronal Fluid-Sensitive FS |
| `11` | **Fracture** | Osseous Disruption | Coronal T1, Sagittal T1 |

---

## 4. Canonical 6-Slot Anatomical Representation

Because clinical knee MRI protocols vary across institutions, scanner manufacturers, and field strengths, studies contain variable numbers of series. The vision architecture maps heterogeneous series into a standardized **6-slot anatomical tensor representation**:

| Slot Index | Canonical Designation | Anatomical Plane | Contrast / Weighting | Clinical Focus |
| :---: | :--- | :--- | :--- | :--- |
| `0` | `SAG_FLUID_FS` | Sagittal | Fluid-Sensitive, Fat-Suppressed | Cruciate ligaments, meniscal roots |
| `1` | `COR_FLUID_FS` | Coronal | Fluid-Sensitive, Fat-Suppressed | Collaterals (MCL), bone contusions |
| `2` | `AX_FLUID_FS` | Axial | Fluid-Sensitive, Fat-Suppressed | Patellofemoral joint, effusion, cysts |
| `3` | `SAG_FLUID_NOFS` | Sagittal | Fluid-Sensitive, Non-Fat-Suppressed | Meniscal fibrocartilage anatomy (D6) |
| `4` | `COR_T1` | Coronal | T1-Weighted Anatomical | Trabecular bone architecture, fractures |
| `5` | `SAG_T1` | Sagittal | T1-Weighted Anatomical | Cortical bone, chronic degeneration |

If a study lacks a specific series, that slot is zero-padded and its slot mask is set to `0.0`. Cross-attention queries in the Anatomical SlotHead dynamically attend only to active, valid slots.

---

## 5. Preprocessing & Standardized Volumetric Format

Raw DICOM series are preprocessed into standardized compressed NumPy volumes using `phase_c/c1/scripts/preprocess_c1.py`:

- **Standardized Volumetric Shape**: `(32, 256, 256)` — 32 interpolated depth slices with $256 \times 256$ in-plane resolution.
- **Data Type**: `float16` (normalized intensity values in range $[0.0, 1.0]$).
- **Percentile Windowing**: Rescaled using dynamic $P_{1}$ to $P_{99}$ contrast windowing to suppress scanner intensity artifacts.
- **Aspect Ratio Preservation**: In-plane dimensions are scaled with bilateral symmetric zero-padding to preserve square knee anatomy.
- **Output Storage Layout**:
  ```text
  processed/
  └── <StudyInstanceUID>/
      ├── <SeriesInstanceUID_1>.npz   <-- Compressed dictionary: {"volume": ndarray}
      ├── <SeriesInstanceUID_2>.npz
      └── ...
  ```

---

## 6. Gold-58 Quarantined Evaluation Cohort

The project maintains a strictly isolated, locked holdout evaluation cohort of **58 clinical studies** (`csv/gold_studies.csv` and `labels/gold58/`):

> [!CAUTION]
> **GOVERNANCE RULE**: The Gold-58 cohort is strictly quarantined for final benchmark verification. It must **NEVER** be included in training splits, used for cross-validation hyperparameter selection, or used for pseudo-label threshold tuning. Zero data leakage is verified by `validation/acceptance/audit_acceptance.py`.

---

## 7. Configuring Data Paths via Environment Variables

To run the pipeline on any machine without modifying code, set the following environment variables:

```bash
# Root of the external raw RSNA dataset (holding train_series/, train.csv, train_series.csv)
export RSNA_DATA_ROOT="/path/to/RSNA_Knee_Data"

# Root of the code repository
export RSNA_PROJECT_ROOT="/path/to/RSNA_Code"

# Directory where preprocessed .npz arrays will be written or read
export RSNA_PROCESSED_DIR="${RSNA_PROJECT_ROOT}/phase_c/c1/processed"
```
