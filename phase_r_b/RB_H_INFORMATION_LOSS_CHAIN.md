# PHASE R/B — FORENSIC INFORMATION-LOSS CHAIN AUDIT (R/B-H)

**Date**: September 16, 2026  
**Auditor**: Autonomous Agentic AI (Google DeepMind Antigravity)  
**Objective**: Construct an unbroken, empirically grounded forensic audit of the end-to-end data pipeline—from raw clinical DICOM acquisition to the final pathology classifier—classifying the exact information loss severity at every stage using measured HPC evidence.

---

## 1. Executive Summary & Chain Diagram

```text
RAW CLINICAL DICOM (523 GB, native resolution, variable slices)
       │
       ▼ [1. Series Selection] ──► Severity: MEDIUM (Discarding extra series in multi-series exams)
       │
       ▼ [2. Sequence Resolution] ──► Severity: HIGH (Protocol asymmetry: Slot 3 missing in 80.41%)
       │
       ▼ [3. Collision Resolution] ──► Severity: MEDIUM (39.94% collision rate, single series chosen)
       │
       ▼ [4. 32-Slice Resampling] ──► Severity: MEDIUM (Z-axis interpolation & partial volume blur)
       │
       ▼ [5. Central-Band Selection] ──► Severity: HIGH (Truncation of peripheral anatomy: Slices 0-5, 26-31)
       │
       ▼ [6. 3-Slice Group Repr.] ──► Severity: MEDIUM (Sparse 8-group sampling across volume depth)
       │
       ▼ [7. 256×256 In-Plane Resize] ──► Severity: HIGH (55.4% areal pixel density destroyed)
       │
       ▼ [8. ViT Patch Tokenization] ──► Severity: HIGH (7.0 mm patch token smears 1.5 mm ligaments)
       │
       ▼ [9. Slot Aggregation] ──► Severity: LOW-MEDIUM (Zero-padding missing slots dilutes attention)
       │
       ▼ [10. Pathology Classifier] ──► Severity: LOW (Standard calibrated multi-label BCE head)
       │
FINAL TARGET PREDICTIONS (Trapped at ~0.81–0.83 Macro AUC)
```

---

## 2. Stage-by-Stage Forensic Chain Audit

### Stage 1: Series Selection
* **Pipeline Action**: Scan all series per study in `train_series.csv` and filter by anatomical plane (Sagittal, Coronal, Axial) and magnetic resonance sequence type.
* **Measured Evidence**:
  - Development cohort contains 24,033 series across 4,349 studies (Mean: 5.53 series/study; Range: 2 to 14 series/study).
  - Studies with $>6$ series have secondary or reformatted series discarded.
* **Information Loss Severity**: **MEDIUM**
* **Technical Rationale**: For high-end protocols with dedicated oblique sagittal sequences or thin-slice 3D reconstructions, discarding non-canonical series loses specialized views of the ACL and meniscus root.

---

### Stage 2: Sequence Resolution & Protocol Asymmetry
* **Pipeline Action**: Map available series into the canonical 6 slots (`SAG_FLUID_FS`, `COR_FLUID_FS`, `AX_FLUID_FS`, `SAG_FLUID_NOFS`, `COR_T1`, `SAG_T1`).
* **Measured Evidence**:
  - `Slot 0 (SAG_FLUID_FS)`: 98.41% present.
  - `Slot 1 (COR_FLUID_FS)`: 98.23% present.
  - `Slot 2 (AX_FLUID_FS)`: 96.48% present.
  - `Slot 3 (SAG_FLUID_NOFS)`: **80.41% MISSING** (Present in only 852 of 4,349 studies).
  - `Slot 4 (COR_T1)`: **22.72% MISSING** (Present in 3,361 of 4,349 studies).
  - `Slot 5 (SAG_T1)`: 2.16% missing (Present in 4,255 of 4,349 studies).
  - **Complete 6/6 Slot Protocol**: Present in only **13.20% of studies** (574 studies).
* **Information Loss Severity**: **HIGH**
* **Technical Rationale**: Severe clinical protocol heterogeneity. Models are forced to evaluate 80.4% of studies with zero-padded Slot 3 representations, impairing contrast comparisons between fat-suppressed and non-fat-suppressed fluid signals (vital for meniscal tear grading and marrow contusion vs artifact).

---

### Stage 3: Collision Resolution
* **Pipeline Action**: When multiple series map to the same slot, select a single winner using DICOM metadata rules and tissue attenuation profiles.
* **Measured Evidence**:
  - **1,737 of 4,349 studies (39.94%)** experienced at least one slot collision.
  - 1,691 studies had collisions in Slot 0, 1, or 2.
  - The loser series is completely eliminated from the training tensor.
* **Information Loss Severity**: **MEDIUM**
* **Technical Rationale**: Often both colliding series contain useful complementary information (e.g. proton-density with fat saturation vs T2-weighted fluid sensitive, or differing slice angles). Forcing a winner-take-all choice discards valid clinical data.

---

### Stage 4: 32-Slice Resampling
* **Pipeline Action**: Uniformly resample variable acquisition slice counts (16 to 64 slices) to a fixed 32-slice 3D volume using linear z-interpolation.
* **Measured Evidence**:
  - Processed volumes frozen at $[32, 256, 256]$ at 3.0 mm nominal z-spacing (`phase_c/c1/processed/`).
  - Studies acquired at thin slices (e.g. 1.5–2.0 mm 3D FSE) undergo z-axis downsampling, smearing axial/coronal planes.
* **Information Loss Severity**: **MEDIUM**
* **Technical Rationale**: Linear interpolation across slice boundaries causes partial volume averaging, softening high-frequency edges of fine ligament fibers and articular cartilage.

---

### Stage 5: Central-Band Selection
* **Pipeline Action**: In Phase D6/D8, model input was restricted to central slice indices (e.g. slices 6–25 of 32) or central slice clusters.
* **Measured Evidence**:
  - Empirical slice depth analysis across 827 volumes demonstrated that peripheral slices contain substantial foreground tissue:
    - Slices 0–5 contain **58.52% foreground tissue**.
    - Slices 26–31 contain **76.87% foreground tissue** (Ratio to central band: 0.831).
  - The MCL is physically located in coronal slices 0–6.
  - Popliteal cysts (Baker's) reside in sagittal slices 22–30.
* **Information Loss Severity**: **HIGH**
* **Technical Rationale**: Explicitly responsible for the **MCL performance collapse (0.6216 AUC, PR-AUC: 0.0553)**. Peripheral slice clipping physically amputates collateral ligaments and outer meniscocapsular junctions from the network's view.

---

### Stage 6: 3-Slice Group Representation
* **Pipeline Action**: Group adjacent 3 slices into pseudo-RGB channels, evaluating 8 distinct groups along the depth dimension.
* **Measured Evidence**:
  - Slices are sampled sparsely across the 32-slice volume (stride of 3 or 4).
  - Intervening slices are bypassed.
* **Information Loss Severity**: **MEDIUM**
* **Technical Rationale**: While pseudo-RGB grouping allows transfer learning from 2D vision backbones (DINOv2), it provides zero inter-slice cross-attention within the group and skips intermediate slices where hairline fractures or small meniscal flap tears reside.

---

### Stage 7: 256×256 In-Plane Spatial Resizing
* **Pipeline Action**: Bilinear downsampling from native DICOM acquisition matrices ($320\times 320$, $384\times 384$, $512\times 512$) down to fixed $256\times 256$.
* **Measured Evidence**:
  - Native DICOM PixelSpacing mean: **0.334 mm** (96.7% acquired at $\ge 320\times 320$; 57.3% at $\ge 432\times 432$).
  - Resampled PixelSpacing: **0.500 mm**.
  - **Areal pixel density destroyed: 55.36%**. Linear spatial resolution degraded by **33.2%**.
* **Information Loss Severity**: **HIGH (CRITICAL BOTTLENECK)**
* **Technical Rationale**: This is the primary physical bottleneck for subtle pathologies. Micro-trabecular fracture lines ($<0.4$ mm) and thin ligamentous tears ($1.0$–$2.0$ mm) are blurred into surrounding bone marrow and fat, destroying the diagnostic signal before the neural network ever touches the image.

---

### Stage 8: ViT Patch Tokenization
* **Pipeline Action**: DINOv2 ViT-S/14 patch projection divides $256\times 256$ slices into non-overlapping $14\times 14$ pixel patches.
* **Measured Evidence**:
  - Each patch token covers $(14 \times 0.50\text{ mm}) \times (14 \times 0.50\text{ mm}) = \mathbf{7.0\text{ mm} \times 7.0\text{ mm}}$ physical anatomy ($49\text{ mm}^2$).
  - The superficial MCL is approximately $1.5\text{ mm}$ thick.
* **Information Loss Severity**: **HIGH**
* **Technical Rationale**: A $1.5\text{ mm}$ ligament occupying $3\text{ mm}^2$ inside a $49\text{ mm}^2$ patch accounts for less than 7% of the linear projection weight. The signal is swamped by surrounding tissue. This explains why **D8-C's local depthwise convolutional stem** ($3\times 3$ receptive field) outperformed pure ViT patch tokens on Gold-58 (`0.6487` vs `0.6331`).

---

### Stage 9: Slot Aggregation
* **Pipeline Action**: Concatenate slot embeddings across the 6 slots, using zero-filled dummy embeddings for missing slots.
* **Measured Evidence**:
  - Missing slot rates: Slot 3 (80.41%), Slot 4 (22.72%).
* **Information Loss Severity**: **LOW-MEDIUM**
* **Technical Rationale**: Cross-slot self-attention heads learn to mask zero-padded tokens, but the presence of empty slots introduces noise in the pooled global study representation.

---

### Stage 10: Pathology Classification Head
* **Pipeline Action**: Linear projection from the pooled representation to 12 independent binary logits evaluated with Sigmoid BCE.
* **Measured Evidence**:
  - Cross-entropy loss with sample weighting.
* **Information Loss Severity**: **LOW**
* **Technical Rationale**: The classification head is standard, well-calibrated, and mathematically sufficient for multi-label binary evaluation.

---

## 3. Forensic Chain Severity Summary Table

| Pipeline Stage | Operational Mechanism | Measured HPC Evidence | Information Loss Severity | Primary Impacted Pathology Targets |
| :--- | :--- | :--- | :---: | :--- |
| **1. Series Selection** | Filter canonical series, discard surplus | 24,033 series across 4,349 studies | **MEDIUM** | ACL, Meniscus Root |
| **2. Sequence Resolution**| 6-slot allocation | Slot 3 missing in 80.41% | **HIGH** | Meniscal tears, Contusion |
| **3. Collision Resolution**| Single winner selected per slot | 39.94% collision rate | **MEDIUM** | Effusion, Meniscal tears |
| **4. 32-Slice Resampling** | Linear z-axis interpolation | Frozen 32-slice tensor | **MEDIUM** | Cartilage edges, Small tears |
| **5. Central-Band Trunc.** | Discard slices 0–5 and 26–31 | Peripheral tissue: 58.5% & 76.9% | **HIGH** | **MCL (0.6216 AUC)**, Baker's |
| **6. 3-Slice Grouping** | Sparse 8-group sampling | Bypasses intermediate slices | **MEDIUM** | Hairline fractures, Meniscus |
| **7. 256×256 Resize** | Bilinear spatial downsampling | **55.4% pixel density destroyed** | **HIGH (CRITICAL)**| **Fracture (0.7403)**, MCL, Menisci |
| **8. Patch Tokenization** | $14\times 14$ ViT patch projection | **$7.0\times 7.0\text{ mm}$ token footprint** | **HIGH** | MCL, Fracture lines |
| **9. Slot Aggregation** | Zero-padding missing slots | Slot 3 empty in 80.4% of studies | **LOW-MEDIUM** | Global study feature pooling |
| **10. Classifier Head** | Multi-label linear BCE head | Standard calibration | **LOW** | Negligible loss |
