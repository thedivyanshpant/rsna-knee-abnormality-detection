# Phase C3 Stage 8: 8-Study Overfit Sanity Audit

- **Execution Timestamp**: 2026-09-12T09:35:40.360276+00:00
- **Hardware**: NVIDIA A100 80GB PCIe (Device `cuda:0`)
- **Total Trainable Parameters**: 2,052,828
- **Initial Loss**: `0.71316`
- **Final Loss**: `0.43032`
- **Best Loss**: `0.42305`
- **Theoretical Minimum Shannon Entropy Bound**: `0.40771`
- **Gap to Theoretical Minimum**: `0.02261` (Tolerance: < 0.05)
- **Relative Loss Reduction**: **39.66%**
- **Mean Absolute Error (Y vs Pred)**: `0.0516` (Tolerance: < 0.15)
- **Total Runtime**: `59.86 s`
- **Gate Status**: **PASS**

## 1. Selected Non-Gold Studies

1. `1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260`
2. `1.2.826.0.1.3680043.8.498.10004945927472656027199792075652399585`
3. `1.2.826.0.1.3680043.8.498.10013663742400736029415768216599146542`
4. `1.2.826.0.1.3680043.8.498.10018552945042470316597008531304016339`
5. `1.2.826.0.1.3680043.8.498.10018749256316287542884435387067561101`
6. `1.2.826.0.1.3680043.8.498.10021179701366615563817491713738306606`
7. `1.2.826.0.1.3680043.8.498.10021288557171866794624317186288618513`
8. `1.2.826.0.1.3680043.8.498.10025765742726180988537955593315007559`

## 2. Target Level Predictions vs Soft Targets (Sample Study 1)

| Target | Soft Ground Truth Y | Model Output Prob | Sample Weight W |
| :--- | :---: | :---: | :---: |
| ACL | 0.2000 | 0.1621 | 0.8500 |
| MCL | 0.2000 | 0.1592 | 0.8500 |
| Medial Meniscus | 0.2000 | 0.1680 | 0.8500 |
| Lateral Meniscus | 0.2000 | 0.1758 | 0.8500 |
| Medial OA | 0.9500 | 0.8047 | 0.9625 |
| Lateral OA | 0.2000 | 0.1768 | 0.8500 |
| PF OA | 0.2000 | 0.1641 | 0.8500 |
| Effusion | 0.2000 | 0.5625 | 0.8500 |
| Synovitis | 0.2000 | 0.1533 | 0.8500 |
| Baker's | 0.2000 | 0.1816 | 0.8500 |
| Contusion | 0.2000 | 0.1748 | 0.8500 |
| Fracture | 0.2000 | 0.1592 | 0.8500 |
