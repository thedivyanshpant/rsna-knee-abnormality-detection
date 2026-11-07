# Phase C3 Stage 9: Single-A100 Sanity Audit Report

- **Execution Timestamp**: 2026-09-12T09:36:11.562073+00:00
- **Hardware**: NVIDIA A100 80GB PCIe (80GB PCIe)
- **Precision**: BFloat16 AMP
- **Batch Size**: 2
- **Peak VRAM Allocated**: `691.4 MB` (0.84% of 80GB)
- **Mean Training Throughput**: `5.7 studies/sec`
- **Mean Epoch Time (32 studies)**: `5.58 s`
- **Checkpoint Saved & Resumed**: **PASS** (checkpoint_epoch_2.pt)
- **Gate Decision**: **PASS**

## Epoch History

| Epoch | Train Loss | Val Loss | Epoch Time (s) | Throughput (studies/s) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.70191 | 0.65963 | 7.51 | 4.3 |
| 2 | 0.53908 | 0.52687 | 4.65 | 6.9 |
| 3 | 0.49534 | 0.48533 | 4.58 | 7.0 |
