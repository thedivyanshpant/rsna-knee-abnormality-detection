# Phase C3 Stage 9: Single-A100 Sanity Audit Report

- **Execution Timestamp**: 2026-09-12T08:37:35.901027+00:00
- **Hardware**: NVIDIA A100 80GB PCIe (80GB PCIe)
- **Precision**: BFloat16 AMP
- **Batch Size**: 2
- **Peak VRAM Allocated**: `691.4 MB` (0.84% of 80GB)
- **Mean Training Throughput**: `5.6 studies/sec`
- **Mean Epoch Time (32 studies)**: `5.76 s`
- **Checkpoint Saved & Resumed**: **PASS** (checkpoint_epoch_2.pt)
- **Gate Decision**: **PASS**

## Epoch History

| Epoch | Train Loss | Val Loss | Epoch Time (s) | Throughput (studies/s) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.70191 | 0.65963 | 7.97 | 4.0 |
| 2 | 0.53908 | 0.52687 | 4.72 | 6.8 |
| 3 | 0.49534 | 0.48533 | 4.61 | 6.9 |
