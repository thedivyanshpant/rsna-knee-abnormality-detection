# Phase C3 Stage 10: 2-GPU DDP Sanity Audit Report

- **Execution Timestamp**: 2026-09-12T08:38:39.516198+00:00
- **Hardware**: 2 × NVIDIA A100 80GB PCIe
- **DDP Backend**: NCCL
- **Effective Batch Size**: 4 (2/GPU)
- **Cross-Rank Parameter Synchronization**: **PASS** (`param_diff == 0.0`)
- **Cross-Rank Disjoint Sampling**: **PASS** (Zero sample overlap within epochs)
- **Multi-GPU Throughput**: **11.0 studies/sec**
- **Peak VRAM / GPU**: `656.8 MB` (0.80% of 80GB)
- **Checkpointing**: Rank-Safe (`ddp_checkpoint.pt`)
- **Gate Status**: **PASS**

## Epoch History

| Epoch | Global Train Loss | Global Val Loss | Epoch Time (s) | Throughput (studies/s) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.69008 | 0.63803 | 6.87 | 9.3 |
| 2 | 0.53446 | 0.52409 | 4.82 | 13.3 |
