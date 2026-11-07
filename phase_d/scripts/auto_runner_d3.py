#!/usr/bin/env python3
"""
Phase D Sequential Execution Daemon: Runs D3 CV immediately after D2 CV finishes.
RSNA Knee Abnormality Detection

Ensures:
1. Zero GPU contention between D2 (ResNet-50) and D3 (DINOv2).
2. D2 finishes all 5 folds, aggregations, and multi-arm blends first.
3. D3 automatically begins 5-fold training on remediated labels upon D2 completion.
"""

import os
import sys
import time
import json
import subprocess

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
D2_SUMMARY_JSON = os.path.join(PROJECT_ROOT, "phase_d/d2/predictions/blends/d2_multiarm_blends_summary.json")
D3_ORCHESTRATOR = os.path.join(PROJECT_ROOT, "phase_d/d3/scripts/orchestrate_d3_cv.py")
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)


def main():
    print("=" * 80, flush=True)
    print("  PHASE D AUTOMATED SEQUENTIAL EXECUTION DAEMON", flush=True)
    print("=" * 80, flush=True)
    print(f"Monitoring D2 completion target: {D2_SUMMARY_JSON}", flush=True)

    # Poll every 30 seconds for D2 completion
    while not os.path.exists(D2_SUMMARY_JSON):
        # Check if D2 processes are active
        res = subprocess.run(["pgrep", "-f", "orchestrate_d2_cv.py"], stdout=subprocess.PIPE, text=True)
        if not res.stdout.strip():
            # Check if fold 4 finished
            f4_ckpt = os.path.join(PROJECT_ROOT, "phase_d/d2/checkpoints/fold_4/best_model.pt")
            if not os.path.exists(f4_ckpt):
                print("[WARNING] orchestrate_d2_cv.py is not running and D2 summary not found. Sleeping...", flush=True)
        time.sleep(30)

    print(f"\n[DETECTED] Phase D2 completely finished! Found: {D2_SUMMARY_JSON}", flush=True)
    time.sleep(10)  # brief cool-down

    print("\n" + "=" * 80, flush=True)
    print("  LAUNCHING PHASE D3 5-FOLD DINOv2 CV ON REMEDIATED LABELS", flush=True)
    print("=" * 80, flush=True)

    cmd = [PYTHON_BIN, D3_ORCHESTRATOR]
    t0 = time.time()
    result = subprocess.run(cmd, env=os.environ.copy())
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[ERROR] Phase D3 execution failed with exit code {result.returncode} in {elapsed:.1f}s", flush=True)
        sys.exit(result.returncode)

    print(f"[SUCCESS] Phase D3 execution completed successfully in {elapsed:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
