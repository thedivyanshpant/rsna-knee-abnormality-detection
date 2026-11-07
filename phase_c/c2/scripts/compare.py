"""
Phase C2: Final comparison report generator.

Reads all 4 experiment summaries and produces a structured comparison
answering: Does preprocessing materially affect downstream performance,
and which gives the best trade-off?
"""
import json
import os
import sys
import pandas as pd
import numpy as np

from config import RESULTS_DIR, PIPELINE_NAMES, TARGETS


def load_summaries():
    """Load all available experiment summaries."""
    summaries = {}
    for pipeline in PIPELINE_NAMES:
        path = os.path.join(RESULTS_DIR, f"{pipeline}_experiment_summary.json")
        if os.path.exists(path):
            with open(path) as f:
                summaries[pipeline] = json.load(f)
    return summaries


def generate_comparison():
    """Generate the full comparison report."""
    summaries = load_summaries()
    if len(summaries) < 4:
        print(f"WARNING: Only {len(summaries)} of 4 experiments found: {list(summaries.keys())}")
        if len(summaries) == 0:
            print("No results to compare. Run experiments first.")
            return

    print("\n" + "=" * 80)
    print("  PHASE C2: RAW vs PIPELINE A vs PIPELINE B vs PIPELINE C")
    print("  Final Controlled Experiment Comparison")
    print("=" * 80)

    # ── 1. Headline Metrics ──
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    HEADLINE METRICS                            │")
    print("├──────────────┬──────────┬──────────┬──────────┬───────────────┤")
    print("│ Metric       │   RAW    │ Pipe A   │ Pipe B   │    Pipe C     │")
    print("├──────────────┼──────────┼──────────┼──────────┼───────────────┤")

    metrics_order = [
        ("Macro F1", "best_macro_f1"),
        ("Macro AUC", "best_macro_auc"),
        ("Study Acc", "best_exact_study_accuracy"),
        ("Val Loss", "best_val_loss"),
        ("Best Epoch", "best_epoch"),
        ("Train Time(s)", "total_training_time_sec"),
        ("Peak VRAM(MB)", "peak_vram_mb"),
        ("Params", "total_params"),
    ]

    for display_name, key in metrics_order:
        values = []
        for p in PIPELINE_NAMES:
            if p in summaries:
                v = summaries[p].get(key, "N/A")
                if isinstance(v, float):
                    values.append(f"{v:.4f}")
                elif isinstance(v, int):
                    values.append(f"{v:,}")
                else:
                    values.append(str(v))
            else:
                values.append("N/A")
        print(f"│ {display_name:<12} │ {values[0]:>8} │ {values[1]:>8} │ {values[2]:>8} │ {values[3]:>13} │")

    print("└──────────────┴──────────┴──────────┴──────────┴───────────────┘")

    # ── 2. Per-Target F1 ──
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    PER-TARGET F1 SCORES                        │")
    print("├──────────────────┬──────┬──────┬──────┬────────────────────────┤")
    print("│ Target           │ RAW  │ A    │ B    │ C                      │")
    print("├──────────────────┼──────┼──────┼──────┼────────────────────────┤")

    for target in TARGETS:
        values = []
        for p in PIPELINE_NAMES:
            if p in summaries and "per_target_f1" in summaries[p]:
                v = summaries[p]["per_target_f1"].get(target, float("nan"))
                values.append(f"{v:.3f}" if not np.isnan(v) else "N/A")
            else:
                values.append("N/A")
        short_target = target[:16]
        print(f"│ {short_target:<16} │ {values[0]:>4} │ {values[1]:>4} │ {values[2]:>4} │ {values[3]:>22} │")

    print("└──────────────────┴──────┴──────┴──────┴────────────────────────┘")

    # ── 3. Per-Target AUC ──
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    PER-TARGET ROC-AUC                          │")
    print("├──────────────────┬──────┬──────┬──────┬────────────────────────┤")
    print("│ Target           │ RAW  │ A    │ B    │ C                      │")
    print("├──────────────────┼──────┼──────┼──────┼────────────────────────┤")

    for target in TARGETS:
        values = []
        for p in PIPELINE_NAMES:
            if p in summaries and "per_target_auc" in summaries[p]:
                v = summaries[p]["per_target_auc"].get(target, float("nan"))
                values.append(f"{v:.3f}" if not np.isnan(v) else "N/A")
            else:
                values.append("N/A")
        short_target = target[:16]
        print(f"│ {short_target:<16} │ {values[0]:>4} │ {values[1]:>4} │ {values[2]:>4} │ {values[3]:>22} │")

    print("└──────────────────┴──────┴──────┴──────┴────────────────────────┘")

    # ── 4. Storage/RAM Trade-off ──
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│              STORAGE & EFFICIENCY TRADE-OFF                    │")
    print("├──────────────────┬──────────┬──────────┬──────────┬────────────┤")

    # Pull C0 numbers
    c0_storage = {
        "raw": {"stored_gb": 10.1808, "ram_mb_study": 260.07, "extrapolated_gb": 523.54},
        "pipeline_a": {"stored_gb": 5.5339, "ram_mb_study": 260.07, "extrapolated_gb": 284.58},
        "pipeline_b": {"stored_gb": 5.0150, "ram_mb_study": 82.69, "extrapolated_gb": 257.89},
        "pipeline_c": {"stored_gb": 1.3715, "ram_mb_study": 22.75, "extrapolated_gb": 70.53},
    }

    print("│ Metric           │   RAW    │ Pipe A   │ Pipe B   │  Pipe C    │")
    print("├──────────────────┼──────────┼──────────┼──────────┼────────────┤")
    for metric_name, key in [
        ("Pilot Store(GB)", "stored_gb"),
        ("RAM/Study(MB)", "ram_mb_study"),
        ("Full DS(GB)", "extrapolated_gb"),
    ]:
        values = [f"{c0_storage[p][key]:.2f}" for p in PIPELINE_NAMES]
        print(f"│ {metric_name:<16} │ {values[0]:>8} │ {values[1]:>8} │ {values[2]:>8} │ {values[3]:>10} │")

    # Add predictive performance per GB
    for p in PIPELINE_NAMES:
        if p in summaries:
            f1 = summaries[p].get("best_macro_f1", 0)
            gb = c0_storage[p]["extrapolated_gb"]
            c0_storage[p]["f1_per_gb"] = f1 / gb if gb > 0 else 0
    values = []
    for p in PIPELINE_NAMES:
        if "f1_per_gb" in c0_storage[p]:
            values.append(f"{c0_storage[p]['f1_per_gb']:.5f}")
        else:
            values.append("N/A")
    print(f"│ {'F1/GB Ratio':<16} │ {values[0]:>8} │ {values[1]:>8} │ {values[2]:>8} │ {values[3]:>10} │")

    print("└──────────────────┴──────────┴──────────┴──────────┴────────────┘")

    # ── 5. Save comparison CSV ──
    rows = []
    for p in PIPELINE_NAMES:
        if p in summaries:
            s = summaries[p]
            row = {
                "pipeline": p,
                "macro_f1": s.get("best_macro_f1"),
                "macro_auc": s.get("best_macro_auc"),
                "exact_study_accuracy": s.get("best_exact_study_accuracy"),
                "val_loss": s.get("best_val_loss"),
                "best_epoch": s.get("best_epoch"),
                "training_time_sec": s.get("total_training_time_sec"),
                "peak_vram_mb": s.get("peak_vram_mb"),
                "total_params": s.get("total_params"),
                "pilot_stored_gb": c0_storage[p]["stored_gb"],
                "ram_mb_per_study": c0_storage[p]["ram_mb_study"],
                "extrapolated_full_gb": c0_storage[p]["extrapolated_gb"],
            }
            # Add per-target F1
            for target in TARGETS:
                row[f"f1_{target}"] = s.get("per_target_f1", {}).get(target)
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "c2_final_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nComparison CSV saved: {csv_path}")

    # Save as JSON too
    json_path = os.path.join(RESULTS_DIR, "c2_final_comparison.json")
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Comparison JSON saved: {json_path}")

    return summaries


if __name__ == "__main__":
    generate_comparison()
