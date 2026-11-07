#!/usr/bin/env python3
"""
Phase D2 Master 5-Fold Orchestrator, OOF Aggregator & D1.2 Blending Engine.
RSNA Knee Abnormality Detection

Orchestrates:
1. Production DDP training of RadImageNet ResNet-50 across Folds 0..4 on 2 x A100 GPUs.
2. Global 4,349-study OOF aggregation, verification, and metric calculation.
3. Multi-arm ensembling with frozen Phase D1.2-B2 predictions:
   - 50/50 Probability Mean
   - 50/50 Percentile Rank Mean
   - 75% D1.2 + 25% D2 CNN
   - 25% D1.2 + 75% D2 CNN
4. Generates comprehensive structured metrics and integrity records.
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase_d", "d1", "scripts"))
from d1_2_eval_utils import (
    roc_auc_score_np,
    pr_auc_score_np,
    evaluate_oof_df,
    PATHOLOGY_NAMES,
)

CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
GOLD_CSV = os.path.join(PROJECT_ROOT, "csv/gold_studies.csv")

D2_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d2/scripts")
D2_CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d2/checkpoints")
D2_VAL_PRED_DIR = os.path.join(PROJECT_ROOT, "phase_d/d2/predictions/val")
D2_OOF_DIR = os.path.join(PROJECT_ROOT, "phase_d/d2/predictions/oof")
D2_BLENDS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d2/predictions/blends")
D2_REPORTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d2/reports")

D1_TEMPLATE_OOF_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/results/oof/d1_1_global_oof_predictions.csv")
D1_2_B2_OOF_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/predictions/d1_2_b_hflip/d1_2_b_hflip_global_oof.csv")


def run_command(cmd: list, description: str):
    print(f"\n{'='*80}\n[ORCHESTRATOR] {description}\nCOMMAND: {' '.join(cmd)}\n{'='*80}")
    t0 = time.time()
    result = subprocess.run(cmd, env=os.environ.copy())
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[ERROR] {description} FAILED with exit code {result.returncode} in {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"[SUCCESS] {description} completed in {elapsed:.1f}s")


def train_all_folds():
    """Train Folds 0 through 4 sequentially using 2 x A100 DDP."""
    python_bin = sys.executable
    train_script = os.path.join(D2_SCRIPTS_DIR, "train_ddp_d2.py")

    for fold in range(5):
        ckpt_path = os.path.join(D2_CHECKPOINTS_DIR, f"fold_{fold}", "best_model.pt")
        val_pred_path = os.path.join(D2_VAL_PRED_DIR, f"fold_{fold}_val_predictions.csv")

        if os.path.exists(ckpt_path) and os.path.exists(val_pred_path):
            print(f"[RESUME] Fold {fold} already completed. Checkpoint: {ckpt_path}")
            continue

        cmd = [
            python_bin, "-m", "torch.distributed.run",
            "--nproc_per_node=2",
            f"--master_port={29510 + fold}",
            train_script,
            f"--fold={fold}",
            "--epochs=10",
            "--batch_size=2",
            "--grad_accum=2",
            "--backbone_lr=1e-4",
            "--head_lr=1e-3",
        ]
        run_command(cmd, f"Training Phase D2 RadImageNet ResNet-50 Fold {fold}")


def aggregate_d2_oof():
    """Aggregate 5-fold validation predictions into standard global 4,349-study OOF format."""
    print("\n" + "=" * 80)
    print("  AGGREGATING PHASE D2 5-FOLD OUT-OF-FOLD (OOF) PREDICTIONS")
    print("=" * 80)

    template_df = pd.read_csv(D1_TEMPLATE_OOF_CSV)
    uids_order = template_df["StudyInstanceUID"].tolist()

    val_dfs = []
    for fold in range(5):
        p_path = os.path.join(D2_VAL_PRED_DIR, f"fold_{fold}_val_predictions.csv")
        assert os.path.exists(p_path), f"Missing validation predictions for Fold {fold}: {p_path}"
        df_k = pd.read_csv(p_path)
        val_dfs.append(df_k)
        print(f"Fold {fold}: loaded {len(df_k)} studies")

    raw_oof_df = pd.concat(val_dfs, axis=0).set_index("StudyInstanceUID").reindex(uids_order).reset_index()

    # Build standardized OOF dataframe matching d1_1 schema
    standard_oof_df = template_df.copy()
    for t in PATHOLOGY_NAMES:
        standard_oof_df[f"pred_{t}"] = raw_oof_df[t].values

    oof_csv_path = os.path.join(D2_OOF_DIR, "d2_global_oof_predictions.csv")
    standard_oof_df.to_csv(oof_csv_path, index=False)
    print(f"[SAVED] D2 Global OOF predictions to: {oof_csv_path}")

    # Evaluate standalone D2 CNN metrics via standard auditor
    eval_results = evaluate_oof_df(standard_oof_df, experiment_id="D2_STANDALONE_RADIMAGENET_RESNET50")

    summary_json_path = os.path.join(D2_OOF_DIR, "d2_oof_evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"[SAVED] D2 Standalone Evaluation Summary to: {summary_json_path}")

    print("\n" + "-" * 65)
    print(f"D2 Standalone Global Macro ROC-AUC (Unweighted): {eval_results['global_macro_roc_auc_unweighted']:.4f}")
    print(f"D2 Standalone Global Macro ROC-AUC (Weighted):   {eval_results['global_macro_roc_auc_weighted']:.4f}")
    print(f"D2 Standalone Global Macro PR-AUC:               {eval_results['global_macro_pr_auc']:.4f}")
    print("-" * 65)

    return standard_oof_df, eval_results


def evaluate_d2_blends(d2_oof_df: pd.DataFrame):
    """Evaluate controlled multi-arm ensembles between frozen D1.2-B2 and D2 CNN."""
    print("\n" + "=" * 80)
    print("  EVALUATING MULTI-ARM ENSEMBLES: FROZEN D1.2-B2 + D2 RADIMAGENET CNN")
    print("=" * 80)

    assert os.path.exists(D1_2_B2_OOF_CSV), f"Missing frozen D1.2-B2 predictions: {D1_2_B2_OOF_CSV}"
    d1_df = pd.read_csv(D1_2_B2_OOF_CSV)
    uids = d1_df["StudyInstanceUID"].tolist()
    d2_aligned = d2_oof_df.set_index("StudyInstanceUID").reindex(uids).reset_index()

    with open(CONFIG_CV) as f:
        cv_splits = json.load(f)

    blend_specs = [
        ("D2_B1_PROB_50_50", "50% D1.2-B2 + 50% D2 CNN (Prob Mean)", 0.50, 0.50, "prob"),
        ("D2_B2_RANK_50_50", "50% D1.2-B2 + 50% D2 CNN (Within-Fold Rank Mean)", 0.50, 0.50, "rank"),
        ("D2_C1_WEIGHT_75_25", "75% D1.2-B2 + 25% D2 CNN (Within-Fold Rank Blend)", 0.75, 0.25, "rank"),
        ("D2_C2_WEIGHT_25_75", "25% D1.2-B2 + 75% D2 CNN (Within-Fold Rank Blend)", 0.25, 0.75, "rank"),
    ]

    blend_results = {}

    for blend_id, desc, w1, w2, mode in blend_specs:
        print(f"\n--- Evaluating: {blend_id} ({desc}) ---")
        blend_df = d1_df.copy()

        # Iterate per fold to enforce strict within-fold rank normalization
        for fold in range(5):
            val_uids = cv_splits["folds"][f"fold_{fold}"]["val_studies"]
            mask = blend_df["StudyInstanceUID"].isin(val_uids)
            idx = blend_df[mask].index

            p1_fold = d1_df.loc[idx, [f"pred_{t}" for t in PATHOLOGY_NAMES]].values
            p2_fold = d2_aligned.loc[idx, [f"pred_{t}" for t in PATHOLOGY_NAMES]].values

            if mode == "prob":
                p_blend = w1 * p1_fold + w2 * p2_fold
            elif mode == "rank":
                N_fold = len(val_uids)
                r1_fold = np.zeros_like(p1_fold)
                r2_fold = np.zeros_like(p2_fold)
                for c in range(len(PATHOLOGY_NAMES)):
                    r1_fold[:, c] = rankdata(p1_fold[:, c], method="average") / N_fold
                    r2_fold[:, c] = rankdata(p2_fold[:, c], method="average") / N_fold
                p_blend = w1 * r1_fold + w2 * r2_fold

            for c, t in enumerate(PATHOLOGY_NAMES):
                blend_df.loc[idx, f"pred_{t}"] = p_blend[:, c]

        out_csv = os.path.join(D2_BLENDS_DIR, f"{blend_id.lower()}_predictions.csv")
        blend_df.to_csv(out_csv, index=False)

        metrics = evaluate_oof_df(blend_df, experiment_id=blend_id)
        macro_auc = metrics["global_macro_roc_auc_unweighted"]
        macro_prauc = metrics["global_macro_pr_auc"]
        delta_vs_d1_2 = macro_auc - 0.7592

        blend_results[blend_id] = {
            "experiment_id": blend_id,
            "description": desc,
            "macro_roc_auc_unweighted": macro_auc,
            "macro_roc_auc_weighted": metrics["global_macro_roc_auc_weighted"],
            "macro_pr_auc": macro_prauc,
            "delta_vs_d1_2": delta_vs_d1_2,
            "relative_gain_pct": (delta_vs_d1_2 / 0.7592) * 100.0,
            "per_target": metrics["per_target"],
            "prediction_csv": out_csv,
        }
        print(f"[{blend_id}] Unw AUC: {macro_auc:.4f} | Wtd AUC: {metrics['global_macro_roc_auc_weighted']:.4f} | PR-AUC: {macro_prauc:.4f} | Delta vs D1.2: {delta_vs_d1_2:+.4f}")

    blends_summary_path = os.path.join(D2_BLENDS_DIR, "d2_multiarm_blends_summary.json")
    with open(blends_summary_path, "w") as f:
        json.dump(blend_results, f, indent=2)
    print(f"\n[SAVED] Multi-Arm Blends Summary to: {blends_summary_path}")


def main():
    print("=" * 80)
    print("  PHASE D2 MASTER WORKFLOW: RADIMAGENET RESNET-50 5-FOLD CV & BLENDING")
    print("=" * 80)
    train_all_folds()
    d2_oof_df, _ = aggregate_d2_oof()
    evaluate_d2_blends(d2_oof_df)
    print("\n[PHASE D2 COMPLETE] All training, OOF evaluation, and multi-arm blending concluded.")


if __name__ == "__main__":
    main()
