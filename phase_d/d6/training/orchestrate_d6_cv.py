#!/usr/bin/env python3
"""
Phase D6 Master 5-Fold CV Orchestrator, Multi-Node Runner & Evaluation Engine.
RSNA Knee Abnormality Detection

Orchestrates:
1. Multi-node DDP training of DINOv2 ViT-S/14 on Phase D6 6-Slot Resolved Representation across Folds 0..4:
   - Round 1: Fold 0 (rpgpu007), Fold 1 (rpgpu018), Fold 2 (rpgpu028) in parallel (2 x A100 per node)
   - Round 2: Fold 3 (rpgpu007), Fold 4 (rpgpu018) in parallel (2 x A100 per node)
2. Global 4,349-study OOF aggregation and verification.
3. Dual-ground-truth evaluation:
   - Remediated D3 ground truth (isolating representation impact on remediated labels)
   - Legacy Phase B ground truth (historical benchmark continuity)
4. Comprehensive statistical metrics JSON, per-target CSV, and classification into CASE A/B/C/D.
"""

import os
import sys
import time
import json
import hashlib
import subprocess
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

import shutil

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase_d", "d1", "scripts"))
from d1_2_eval_utils import (
    roc_auc_score_np,
    pr_auc_score_np,
    evaluate_oof_df,
    PATHOLOGY_NAMES,
)

CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
D3_LABELS_CSV = os.path.join(PROJECT_ROOT, "phase_d/d3/labels/phase_d3_candidate_label_corpus.csv")
PROD_LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
GOLD_CSV = os.path.join(PROJECT_ROOT, "csv/gold_studies.csv")

D6_ROOT = os.path.join(PROJECT_ROOT, "phase_d/d6")
D6_TRAIN_SCRIPT = os.path.join(D6_ROOT, "training/train_ddp_d6.py")
D6_CHECKPOINTS_DIR = os.path.join(D6_ROOT, "checkpoints")
D6_VAL_PRED_DIR = os.path.join(D6_ROOT, "predictions/val")
D6_OOF_DIR = os.path.join(D6_ROOT, "oof")
D6_METRICS_DIR = os.path.join(D6_ROOT, "metrics")
D6_REPORTS_DIR = os.path.join(D6_ROOT, "reports")
D6_LOGS_DIR = os.path.join(D6_ROOT, "logs")

D1_TEMPLATE_OOF_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/results/oof/d1_1_global_oof_predictions.csv")

PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)
TORCHRUN_BIN = os.environ.get("TORCHRUN_BIN", shutil.which("torchrun") or "torchrun")


def launch_fold_on_node(fold: int, node_host: str, master_port: int) -> subprocess.Popen:
    """Launch 2-GPU DDP training for a specific fold on a target node."""
    log_file = os.path.join(D6_LOGS_DIR, f"fold_{fold}_training.log")
    os.makedirs(D6_LOGS_DIR, exist_ok=True)
    f_log = open(log_file, "w")

    cmd_inner = (
        f"cd {PROJECT_ROOT} && "
        f"NCCL_SOCKET_IFNAME=eth2 GLOO_SOCKET_IFNAME=eth2 {TORCHRUN_BIN} "
        f"--nproc_per_node=2 --master_port={master_port} {D6_TRAIN_SCRIPT} "
        f"--fold={fold} --epochs=10 --batch_size=2 --grad_accum=2 "
        f"--backbone_lr=8e-6 --head_lr=1e-3 --weight_decay=0.02"
    )

    if node_host in ["localhost", "rpgpu007"]:
        print(f"[LAUNCH] Fold {fold} on LOCAL node rpgpu007 (port {master_port}) -> {log_file}", flush=True)
        env = os.environ.copy()
        env["NCCL_SOCKET_IFNAME"] = "eth2"
        env["GLOO_SOCKET_IFNAME"] = "eth2"
        cmd = [
            TORCHRUN_BIN,
            "--nproc_per_node=2",
            f"--master_port={master_port}",
            D6_TRAIN_SCRIPT,
            f"--fold={fold}",
            "--epochs=10",
            "--batch_size=2",
            "--grad_accum=2",
            "--backbone_lr=8e-6",
            "--head_lr=1e-3",
            "--weight_decay=0.02"
        ]
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=f_log, stderr=subprocess.STDOUT)
    else:
        print(f"[LAUNCH] Fold {fold} on REMOTE node {node_host} (port {master_port}) via SSH -> {log_file}", flush=True)
        ssh_cmd = f'ssh -o StrictHostKeyChecking=no {node_host} "{cmd_inner}"'
        proc = subprocess.Popen(ssh_cmd, cwd=PROJECT_ROOT, shell=True, stdout=f_log, stderr=subprocess.STDOUT)

    return proc


def train_5_folds_concurrent():
    """Train all 5 folds using 2 rounds across available cluster nodes."""
    print("=" * 80)
    print("PHASE D6: 5-FOLD MULTI-NODE CONCURRENT TRAINING ORCHESTRATION")
    print("=" * 80)

    # Check already completed folds
    folds_to_run = []
    for f in range(5):
        ckpt = os.path.join(D6_CHECKPOINTS_DIR, f"fold_{f}", "best_model.pt")
        pred = os.path.join(D6_VAL_PRED_DIR, f"fold_{f}_val_predictions.csv")
        if os.path.exists(ckpt) and os.path.exists(pred):
            print(f"[SKIP] Fold {f} already completed: {ckpt}", flush=True)
        else:
            folds_to_run.append(f)

    if not folds_to_run:
        print("[ALL COMPLETE] All 5 folds are already completed. Proceeding to aggregation.", flush=True)
        return

    # Round 1: Folds 0, 1, 2
    round_1_folds = [f for f in folds_to_run if f in [0, 1, 2]]
    if round_1_folds:
        print(f"\n--- LAUNCHING ROUND 1: Folds {round_1_folds} ---", flush=True)
        t_r1_start = time.time()
        node_assignments = {0: "rpgpu007", 1: "rpgpu018", 2: "rpgpu028"}
        port_assignments = {0: 29540, 1: 29541, 2: 29542}
        procs_r1 = []

        for f in round_1_folds:
            node = node_assignments[f]
            port = port_assignments[f]
            p = launch_fold_on_node(f, node, port)
            procs_r1.append((f, node, p))

        # Wait for Round 1
        for f, node, p in procs_r1:
            ret = p.wait()
            if ret != 0:
                print(f"[ERROR] Fold {f} on {node} failed with return code {ret}!", flush=True)
                sys.exit(1)
            print(f"[PASS] Fold {f} on {node} completed successfully!", flush=True)
        print(f"--- ROUND 1 FINISHED in {time.time() - t_r1_start:.1f}s ---\n", flush=True)

    # Round 2: Folds 3, 4
    round_2_folds = [f for f in folds_to_run if f in [3, 4]]
    if round_2_folds:
        print(f"\n--- LAUNCHING ROUND 2: Folds {round_2_folds} ---", flush=True)
        t_r2_start = time.time()
        node_assignments = {3: "rpgpu007", 4: "rpgpu018"}
        port_assignments = {3: 29543, 4: 29544}
        procs_r2 = []

        for f in round_2_folds:
            node = node_assignments[f]
            port = port_assignments[f]
            p = launch_fold_on_node(f, node, port)
            procs_r2.append((f, node, p))

        # Wait for Round 2
        for f, node, p in procs_r2:
            ret = p.wait()
            if ret != 0:
                print(f"[ERROR] Fold {f} on {node} failed with return code {ret}!", flush=True)
                sys.exit(1)
            print(f"[PASS] Fold {f} on {node} completed successfully!", flush=True)
        print(f"--- ROUND 2 FINISHED in {time.time() - t_r2_start:.1f}s ---\n", flush=True)


def aggregate_d6_oof() -> pd.DataFrame:
    """Aggregate 5-fold validation predictions into standardized global 4,349-study OOF format."""
    print("\n" + "=" * 80)
    print("  AGGREGATING PHASE D6 5-FOLD OUT-OF-FOLD (OOF) PREDICTIONS")
    print("=" * 80, flush=True)

    template_df = pd.read_csv(D1_TEMPLATE_OOF_CSV)
    uids_order = template_df["StudyInstanceUID"].tolist()

    val_dfs = []
    for fold in range(5):
        p_path = os.path.join(D6_VAL_PRED_DIR, f"fold_{fold}_val_predictions.csv")
        assert os.path.exists(p_path), f"Missing validation predictions for Fold {fold}: {p_path}"
        df_k = pd.read_csv(p_path)
        val_dfs.append(df_k)

    merged_val = pd.concat(val_dfs, ignore_index=True)
    assert len(merged_val) == 4349, f"Expected 4349 OOF studies, got {len(merged_val)}"
    assert merged_val["StudyInstanceUID"].nunique() == 4349, "Duplicate studies in OOF predictions!"

    # Align with template order
    merged_val = merged_val.set_index("StudyInstanceUID").reindex(uids_order).reset_index()

    # Integrity assertions
    assert not merged_val.isna().any().any(), "NaN detected in aggregated OOF predictions!"
    assert not np.isinf(merged_val.select_dtypes(include=[np.number]).values).any(), "Inf in OOF predictions!"

    gold_df = pd.read_csv(GOLD_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())
    leakage = set(merged_val["StudyInstanceUID"].tolist()) & gold_uids
    assert len(leakage) == 0, f"FATAL: Gold studies detected in OOF: {leakage}"

    os.makedirs(D6_OOF_DIR, exist_ok=True)
    oof_csv = os.path.join(D6_OOF_DIR, "d6_global_oof_predictions.csv")
    merged_val.to_csv(oof_csv, index=False)
    print(f"[SAVED] Global OOF predictions: {oof_csv} (4,349 rows, exactly 0 Gold studies)", flush=True)
    return merged_val


def evaluate_d6_dual(oof_df: pd.DataFrame) -> Dict[str, Any]:
    """Execute rigorous dual evaluation on global OOF against Remediated D3 and Legacy Phase B GT."""
    print("\n" + "=" * 80)
    print("  PHASE D6 DUAL-GROUND-TRUTH AUTHORITATIVE EVALUATION")
    print("=" * 80, flush=True)

    # 1. Evaluate against Remediated D3 Ground Truth
    res_remediated = evaluate_oof_df(oof_df, experiment_id="D6_REMEDIATED")

    # 2. Evaluate against Original Production Labels (Legacy Phase B)
    prod_labels_df = pd.read_csv(PROD_LABELS_CSV)
    prod_targets_map = {}
    prod_weights_map = {}
    for uid, grp in prod_labels_df.groupby("StudyInstanceUID"):
        grp_indexed = grp.set_index("target")
        t_arr = [grp_indexed.loc[t, "label_probability"] if t in grp_indexed.index else 0.20 for t in PATHOLOGY_NAMES]
        w_arr = [grp_indexed.loc[t, "sample_weight"] if t in grp_indexed.index else 0.85 for t in PATHOLOGY_NAMES]
        prod_targets_map[uid] = t_arr
        prod_weights_map[uid] = w_arr

    oof_legacy = oof_df.copy()
    for t_idx, t in enumerate(PATHOLOGY_NAMES):
        oof_legacy[f"true_{t}"] = [prod_targets_map[u][t_idx] for u in oof_df["StudyInstanceUID"]]
        oof_legacy[f"weight_{t}"] = [prod_weights_map[u][t_idx] for u in oof_df["StudyInstanceUID"]]

    res_legacy = evaluate_oof_df(oof_legacy, experiment_id="D6_LEGACY")

    # D3 authoritative baselines:
    D3_REMEDIATED_MACRO_AUC = 0.8135
    D3_REMEDIATED_WEIGHTED_AUC = 0.8204
    D3_REMEDIATED_PRAUC = 0.3080
    D3_LEGACY_MACRO_AUC = 0.8184
    D3_LEGACY_PRAUC = 0.2319

    d6_rem_unw = res_remediated["global_macro_roc_auc_unweighted"]
    d6_rem_wtd = res_remediated["global_macro_roc_auc_weighted"]
    d6_rem_prauc = res_remediated["global_macro_pr_auc"]

    d6_leg_unw = res_legacy["global_macro_roc_auc_unweighted"]
    d6_leg_wtd = res_legacy["global_macro_roc_auc_weighted"]
    d6_leg_prauc = res_legacy["global_macro_pr_auc"]

    delta_rem_auc = d6_rem_unw - D3_REMEDIATED_MACRO_AUC
    delta_rem_prauc = d6_rem_prauc - D3_REMEDIATED_PRAUC
    rel_gain_rem_pct = (delta_rem_auc / D3_REMEDIATED_MACRO_AUC) * 100.0

    delta_leg_auc = d6_leg_unw - D3_LEGACY_MACRO_AUC
    delta_leg_prauc = d6_leg_prauc - D3_LEGACY_PRAUC

    # Compute fold mean and std
    fold_aucs_rem = [res_remediated["per_fold"][f"fold_{k}"]["macro_roc_auc_unweighted"] for k in range(5)]
    fold_wtd_rem = [res_remediated["per_fold"][f"fold_{k}"]["macro_roc_auc_weighted"] for k in range(5)]
    fold_mean_auc = float(np.mean(fold_aucs_rem))
    fold_std_auc = float(np.std(fold_aucs_rem))

    # Classification logic
    if delta_rem_auc >= 0.0050:
        verdict = "CASE A — STRONG REPRESENTATION IMPROVEMENT"
    elif 0.0005 <= delta_rem_auc < 0.0050:
        verdict = "CASE B — MARGINAL REPRESENTATION IMPROVEMENT"
    elif -0.0020 <= delta_rem_auc < 0.0005:
        verdict = "CASE C — NO MEANINGFUL IMPROVEMENT"
    else:
        verdict = "CASE D — REPRESENTATION REGRESSION"

    print(f"\n--- AUTHORITATIVE RESULTS ---")
    print(f"D3 Baseline Remediated Macro ROC-AUC: {D3_REMEDIATED_MACRO_AUC:.4f} | PR-AUC: {D3_REMEDIATED_PRAUC:.4f}")
    print(f"D6 Candidate Remediated Macro ROC-AUC:  {d6_rem_unw:.4f} | PR-AUC: {d6_rem_prauc:.4f}")
    print(f"Delta vs D3 Baseline:                  {delta_rem_auc:+.4f} ({rel_gain_rem_pct:+.2f}%)")
    print(f"Delta PR-AUC vs D3 Baseline:           {delta_rem_prauc:+.4f}")
    print(f"D6 5-Fold Cross-Validation:            {fold_mean_auc:.4f} ± {fold_std_auc:.4f}")
    print(f"Legacy Phase B Macro ROC-AUC:          {d6_leg_unw:.4f} (Delta: {delta_leg_auc:+.4f})")
    print(f"FINAL SCIENTIFIC VERDICT:              {verdict}")

    # Build per-target CSV table
    per_target_rows = []
    d3_summary_path = os.path.join(PROJECT_ROOT, "phase_d/d5/evaluations/d5_oof_evaluation_summary.json")
    with open(d3_summary_path) as f:
        d3_eval = json.load(f)["D5-A_STANDARD"]["remediated"]["per_target"]

    print("\n--- 12-TARGET PER-PATHOLOGY BREAKDOWN ---")
    print(f"{'Target':18s} | {'D3 AUC':8s} | {'D6 AUC':8s} | {'Delta AUC':10s} | {'D3 PR':8s} | {'D6 PR':8s} | {'Delta PR':10s}")
    print("-" * 85)

    for t in PATHOLOGY_NAMES:
        d3_t_auc = d3_eval[t]["roc_auc_unweighted"]
        d3_t_pr = d3_eval[t]["pr_auc"]
        d6_t_auc = res_remediated["per_target"][t]["roc_auc_unweighted"]
        d6_t_pr = res_remediated["per_target"][t]["pr_auc"]
        d_auc = d6_t_auc - d3_t_auc
        d_pr = d6_t_pr - d3_t_pr

        print(f"{t:18s} | {d3_t_auc:8.4f} | {d6_t_auc:8.4f} | {d_auc:+10.4f} | {d3_t_pr:8.4f} | {d6_t_pr:8.4f} | {d_pr:+10.4f}")
        per_target_rows.append({
            "target": t,
            "d3_roc_auc": round(d3_t_auc, 4),
            "d6_roc_auc": round(d6_t_auc, 4),
            "delta_roc_auc": round(d_auc, 4),
            "d3_pr_auc": round(d3_t_pr, 4),
            "d6_pr_auc": round(d6_t_pr, 4),
            "delta_pr_auc": round(d_pr, 4),
            "pos_count": res_remediated["per_target"][t]["pos_count"],
            "neg_count": res_remediated["per_target"][t]["neg_count"]
        })

    per_target_df = pd.DataFrame(per_target_rows)
    per_target_csv = os.path.join(D6_METRICS_DIR, "d6_per_target_metrics.csv")
    os.makedirs(D6_METRICS_DIR, exist_ok=True)
    per_target_df.to_csv(per_target_csv, index=False)

    # Save summary JSON
    summary_out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_id": "PHASE_D6_RICHER_SEQUENCE_RESOLUTION",
        "verdict": verdict,
        "d3_baseline": {
            "remediated_macro_roc_auc": D3_REMEDIATED_MACRO_AUC,
            "remediated_weighted_roc_auc": D3_REMEDIATED_WEIGHTED_AUC,
            "remediated_macro_prauc": D3_REMEDIATED_PRAUC,
            "legacy_macro_roc_auc": D3_LEGACY_MACRO_AUC,
            "legacy_macro_prauc": D3_LEGACY_PRAUC
        },
        "d6_results": {
            "remediated_macro_roc_auc_unweighted": round(d6_rem_unw, 4),
            "remediated_macro_roc_auc_weighted": round(d6_rem_wtd, 4),
            "remediated_macro_prauc": round(d6_rem_prauc, 4),
            "remediated_fold_mean": round(fold_mean_auc, 4),
            "remediated_fold_std": round(fold_std_auc, 4),
            "delta_unweighted_vs_d3": round(delta_rem_auc, 4),
            "delta_prauc_vs_d3": round(delta_rem_prauc, 4),
            "relative_gain_pct": round(rel_gain_rem_pct, 2),
            "legacy_macro_roc_auc_unweighted": round(d6_leg_unw, 4),
            "legacy_macro_roc_auc_weighted": round(d6_leg_wtd, 4),
            "legacy_macro_prauc": round(d6_leg_prauc, 4),
            "delta_legacy_vs_d3": round(delta_leg_auc, 4)
        },
        "per_fold": res_remediated["per_fold"],
        "per_target": {r["target"]: r for r in per_target_rows},
        "audit": {
            "total_studies": len(oof_df),
            "unique_studies": oof_df["StudyInstanceUID"].nunique(),
            "gold_leakage": 0,
            "nan_count": 0,
            "inf_count": 0,
            "pass_status": True
        }
    }

    summary_json_path = os.path.join(D6_METRICS_DIR, "d6_oof_evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\nSaved evaluation summary: {summary_json_path}")
    print(f"Saved per-target metrics: {per_target_csv}")
    return summary_out


def main():
    t0 = time.time()
    train_5_folds_concurrent()
    oof_df = aggregate_d6_oof()
    summary = evaluate_d6_dual(oof_df)
    total_time = time.time() - t0
    print(f"\n[PHASE D6 COMPLETE] Total execution time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
