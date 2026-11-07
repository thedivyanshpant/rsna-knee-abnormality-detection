#!/usr/bin/env python3
"""
Phase D7-D: Full Factorial Combined D7 Orchestrator.
RSNA Knee Abnormality Detection

Orchestrates:
1. Multi-node DDP training of Phase D7-D (ViT-B/14 + Depth Positional Attention Aggregator + 12 Multi-Window Groups) across Folds 0..4:
   - Round 1: Fold 0 (rpgpu007), Fold 1 (rpgpu018), Fold 2 (rpgpu028) in parallel
   - Round 2: Fold 3 (rpgpu007), Fold 4 (rpgpu018) in parallel
2. Global 4,349-study OOF aggregation and integrity audit.
3. Dual-ground-truth evaluation:
   - Remediated D3 ground truth
   - Legacy Phase B ground truth
4. Comprehensive comparison vs D6 Champion, D7-A, D7-B, and D7-C.
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

D7_ROOT = os.path.join(PROJECT_ROOT, "phase_d/d7")
D7_TRAIN_SCRIPT = os.path.join(D7_ROOT, "training/train_ddp_d7.py")
D7D_DIR = os.path.join(D7_ROOT, "d7d")
D7_OOF_DIR = os.path.join(D7_ROOT, "oof")
D7_METRICS_DIR = os.path.join(D7_ROOT, "metrics")
D7_REPORTS_DIR = os.path.join(D7_ROOT, "reports")
D7_LOGS_DIR = os.path.join(D7_ROOT, "logs")

D1_TEMPLATE_OOF_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/results/oof/d1_1_global_oof_predictions.csv")
TORCHRUN_BIN = os.environ.get("TORCHRUN_BIN", shutil.which("torchrun") or "torchrun")

D6_METRICS_PATH = os.path.join(PROJECT_ROOT, "phase_d/d6/metrics/d6_oof_evaluation_summary.json")
D7A_METRICS_PATH = os.path.join(D7_METRICS_DIR, "d7a_oof_evaluation_summary.json")
D7B_METRICS_PATH = os.path.join(D7_METRICS_DIR, "d7b_oof_evaluation_summary.json")
D7C_METRICS_PATH = os.path.join(D7_METRICS_DIR, "d7c_oof_evaluation_summary.json")

NODE_CONFIG = {
    0: {"host": "rpgpu007", "is_local": True},
    1: {"host": "rpgpu018", "is_local": False},
    2: {"host": "rpgpu028", "is_local": False},
    3: {"host": "rpgpu007", "is_local": True},
    4: {"host": "rpgpu018", "is_local": False},
}
PORT_BASE = 29730


def launch_fold_on_node(
    fold: int,
    node_host: str,
    is_local: bool,
    master_port: int,
    epochs: int = 10,
    batch_size: int = 2,
    grad_accum: int = 2,
    backbone_lr: float = 8e-6,
    head_lr: float = 1e-3,
    weight_decay: float = 0.02,
) -> subprocess.Popen:
    log_file = os.path.join(D7_LOGS_DIR, f"d7d_fold_{fold}_training.log")
    os.makedirs(D7_LOGS_DIR, exist_ok=True)
    f_log = open(log_file, "w")

    cmd_inner = (
        f"cd {PROJECT_ROOT} && "
        f"NCCL_SOCKET_IFNAME=eth2 GLOO_SOCKET_IFNAME=eth2 {TORCHRUN_BIN} "
        f"--nproc_per_node=2 --master_port={master_port} {D7_TRAIN_SCRIPT} "
        f"--model_type=d7d --fold={fold} --epochs={epochs} --batch_size={batch_size} "
        f"--grad_accum={grad_accum} --backbone_lr={backbone_lr} --head_lr={head_lr} "
        f"--weight_decay={weight_decay}"
    )

    if is_local:
        print(f"[D7-D LAUNCH] Fold {fold} on LOCAL node {node_host} (port {master_port}) -> {log_file}", flush=True)
        env = os.environ.copy()
        env["NCCL_SOCKET_IFNAME"] = "eth2"
        env["GLOO_SOCKET_IFNAME"] = "eth2"
        cmd = [
            TORCHRUN_BIN,
            "--nproc_per_node=2",
            f"--master_port={master_port}",
            D7_TRAIN_SCRIPT,
            "--model_type=d7d",
            f"--fold={fold}",
            f"--epochs={epochs}",
            f"--batch_size={batch_size}",
            f"--grad_accum={grad_accum}",
            f"--backbone_lr={backbone_lr}",
            f"--head_lr={head_lr}",
            f"--weight_decay={weight_decay}",
        ]
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=f_log, stderr=subprocess.STDOUT)
    else:
        print(f"[D7-D LAUNCH] Fold {fold} on REMOTE node {node_host} (port {master_port}) via SSH -> {log_file}", flush=True)
        ssh_cmd = f'ssh -o StrictHostKeyChecking=no {node_host} "{cmd_inner}"'
        proc = subprocess.Popen(ssh_cmd, cwd=PROJECT_ROOT, shell=True, stdout=f_log, stderr=subprocess.STDOUT)

    return proc


def train_d7d_folds() -> None:
    ckpt_dir = os.path.join(D7D_DIR, "checkpoints")
    pred_dir = os.path.join(D7D_DIR, "predictions", "val")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE D7-D: FULL FACTORIAL COMBINATION TRAINING (FOLDS 0..4)")
    print("=" * 80, flush=True)

    folds_to_run = []
    for f in range(5):
        ckpt_path = os.path.join(ckpt_dir, f"fold_{f}", "best_model.pt")
        pred_path = os.path.join(pred_dir, f"fold_{f}_val_predictions.csv")
        if os.path.exists(ckpt_path) and os.path.exists(pred_path) and os.path.getsize(ckpt_path) > 10_000_000:
            print(f"[SKIP] D7-D Fold {f} already completed: {ckpt_path}", flush=True)
        else:
            folds_to_run.append(f)

    if not folds_to_run:
        print("[ALL COMPLETE] All 5 folds for D7-D already completed.", flush=True)
        return

    # Round 1: Folds 0, 1, 2
    round_1_folds = [f for f in folds_to_run if f in [0, 1, 2]]
    if round_1_folds:
        print(f"\n--- D7-D ROUND 1: Launching Folds {round_1_folds} in Parallel ---", flush=True)
        t_r1_start = time.time()
        procs_r1 = []
        for f in round_1_folds:
            cfg = NODE_CONFIG[f]
            port = PORT_BASE + f
            p = launch_fold_on_node(f, cfg["host"], cfg["is_local"], port)
            procs_r1.append((f, cfg["host"], p))

        for f, node, p in procs_r1:
            ret = p.wait()
            if ret != 0:
                print(f"[ERROR] D7-D Fold {f} on {node} failed with exit code {ret}!", flush=True)
                sys.exit(1)
            print(f"[PASS] D7-D Fold {f} on {node} finished successfully!", flush=True)
        print(f"--- D7-D ROUND 1 FINISHED in {time.time() - t_r1_start:.1f}s ---\n", flush=True)

    # Round 2: Folds 3, 4
    round_2_folds = [f for f in folds_to_run if f in [3, 4]]
    if round_2_folds:
        print(f"\n--- D7-D ROUND 2: Launching Folds {round_2_folds} in Parallel ---", flush=True)
        t_r2_start = time.time()
        procs_r2 = []
        for f in round_2_folds:
            cfg = NODE_CONFIG[f]
            port = PORT_BASE + f
            p = launch_fold_on_node(f, cfg["host"], cfg["is_local"], port)
            procs_r2.append((f, cfg["host"], p))

        for f, node, p in procs_r2:
            ret = p.wait()
            if ret != 0:
                print(f"[ERROR] D7-D Fold {f} on {node} failed with exit code {ret}!", flush=True)
                sys.exit(1)
            print(f"[PASS] D7-D Fold {f} on {node} finished successfully!", flush=True)
        print(f"--- D7-D ROUND 2 FINISHED in {time.time() - t_r2_start:.1f}s ---\n", flush=True)


def aggregate_d7d_oof() -> pd.DataFrame:
    print("\n[AGGREGATE] Aggregating 5-Fold OOF Predictions for D7-D...", flush=True)
    template_df = pd.read_csv(D1_TEMPLATE_OOF_CSV)
    uids_order = template_df["StudyInstanceUID"].tolist()
    pred_dir = os.path.join(D7D_DIR, "predictions", "val")

    val_dfs = []
    for f in range(5):
        p_path = os.path.join(pred_dir, f"fold_{f}_val_predictions.csv")
        assert os.path.exists(p_path), f"Missing predictions for D7-D fold {f}: {p_path}"
        val_dfs.append(pd.read_csv(p_path))

    merged = pd.concat(val_dfs, ignore_index=True)
    assert len(merged) == 4349, f"Expected 4349 OOF studies, got {len(merged)}"
    assert merged["StudyInstanceUID"].nunique() == 4349, "Duplicate studies in OOF!"

    merged = merged.set_index("StudyInstanceUID").reindex(uids_order).reset_index()

    assert not merged.isna().any().any(), "NaN detected in D7-D OOF predictions!"
    assert not np.isinf(merged.select_dtypes(include=[np.number]).values).any(), "Inf in D7-D OOF!"

    gold_df = pd.read_csv(GOLD_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())
    leakage = set(merged["StudyInstanceUID"].tolist()) & gold_uids
    assert len(leakage) == 0, f"FATAL: Gold studies detected in D7-D OOF: {leakage}"

    os.makedirs(D7_OOF_DIR, exist_ok=True)
    out_csv = os.path.join(D7_OOF_DIR, "d7d_global_oof_predictions.csv")
    merged.to_csv(out_csv, index=False)
    print(f"[SAVED] D7-D OOF predictions: {out_csv} (4,349 rows, exactly 0 Gold studies)", flush=True)
    return merged


def evaluate_d7d(oof_df: pd.DataFrame) -> Dict[str, Any]:
    print("\n[EVALUATION] Executing Dual-Ground-Truth Evaluation for D7-D...", flush=True)

    # 1. Remediated D3 GT
    res_remediated = evaluate_oof_df(oof_df, experiment_id="D7D_REMEDIATED")

    # 2. Original Production GT (Legacy Phase B)
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

    res_legacy = evaluate_oof_df(oof_legacy, experiment_id="D7D_LEGACY")

    # Load Baselines
    with open(D6_METRICS_PATH) as f:
        d6_summary = json.load(f)
    d6_auc = d6_summary["d6_results"]["remediated_macro_roc_auc_unweighted"]
    d6_prauc = d6_summary["d6_results"]["remediated_macro_prauc"]

    with open(D7A_METRICS_PATH) as f:
        d7a_summary = json.load(f)
    d7a_auc = d7a_summary["results"]["remediated_macro_roc_auc_unweighted"]

    with open(D7B_METRICS_PATH) as f:
        d7b_summary = json.load(f)
    d7b_auc = d7b_summary["results"]["remediated_macro_roc_auc_unweighted"]

    with open(D7C_METRICS_PATH) as f:
        d7c_summary = json.load(f)
    d7c_auc = d7c_summary["results"]["remediated_macro_roc_auc_unweighted"]

    d7d_rem_unw = res_remediated["global_macro_roc_auc_unweighted"]
    d7d_rem_wtd = res_remediated["global_macro_roc_auc_weighted"]
    d7d_rem_prauc = res_remediated["global_macro_pr_auc"]

    d7d_leg_unw = res_legacy["global_macro_roc_auc_unweighted"]
    d7d_leg_wtd = res_legacy["global_macro_roc_auc_weighted"]
    d7d_leg_prauc = res_legacy["global_macro_pr_auc"]

    fold_aucs = [res_remediated["per_fold"][f"fold_{k}"]["macro_roc_auc_unweighted"] for k in range(5)]
    fold_mean = float(np.mean(fold_aucs))
    fold_std = float(np.std(fold_aucs))

    delta_d6 = d7d_rem_unw - d6_auc
    delta_d7a = d7d_rem_unw - d7a_auc
    delta_d7b = d7d_rem_unw - d7b_auc
    delta_d7c = d7d_rem_unw - d7c_auc

    # Per-target comparisons
    per_target_summary = {}
    per_target_rows = []
    for t in PATHOLOGY_NAMES:
        t_d6 = d6_summary["per_target"][t]["d6_roc_auc"]
        t_d7a = d7a_summary["per_target"][t]["exp_roc_auc"]
        t_d7b = d7b_summary["per_target"][t]["exp_roc_auc"]
        t_d7c = d7c_summary["per_target"][t]["exp_roc_auc"]
        t_d7d = res_remediated["per_target"][t]["roc_auc_unweighted"]
        t_prauc = res_remediated["per_target"][t]["pr_auc"]

        per_target_summary[t] = {
            "target": t,
            "d6_roc_auc": t_d6,
            "d7a_roc_auc": t_d7a,
            "d7b_roc_auc": t_d7b,
            "d7c_roc_auc": t_d7c,
            "d7d_roc_auc": t_d7d,
            "d7d_pr_auc": t_prauc,
            "delta_vs_d6": round(t_d7d - t_d6, 4) if (t_d7d is not None and t_d6 is not None) else None,
            "delta_vs_d7c": round(t_d7d - t_d7c, 4) if (t_d7d is not None and t_d7c is not None) else None,
            "pos_count": res_remediated["per_target"][t]["pos_count"],
            "neg_count": res_remediated["per_target"][t]["neg_count"],
        }
        per_target_rows.append({
            "Target": t,
            "D6_AUC": t_d6,
            "D7A_AUC": t_d7a,
            "D7B_AUC": t_d7b,
            "D7C_AUC": t_d7c,
            "D7D_AUC": t_d7d,
            "D7D_PRAUC": t_prauc,
            "Delta_vs_D6": round(t_d7d - t_d6, 4) if (t_d7d is not None and t_d6 is not None) else None,
            "Delta_vs_D7C": round(t_d7d - t_d7c, 4) if (t_d7d is not None and t_d7c is not None) else None,
        })

    os.makedirs(D7_METRICS_DIR, exist_ok=True)
    per_target_csv = os.path.join(D7_METRICS_DIR, "d7d_per_target_metrics.csv")
    pd.DataFrame(per_target_rows).to_csv(per_target_csv, index=False)

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_id": "PHASE_D7D_FULL_COMBINATION",
        "results": {
            "remediated_macro_roc_auc_unweighted": round(d7d_rem_unw, 4),
            "remediated_macro_roc_auc_weighted": round(d7d_rem_wtd, 4),
            "remediated_macro_prauc": round(d7d_rem_prauc, 4),
            "remediated_fold_mean": round(fold_mean, 4),
            "remediated_fold_std": round(fold_std, 4),
            "legacy_macro_roc_auc_unweighted": round(d7d_leg_unw, 4),
            "legacy_macro_roc_auc_weighted": round(d7d_leg_wtd, 4),
            "legacy_macro_prauc": round(d7d_leg_prauc, 4),
            "delta_vs_d6": round(delta_d6, 4),
            "delta_vs_d7a": round(delta_d7a, 4),
            "delta_vs_d7b": round(delta_d7b, 4),
            "delta_vs_d7c": round(delta_d7c, 4),
        },
        "per_fold": res_remediated["per_fold"],
        "per_target": per_target_summary,
    }

    summary_json_path = os.path.join(D7_METRICS_DIR, "d7d_oof_evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE D7-D EVALUATION SUMMARY")
    print("=" * 80)
    print(f"D7-D Remediated Macro ROC-AUC: {d7d_rem_unw:.4f} (PR-AUC: {d7d_rem_prauc:.4f})")
    print(f"5-Fold CV Mean ± Std:          {fold_mean:.4f} ± {fold_std:.4f}")
    print(f"Legacy Phase B Macro ROC-AUC:  {d7d_leg_unw:.4f}")
    print(f"Delta vs D6 Champion (0.8185): {delta_d6:+.4f}")
    print(f"Delta vs D7-A (0.8137):        {delta_d7a:+.4f}")
    print(f"Delta vs D7-B (0.8100):        {delta_d7b:+.4f}")
    print(f"Delta vs D7-C (0.8079):        {delta_d7c:+.4f}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Per-Target CSV: {per_target_csv}", flush=True)

    return summary_data


def main():
    t_start = time.time()
    train_d7d_folds()
    oof_df = aggregate_d7d_oof()
    eval_res = evaluate_d7d(oof_df)
    print(f"\n[D7-D FINISHED in {(time.time() - t_start) / 60:.1f} minutes]")


if __name__ == "__main__":
    main()
