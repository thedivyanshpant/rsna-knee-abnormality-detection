#!/usr/bin/env python3
"""
Phase D4 Master Orchestrator, Multi-Node 5-Fold CV Runner, OOF Aggregator & Dual-Evaluation Engine.
RSNA Knee Abnormality Detection

Orchestrates:
1. Multi-node DDP training of RadImageNet ResNet-50 on Phase D3 Remediated Labels across Folds 0..4 on 3 nodes (6 x A100 GPUs).
2. Global 4,349-study OOF aggregation, verification, and metric calculation.
3. Dual-ground-truth evaluation:
   - Evaluates OOF against Remediated D3 Labels (primary training standard)
   - Evaluates OOF against Legacy Phase B Production Labels (benchmark continuity)
4. Controlled Multi-Arm Ensembles (D4-C1 = 75/25 Rank Blend, D4-B1 = 50/50 Rank Blend) with frozen D3 ViT.
5. Emits complete structured metrics JSON, ensemble summary, and cryptographic manifest.
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
D3_LABELS_CSV = os.path.join(PROJECT_ROOT, "phase_d/d3/labels/phase_d3_candidate_label_corpus.csv")
PROD_LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
GOLD_CSV = os.path.join(PROJECT_ROOT, "csv/gold_studies.csv")

D4_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d4/scripts")
D4_CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d4/checkpoints")
D4_VAL_PRED_DIR = os.path.join(PROJECT_ROOT, "phase_d/d4/predictions/val")
D4_OOF_DIR = os.path.join(PROJECT_ROOT, "phase_d/d4/predictions/oof")
D4_BLENDS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d4/predictions/blends")
D4_REPORTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d4/reports")
D4_MANIFESTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d4/manifests")

D1_TEMPLATE_OOF_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/results/oof/d1_1_global_oof_predictions.csv")
D3_VIT_OOF_CSV = os.path.join(PROJECT_ROOT, "phase_d/d3/predictions/d3_global_oof_predictions.csv")

MASTER_ADDR = "172.5.3.7"
WORKER_NODES = ["rpgpu018", "rpgpu028"]


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def train_fold_multinode(fold: int, epochs: int = 10, batch_size: int = 1, grad_accum: int = 1):
    """Launch 6-process multi-node DDP training across rpgpu007, rpgpu018, rpgpu028."""
    print("\n" + "=" * 80, flush=True)
    print(f"  [ORCHESTRATOR D4] Launching Fold {fold} across 3 nodes (6 x A100 GPUs)", flush=True)
    print("=" * 80, flush=True)

    master_port = str(29530 + fold)
    import shutil
    torchrun_bin = os.environ.get("TORCHRUN_BIN", shutil.which("torchrun") or "torchrun")
    train_script = os.path.join(D4_SCRIPTS_DIR, "train_ddp_d4.py")

    t0 = time.time()
    worker_procs = []

    # Launch worker nodes via SSH
    for node_idx, node_host in enumerate(WORKER_NODES, start=1):
        ssh_cmd = (
            f'ssh -o StrictHostKeyChecking=no {node_host} '
            f'"NCCL_SOCKET_IFNAME=eth2 GLOO_SOCKET_IFNAME=eth2 {torchrun_bin} '
            f'--nproc_per_node=2 --nnodes=3 --node_rank={node_idx} '
            f'--master_addr={MASTER_ADDR} --master_port={master_port} '
            f'{train_script} --fold={fold} --epochs={epochs} '
            f'--batch_size={batch_size} --grad_accum={grad_accum}"'
        )
        print(f"[LAUNCH] Worker node {node_host} (rank {node_idx})...", flush=True)
        p = subprocess.Popen(ssh_cmd, shell=True)
        worker_procs.append((node_host, p))

    # Launch master node locally (rank 0)
    master_env = os.environ.copy()
    master_env["NCCL_SOCKET_IFNAME"] = "eth2"
    master_env["GLOO_SOCKET_IFNAME"] = "eth2"

    master_cmd = [
        torchrun_bin,
        "--nproc_per_node=2",
        "--nnodes=3",
        "--node_rank=0",
        f"--master_addr={MASTER_ADDR}",
        f"--master_port={master_port}",
        train_script,
        f"--fold={fold}",
        f"--epochs={epochs}",
        f"--batch_size={batch_size}",
        f"--grad_accum={grad_accum}",
    ]

    print(f"[LAUNCH] Master node rpgpu007 (rank 0)...", flush=True)
    res_master = subprocess.run(master_cmd, env=master_env)

    # Wait for all worker processes to complete
    worker_failed = False
    for node_host, p in worker_procs:
        ret = p.wait()
        if ret != 0:
            print(f"[ERROR] Worker {node_host} exited with code {ret}", flush=True)
            worker_failed = True

    elapsed = time.time() - t0
    if res_master.returncode != 0 or worker_failed:
        print(f"[ERROR] Fold {fold} training failed in {elapsed:.1f}s!", flush=True)
        sys.exit(1)

    print(f"[SUCCESS] Fold {fold} training completed successfully in {elapsed:.1f}s", flush=True)


def train_all_folds(epochs: int = 10, batch_size: int = 1, grad_accum: int = 1):
    """Train Folds 0 through 4 sequentially across the 3 nodes."""
    for fold in range(5):
        val_csv = os.path.join(D4_VAL_PRED_DIR, f"fold_{fold}_val_predictions.csv")
        ckpt_pt = os.path.join(D4_CHECKPOINTS_DIR, f"fold_{fold}", "best_model.pt")
        if os.path.exists(val_csv) and os.path.exists(ckpt_pt):
            print(f"[SKIP] Fold {fold} already completed. Checkpoint and predictions found.", flush=True)
            continue
        train_fold_multinode(fold, epochs=epochs, batch_size=batch_size, grad_accum=grad_accum)


def aggregate_d4_oof() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate 5-fold validation predictions into standardized global 4,349-study OOF format and dual-evaluate."""
    print("\n" + "=" * 80, flush=True)
    print("  AGGREGATING PHASE D4 5-FOLD OUT-OF-FOLD (OOF) PREDICTIONS", flush=True)
    print("=" * 80, flush=True)

    template_df = pd.read_csv(D1_TEMPLATE_OOF_CSV)
    uids_order = template_df["StudyInstanceUID"].tolist()

    val_dfs = []
    for fold in range(5):
        p_path = os.path.join(D4_VAL_PRED_DIR, f"fold_{fold}_val_predictions.csv")
        assert os.path.exists(p_path), f"Missing validation predictions for Fold {fold}: {p_path}"
        df_k = pd.read_csv(p_path)
        val_dfs.append(df_k)
        print(f"Fold {fold}: loaded {len(df_k)} studies", flush=True)

    raw_oof_df = pd.concat(val_dfs, axis=0).set_index("StudyInstanceUID").reindex(uids_order).reset_index()

    # Build standardized OOF dataframe
    d4_oof_df = template_df.copy()
    for t in PATHOLOGY_NAMES:
        d4_oof_df[f"pred_{t}"] = raw_oof_df[t].values

    # Output file: d4_cnn_oof_predictions.csv
    oof_csv_path = os.path.join(D4_OOF_DIR, "d4_cnn_oof_predictions.csv")
    d4_oof_df.to_csv(oof_csv_path, index=False)
    print(f"[SAVED] D4 CNN Global OOF predictions to: {oof_csv_path}", flush=True)

    # 1. Evaluate against Phase D3 Remediated Ground Truth
    print("\n--- Evaluating D4 CNN against D3 Remediated Labels ---", flush=True)
    d3_labels_df = pd.read_csv(D3_LABELS_CSV)
    d4_eval_d3_df = d4_oof_df.copy()
    for t in PATHOLOGY_NAMES:
        sub_l = d3_labels_df[d3_labels_df["target"] == t].set_index("StudyInstanceUID").reindex(uids_order)
        d4_eval_d3_df[f"true_{t}"] = sub_l["label_probability"].values
        d4_eval_d3_df[f"weight_{t}"] = sub_l["sample_weight"].values

    d3_results = evaluate_oof_df(d4_eval_d3_df, experiment_id="D4_CNN_REMEDIATED_LABELS_EVALUATION")

    # 2. Evaluate against Legacy Phase B Production Ground Truth
    print("\n--- Evaluating D4 CNN against Legacy Phase B Production Labels ---", flush=True)
    prod_labels_df = pd.read_csv(PROD_LABELS_CSV)
    d4_eval_prod_df = d4_oof_df.copy()
    for t in PATHOLOGY_NAMES:
        sub_l = prod_labels_df[prod_labels_df["target"] == t].set_index("StudyInstanceUID").reindex(uids_order)
        d4_eval_prod_df[f"true_{t}"] = sub_l["label_probability"].values
        d4_eval_prod_df[f"weight_{t}"] = sub_l["sample_weight"].values

    prod_results = evaluate_oof_df(d4_eval_prod_df, experiment_id="D4_CNN_LEGACY_PRODUCTION_EVALUATION")

    combined_summary = {
        "experiment_id": "PHASE_D4_RADIMAGENET_CNN_REMEDIATED_SUPERVISION",
        "d3_remediated_evaluation": d3_results,
        "legacy_production_evaluation": prod_results,
        "delta_remediated_vs_d3_vit": d3_results["global_macro_roc_auc_unweighted"] - 0.8135,
        "delta_legacy_vs_d1_2": prod_results["global_macro_roc_auc_unweighted"] - 0.7592,
        "delta_legacy_vs_d2_cnn": prod_results["global_macro_roc_auc_unweighted"] - 0.7287,
    }

    summary_json_path = os.path.join(D4_REPORTS_DIR, "d4_oof_evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(combined_summary, f, indent=2)
    print(f"\n[SAVED] D4 CNN Evaluation Summary to: {summary_json_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  PHASE D4 CNN FINAL 5-FOLD EVALUATION SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Against D3 Remediated Ground Truth: Unw Macro AUC = {d3_results['global_macro_roc_auc_unweighted']:.4f} | Wtd AUC = {d3_results['global_macro_roc_auc_weighted']:.4f} | PR-AUC = {d3_results['global_macro_pr_auc']:.4f}")
    print(f"Against Legacy Production Labels:   Unw Macro AUC = {prod_results['global_macro_roc_auc_unweighted']:.4f} | Wtd AUC = {prod_results['global_macro_roc_auc_weighted']:.4f} | PR-AUC = {prod_results['global_macro_pr_auc']:.4f}")
    print("=" * 80, flush=True)

    return d4_oof_df, combined_summary


def evaluate_d4_ensembles(d4_cnn_oof_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate controlled multi-arm ensembles combining frozen D3 ViT and D4 CNN."""
    print("\n" + "=" * 80, flush=True)
    print("  EVALUATING CONTROLLED MULTI-ARM ENSEMBLES: FROZEN D3 ViT + D4 CNN", flush=True)
    print("=" * 80, flush=True)

    assert os.path.exists(D3_VIT_OOF_CSV), f"Missing frozen D3 ViT OOF: {D3_VIT_OOF_CSV}"
    d3_vit_df = pd.read_csv(D3_VIT_OOF_CSV)
    uids = d3_vit_df["StudyInstanceUID"].tolist()
    d4_aligned = d4_cnn_oof_df.set_index("StudyInstanceUID").reindex(uids).reset_index()

    with open(CONFIG_CV) as f:
        cv_splits = json.load(f)

    d3_labels_df = pd.read_csv(D3_LABELS_CSV)
    prod_labels_df = pd.read_csv(PROD_LABELS_CSV)

    ensemble_specs = [
        ("D4_C1_WEIGHT_75_25", "75% D3 ViT + 25% D4 CNN (Within-Fold Rank Blend)", 0.75, 0.25),
        ("D4_B1_WEIGHT_50_50", "50% D3 ViT + 50% D4 CNN (Within-Fold Rank Blend)", 0.50, 0.50),
    ]

    ensemble_results = {}

    for blend_id, desc, w_vit, w_cnn in ensemble_specs:
        print(f"\n--- Constructing: {blend_id} ({desc}) ---", flush=True)
        blend_df = d3_vit_df.copy()

        # Compute within-fold rank blend per fold
        for fold in range(5):
            val_uids = cv_splits["folds"][f"fold_{fold}"]["val_studies"]
            mask = blend_df["StudyInstanceUID"].isin(val_uids)
            idx = blend_df[mask].index

            p_vit_fold = d3_vit_df.loc[idx, [f"pred_{t}" for t in PATHOLOGY_NAMES]].values
            p_cnn_fold = d4_aligned.loc[idx, [f"pred_{t}" for t in PATHOLOGY_NAMES]].values

            N_fold = len(val_uids)
            r_vit_fold = np.zeros_like(p_vit_fold)
            r_cnn_fold = np.zeros_like(p_cnn_fold)
            for c in range(len(PATHOLOGY_NAMES)):
                r_vit_fold[:, c] = rankdata(p_vit_fold[:, c], method="average") / N_fold
                r_cnn_fold[:, c] = rankdata(p_cnn_fold[:, c], method="average") / N_fold

            p_blend = w_vit * r_vit_fold + w_cnn * r_cnn_fold
            for c_idx, t in enumerate(PATHOLOGY_NAMES):
                blend_df.loc[idx, f"pred_{t}"] = p_blend[:, c_idx]

        # Save prediction table for primary ensemble D4-C1
        if blend_id == "D4_C1_WEIGHT_75_25":
            out_csv = os.path.join(D4_BLENDS_DIR, "d4_ensemble_predictions.csv")
            blend_df.to_csv(out_csv, index=False)
            print(f"[SAVED] D4 Primary Ensemble predictions to: {out_csv}", flush=True)

        # 1. Evaluate against D3 Remediated Ground Truth
        blend_eval_d3 = blend_df.copy()
        for t in PATHOLOGY_NAMES:
            sub_l = d3_labels_df[d3_labels_df["target"] == t].set_index("StudyInstanceUID").reindex(uids)
            blend_eval_d3[f"true_{t}"] = sub_l["label_probability"].values
            blend_eval_d3[f"weight_{t}"] = sub_l["sample_weight"].values
        res_d3 = evaluate_oof_df(blend_eval_d3, experiment_id=f"{blend_id}_REMEDIATED")

        # 2. Evaluate against Legacy Phase B Production Ground Truth
        blend_eval_prod = blend_df.copy()
        for t in PATHOLOGY_NAMES:
            sub_l = prod_labels_df[prod_labels_df["target"] == t].set_index("StudyInstanceUID").reindex(uids)
            blend_eval_prod[f"true_{t}"] = sub_l["label_probability"].values
            blend_eval_prod[f"weight_{t}"] = sub_l["sample_weight"].values
        res_prod = evaluate_oof_df(blend_eval_prod, experiment_id=f"{blend_id}_LEGACY")

        ensemble_results[blend_id] = {
            "description": desc,
            "weights": {"vit": w_vit, "cnn": w_cnn},
            "remediated_evaluation": res_d3,
            "legacy_evaluation": res_prod,
            "delta_remediated_vs_d3_vit": res_d3["global_macro_roc_auc_unweighted"] - 0.8135,
            "delta_legacy_vs_d3_vit": res_prod["global_macro_roc_auc_unweighted"] - 0.8184,
        }

        print(f"[{blend_id}] Remediated GT: Macro AUC = {res_d3['global_macro_roc_auc_unweighted']:.4f} (Wtd: {res_d3['global_macro_roc_auc_weighted']:.4f} | PR: {res_d3['global_macro_pr_auc']:.4f}) | Δ vs D3 ViT: {res_d3['global_macro_roc_auc_unweighted'] - 0.8135:+.4f}", flush=True)
        print(f"[{blend_id}] Legacy GT:     Macro AUC = {res_prod['global_macro_roc_auc_unweighted']:.4f} (Wtd: {res_prod['global_macro_roc_auc_weighted']:.4f} | PR: {res_prod['global_macro_pr_auc']:.4f}) | Δ vs D3 ViT: {res_prod['global_macro_roc_auc_unweighted'] - 0.8184:+.4f}", flush=True)

    summary_json_path = os.path.join(D4_REPORTS_DIR, "d4_ensemble_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(ensemble_results, f, indent=2)
    print(f"\n[SAVED] D4 Ensemble Summary to: {summary_json_path}", flush=True)

    return ensemble_results


def generate_provenance_manifest() -> Dict[str, Any]:
    """Compile cryptographic SHA256 manifest of all D4 inputs and generated artifacts."""
    print("\n" + "=" * 80, flush=True)
    print("  GENERATING PHASE D4 CRYPTOGRAPHIC PROVENANCE MANIFEST", flush=True)
    print("=" * 80, flush=True)

    manifest = {
        "phase": "Phase D4 (Controlled Multi-Arm ViT-CNN Continuation on Remediated Supervision)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "653724"),
        "cluster_nodes": ["rpgpu007", "rpgpu018", "rpgpu028"],
        "gpu_count": 6,
        "input_artifacts": {
            "d3_candidate_label_corpus": {
                "path": D3_LABELS_CSV,
                "sha256": compute_sha256(D3_LABELS_CSV),
            },
            "phase_b_production_corpus": {
                "path": PROD_LABELS_CSV,
                "sha256": compute_sha256(PROD_LABELS_CSV),
            },
            "cv_splits": {
                "path": CONFIG_CV,
                "sha256": compute_sha256(CONFIG_CV),
            },
            "d3_vit_global_oof": {
                "path": D3_VIT_OOF_CSV,
                "sha256": compute_sha256(D3_VIT_OOF_CSV),
            },
        },
        "output_artifacts": {},
    }

    # Record generated checkpoints
    for fold in range(5):
        ckpt_path = os.path.join(D4_CHECKPOINTS_DIR, f"fold_{fold}", "best_model.pt")
        val_path = os.path.join(D4_VAL_PRED_DIR, f"fold_{fold}_val_predictions.csv")
        if os.path.exists(ckpt_path):
            manifest["output_artifacts"][f"checkpoint_fold_{fold}"] = {
                "path": ckpt_path,
                "sha256": compute_sha256(ckpt_path),
                "size_bytes": os.path.getsize(ckpt_path),
            }
        if os.path.exists(val_path):
            manifest["output_artifacts"][f"val_predictions_fold_{fold}"] = {
                "path": val_path,
                "sha256": compute_sha256(val_path),
                "size_bytes": os.path.getsize(val_path),
            }

    # Record OOF predictions
    oof_csv = os.path.join(D4_OOF_DIR, "d4_cnn_oof_predictions.csv")
    if os.path.exists(oof_csv):
        manifest["output_artifacts"]["d4_cnn_oof_predictions"] = {
            "path": oof_csv,
            "sha256": compute_sha256(oof_csv),
            "size_bytes": os.path.getsize(oof_csv),
        }

    ensemble_csv = os.path.join(D4_BLENDS_DIR, "d4_ensemble_predictions.csv")
    if os.path.exists(ensemble_csv):
        manifest["output_artifacts"]["d4_ensemble_predictions"] = {
            "path": ensemble_csv,
            "sha256": compute_sha256(ensemble_csv),
            "size_bytes": os.path.getsize(ensemble_csv),
        }

    # Record summaries
    for r_name in ["d4_oof_evaluation_summary.json", "d4_ensemble_summary.json"]:
        r_path = os.path.join(D4_REPORTS_DIR, r_name)
        if os.path.exists(r_path):
            manifest["output_artifacts"][r_name] = {
                "path": r_path,
                "sha256": compute_sha256(r_path),
                "size_bytes": os.path.getsize(r_path),
            }

    manifest_path = os.path.join(D4_MANIFESTS_DIR, "d4_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SAVED] D4 Cryptographic Manifest to: {manifest_path}", flush=True)

    return manifest


def main():
    print("=" * 80, flush=True)
    print("  PHASE D4 MASTER WORKFLOW: RADIMAGENET RESNET-50 ON D3 SUPERVISION + ENSEMBLE", flush=True)
    print("=" * 80, flush=True)

    train_all_folds(epochs=10, batch_size=1, grad_accum=1)
    d4_oof_df, cnn_summary = aggregate_d4_oof()
    ensemble_summary = evaluate_d4_ensembles(d4_oof_df)
    generate_provenance_manifest()

    print("\n[PHASE D4 COMPLETE] All multi-node training, dual OOF evaluations, and ensembles concluded.", flush=True)


if __name__ == "__main__":
    main()
