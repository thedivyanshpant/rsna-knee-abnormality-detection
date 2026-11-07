#!/usr/bin/env python3
"""
Phase D5: D3 ViT Inference-Time Calibration & Test-Time Augmentation (TTA) Runner.
RSNA Knee Abnormality Detection

Strictly INFERENCE-ONLY evaluation on frozen Phase D3 ViT checkpoints:
- D5-A: Standard D3 ViT inference (Baseline reproduction)
- D5-B: Horizontal-flip TTA (Arithmetic probability mean: 0.5*P_std + 0.5*P_flip)
- D5-C: Within-fold percentile rank normalization on standard predictions
- D5-D: Primary D1.2-B2 replication: Within-fold rank normalization of probability-averaged TTA
- D5-D_alt: Within-fold equal rank average: 0.5*Rank(P_std) + 0.5*Rank(P_flip)

Dual Ground-Truth Evaluation:
1. Phase D3 Candidate Remediated Labels (phase_d/d3/labels/phase_d3_candidate_label_corpus.csv)
2. Phase B Legacy Production Labels (labels/production/phase_b_step3_remediated_training_corpus.csv)

Governance:
- Strict Gold-58 zero leakage assertion.
- No model weights modified.
- No retraining or fine-tuning.
- No label modifications.
- Frozen CV splits preserved.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Any
from multiprocessing import Process, Queue
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase_d.d1.scripts.model_d1 import PhaseD1Model, PATHOLOGY_NAMES
from phase_d.d1.scripts.dataset_d1 import PhaseD1KneeDataset, TARGETS
from phase_d.d1.scripts.d1_2_eval_utils import (
    roc_auc_score_np,
    pr_auc_score_np,
    evaluate_oof_df,
)

CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
D3_LABELS_CSV = os.path.join(PROJECT_ROOT, "phase_d/d3/labels/phase_d3_candidate_label_corpus.csv")
PROD_LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
SLOT_MANIFEST_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/manifests/d1_slot_mapping_manifest.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "phase_c/c1/processed")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d3/checkpoints")
GOLD_CSV = os.path.join(PROJECT_ROOT, "csv/gold_studies.csv")
TEMPLATE_OOF_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/results/oof/d1_1_global_oof_predictions.csv")

D5_BASE_DIR = os.path.join(PROJECT_ROOT, "phase_d/d5")
PRED_DIR = os.path.join(D5_BASE_DIR, "predictions")
FOLD_PRED_DIR = os.path.join(PRED_DIR, "folds")
EVAL_DIR = os.path.join(D5_BASE_DIR, "evaluations")
REPORTS_DIR = os.path.join(D5_BASE_DIR, "reports")


def rank_within_fold(prob_matrix: np.ndarray) -> np.ndarray:
    """Percentile rank normalization [0, 1] per column within a fold."""
    N = prob_matrix.shape[0]
    ranks = np.zeros_like(prob_matrix)
    for c in range(prob_matrix.shape[1]):
        ranks[:, c] = rankdata(prob_matrix[:, c], method="average") / float(N)
    return ranks


def run_inference_worker(folds: List[int], gpu_id: int, out_queue: Queue):
    """Worker running standard and horizontal flip inference for specified folds on given GPU."""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[Worker GPU {gpu_id}] Starting folds {folds} on {device}...", flush=True)

    with open(CONFIG_CV) as f:
        cv_data = json.load(f)

    slot_manifest_df = pd.read_csv(SLOT_MANIFEST_CSV)
    d3_labels_df = pd.read_csv(D3_LABELS_CSV)
    gold_df = pd.read_csv(GOLD_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())

    for fold in folds:
        t0 = time.time()
        val_studies = cv_data["folds"][f"fold_{fold}"]["val_studies"]
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"fold_{fold}", "best_model.pt")

        assert len(set(val_studies) & gold_uids) == 0, f"CRITICAL LEAKAGE: Gold-58 study in Fold {fold} validation set!"
        assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"

        dataset = PhaseD1KneeDataset(
            study_uids=val_studies,
            slot_manifest_df=slot_manifest_df,
            labels_df=d3_labels_df,
            processed_dir=PROCESSED_DIR,
            gold_uids=gold_uids,
            n_groups=12,
            slice_band=(0.2, 0.8),
        )

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model = PhaseD1Model().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        uids = []
        p_std_list = []
        p_flip_list = []
        targets_list = []
        weights_list = []

        with torch.no_grad():
            for batch in loader:
                images = batch["images"].to(device, non_blocking=True)
                slot_mask = batch["slot_mask"].to(device, non_blocking=True)
                uids.extend(batch["study_uid"])
                targets_list.append(batch["targets"].numpy())
                weights_list.append(batch["weights"].numpy())

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # 1. Standard prediction
                    logits_std, _ = model(images, slot_mask)
                    probs_std = torch.sigmoid(logits_std).float().cpu().numpy()

                    # 2. Horizontal flip along width axis (axis -1)
                    images_flip = torch.flip(images, dims=[-1])
                    logits_flip, _ = model(images_flip, slot_mask)
                    probs_flip = torch.sigmoid(logits_flip).float().cpu().numpy()

                p_std_list.append(probs_std)
                p_flip_list.append(probs_flip)

        p_std = np.concatenate(p_std_list, axis=0)
        p_flip = np.concatenate(p_flip_list, axis=0)
        y_true = np.concatenate(targets_list, axis=0)
        y_weights = np.concatenate(weights_list, axis=0)

        # Sanity check against existing D3 validation predictions if available
        d3_existing_val_path = os.path.join(PROJECT_ROOT, "phase_d/d3/predictions/val", f"fold_{fold}_val_predictions.csv")
        if os.path.exists(d3_existing_val_path):
            existing_df = pd.read_csv(d3_existing_val_path).set_index("StudyInstanceUID").reindex(uids)
            existing_p = existing_df[TARGETS].values
            max_diff = np.max(np.abs(p_std - existing_p))
            mean_diff = np.mean(np.abs(p_std - existing_p))
            print(f"[Worker GPU {gpu_id}] Fold {fold} reproduction diff vs saved: max={max_diff:.6f}, mean={mean_diff:.6f}", flush=True)

        # Save per-fold predictions
        os.makedirs(FOLD_PRED_DIR, exist_ok=True)
        fold_save_path = os.path.join(FOLD_PRED_DIR, f"fold_{fold}_d5_predictions.npz")
        np.savez_compressed(
            fold_save_path,
            uids=np.array(uids),
            p_std=p_std,
            p_flip=p_flip,
            y_true=y_true,
            y_weights=y_weights,
            fold=fold,
        )

        elapsed = time.time() - t0
        print(f"[Worker GPU {gpu_id}] Fold {fold} Complete: {len(uids)} studies in {elapsed:.1f}s (saved {fold_save_path})", flush=True)

        out_queue.put((fold, uids, p_std, p_flip, y_true, y_weights))


def evaluate_dual_standards(
    df_oof: pd.DataFrame,
    experiment_id: str,
    uids_order: List[str],
    d3_labels_df: pd.DataFrame,
    prod_labels_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate an OOF dataframe against both Remediated D3 labels and Legacy Phase B labels."""
    # 1. Remediated Evaluation
    df_d3 = df_oof.copy()
    for t in PATHOLOGY_NAMES:
        sub_l = d3_labels_df[d3_labels_df["target"] == t].set_index("StudyInstanceUID").reindex(uids_order)
        df_d3[f"true_{t}"] = sub_l["label_probability"].values
        df_d3[f"weight_{t}"] = sub_l["sample_weight"].values

    eval_d3 = evaluate_oof_df(df_d3, experiment_id=f"{experiment_id}_REMEDIATED")

    # 2. Legacy Production Evaluation
    df_prod = df_oof.copy()
    for t in PATHOLOGY_NAMES:
        sub_p = prod_labels_df[prod_labels_df["target"] == t].set_index("StudyInstanceUID").reindex(uids_order)
        df_prod[f"true_{t}"] = sub_p["label_probability"].values
        df_prod[f"weight_{t}"] = sub_p["sample_weight"].values

    eval_prod = evaluate_oof_df(df_prod, experiment_id=f"{experiment_id}_LEGACY")

    return {
        "remediated": eval_d3,
        "legacy": eval_prod,
    }


def main():
    start_time = time.time()
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(FOLD_PRED_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 80)
    print("  PHASE D5: D3 ViT INFERENCE-TIME CALIBRATION & TTA STUDY")
    print("=" * 80)
    print("Execution mode: 2-GPU parallel inference across 5 folds")
    print("GPU 0: Folds 0, 1 | GPU 1: Folds 2, 3, 4")
    print("Governance: Gold-58 permanently locked. Zero Gold touches.")
    sys.stdout.flush()

    out_queue = Queue()
    p0 = Process(target=run_inference_worker, args=([0, 1], 0, out_queue))
    p1 = Process(target=run_inference_worker, args=([2, 3, 4], 1, out_queue))

    p0.start()
    p1.start()

    fold_results = {}
    for _ in range(5):
        fold, uids, p_std, p_flip, y_true, y_weights = out_queue.get()
        fold_results[fold] = {
            "uids": uids,
            "p_std": p_std,
            "p_flip": p_flip,
            "y_true": y_true,
            "y_weights": y_weights,
        }

    p0.join()
    p1.join()

    inference_runtime = time.time() - start_time
    print(f"\nInference completed across all 5 folds in {inference_runtime:.1f}s!", flush=True)

    # Load canonical order
    template_df = pd.read_csv(TEMPLATE_OOF_CSV)
    uids_order = template_df["StudyInstanceUID"].tolist()
    d3_labels_df = pd.read_csv(D3_LABELS_CSV)
    prod_labels_df = pd.read_csv(PROD_LABELS_CSV)

    # Assemble per-fold matrices and compute transformations
    all_uids = []
    all_folds = []
    p_std_all = []
    p_flip_all = []
    p_b1_all = []       # 0.5*std + 0.5*flip
    p_c1_all = []       # within-fold rank(std)
    p_b2_all = []       # within-fold rank(0.5*std + 0.5*flip)
    p_b2_alt_all = []   # 0.5*rank(std) + 0.5*rank(flip)

    for fold in range(5):
        res = fold_results[fold]
        uids_f = res["uids"]
        std_f = res["p_std"]
        flip_f = res["p_flip"]

        # Probability-space TTA
        b1_f = 0.5 * (std_f + flip_f)

        # Within-fold percentile rank normalization
        c1_f = rank_within_fold(std_f)
        b2_f = rank_within_fold(b1_f)
        b2_alt_f = 0.5 * (rank_within_fold(std_f) + rank_within_fold(flip_f))

        all_uids.extend(uids_f)
        all_folds.extend([fold] * len(uids_f))
        p_std_all.append(std_f)
        p_flip_all.append(flip_f)
        p_b1_all.append(b1_f)
        p_c1_all.append(c1_f)
        p_b2_all.append(b2_f)
        p_b2_alt_all.append(b2_alt_f)

    # Concatenate all folds
    raw_df = pd.DataFrame({"StudyInstanceUID": all_uids, "fold": all_folds})
    p_std_mat = np.concatenate(p_std_all, axis=0)
    p_flip_mat = np.concatenate(p_flip_all, axis=0)
    p_b1_mat = np.concatenate(p_b1_all, axis=0)
    p_c1_mat = np.concatenate(p_c1_all, axis=0)
    p_b2_mat = np.concatenate(p_b2_all, axis=0)
    p_b2_alt_mat = np.concatenate(p_b2_alt_all, axis=0)

    # Reindex to canonical uids_order
    raw_df["order_idx"] = raw_df["StudyInstanceUID"].map(lambda x: uids_order.index(x))
    sort_order = np.argsort(raw_df["order_idx"].values)

    sorted_uids = [all_uids[i] for i in sort_order]
    sorted_folds = [all_folds[i] for i in sort_order]
    assert sorted_uids == uids_order, "Ordering mismatch!"

    p_std_sorted = p_std_mat[sort_order]
    p_flip_sorted = p_flip_mat[sort_order]
    p_b1_sorted = p_b1_mat[sort_order]
    p_c1_sorted = p_c1_mat[sort_order]
    p_b2_sorted = p_b2_mat[sort_order]
    p_b2_alt_sorted = p_b2_alt_mat[sort_order]

    # Build standardized prediction DataFrames
    def make_pred_df(prob_mat: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"StudyInstanceUID": sorted_uids, "fold": sorted_folds})
        for idx, t in enumerate(PATHOLOGY_NAMES):
            df[f"pred_{t}"] = prob_mat[:, idx]
        return df

    df_d5_a = make_pred_df(p_std_sorted)      # Standard D3
    df_d5_b = make_pred_df(p_b1_sorted)       # Flip TTA (prob mean)
    df_d5_c = make_pred_df(p_c1_sorted)       # Standard + RankNorm
    df_d5_d = make_pred_df(p_b2_sorted)       # Primary D5-B2 (Flip TTA + RankNorm)
    df_d5_d_alt = make_pred_df(p_b2_alt_sorted) # D5-B2 alt (Equal Rank Average)
    df_d5_flip_alone = make_pred_df(p_flip_sorted) # Diagnostic flip alone

    # Save prediction CSVs
    p_a_path = os.path.join(PRED_DIR, "d5_standard_oof_predictions.csv")
    p_b_path = os.path.join(PRED_DIR, "d5_flip_tta_oof_predictions.csv")
    p_c_path = os.path.join(PRED_DIR, "d5_ranknorm_oof_predictions.csv")
    p_d_path = os.path.join(PRED_DIR, "d5_b2_oof_predictions.csv")
    p_d_alt_path = os.path.join(PRED_DIR, "d5_b2_alt_rank_avg_oof_predictions.csv")
    p_flip_path = os.path.join(PRED_DIR, "d5_diagnostic_flip_alone_oof_predictions.csv")

    df_d5_a.to_csv(p_a_path, index=False)
    df_d5_b.to_csv(p_b_path, index=False)
    df_d5_c.to_csv(p_c_path, index=False)
    df_d5_d.to_csv(p_d_path, index=False)
    df_d5_d_alt.to_csv(p_d_alt_path, index=False)
    df_d5_flip_alone.to_csv(p_flip_path, index=False)

    print("\nSaved all OOF prediction files:", flush=True)
    print(f"  D5-A Standard:          {p_a_path}")
    print(f"  D5-B Flip TTA:          {p_b_path}")
    print(f"  D5-C RankNorm:          {p_c_path}")
    print(f"  D5-D (Primary D5-B2):   {p_d_path}")
    print(f"  D5-D_alt (Rank Avg):    {p_d_alt_path}")
    print(f"  Diagnostic Flip Alone:  {p_flip_path}")

    # Evaluate all variants against both ground truths
    experiments = {
        "D5-A_STANDARD": df_d5_a,
        "D5-B_FLIP_TTA": df_d5_b,
        "D5-C_RANKNORM": df_d5_c,
        "D5-D_B2_PRIMARY": df_d5_d,
        "D5-D_B2_ALT_RANK_AVG": df_d5_d_alt,
        "D5-DIAG_FLIP_ALONE": df_d5_flip_alone,
    }

    eval_summaries = {}
    for exp_id, df_exp in experiments.items():
        eval_summaries[exp_id] = evaluate_dual_standards(
            df_exp, exp_id, uids_order, d3_labels_df, prod_labels_df
        )

    # Save evaluation summary JSON
    eval_json_path = os.path.join(EVAL_DIR, "d5_oof_evaluation_summary.json")
    with open(eval_json_path, "w") as f:
        json.dump(eval_summaries, f, indent=2)
    print(f"\nSaved evaluation summary JSON to: {eval_json_path}", flush=True)

    # Build per-target metrics table for Primary D5-B2 vs D3 Baseline
    target_rows = []
    d3_rem_pt = eval_summaries["D5-A_STANDARD"]["remediated"]["per_target"]
    d5_b2_rem_pt = eval_summaries["D5-D_B2_PRIMARY"]["remediated"]["per_target"]
    d5_b1_rem_pt = eval_summaries["D5-B_FLIP_TTA"]["remediated"]["per_target"]
    d5_c1_rem_pt = eval_summaries["D5-C_RANKNORM"]["remediated"]["per_target"]

    for t in PATHOLOGY_NAMES:
        d3_auc = d3_rem_pt[t]["roc_auc_unweighted"]
        d3_pr = d3_rem_pt[t]["pr_auc"]
        b2_auc = d5_b2_rem_pt[t]["roc_auc_unweighted"]
        b2_pr = d5_b2_rem_pt[t]["pr_auc"]
        b1_auc = d5_b1_rem_pt[t]["roc_auc_unweighted"]
        b1_pr = d5_b1_rem_pt[t]["pr_auc"]
        c1_auc = d5_c1_rem_pt[t]["roc_auc_unweighted"]
        c1_pr = d5_c1_rem_pt[t]["pr_auc"]

        target_rows.append({
            "Target": t,
            "Positive_Count": d3_rem_pt[t]["pos_count"],
            "D3_Baseline_AUC": d3_auc,
            "D5_B1_FlipTTA_AUC": b1_auc,
            "D5_C1_RankNorm_AUC": c1_auc,
            "D5_B2_Primary_AUC": b2_auc,
            "Delta_B2_vs_D3_AUC": round(b2_auc - d3_auc, 4) if (b2_auc and d3_auc) else None,
            "D3_Baseline_PRAUC": d3_pr,
            "D5_B1_FlipTTA_PRAUC": b1_pr,
            "D5_C1_RankNorm_PRAUC": c1_pr,
            "D5_B2_Primary_PRAUC": b2_pr,
            "Delta_B2_vs_D3_PRAUC": round(b2_pr - d3_pr, 4) if (b2_pr and d3_pr) else None,
        })

    target_df = pd.DataFrame(target_rows)
    target_csv_path = os.path.join(EVAL_DIR, "d5_per_target_metrics.csv")
    target_df.to_csv(target_csv_path, index=False)
    print(f"Saved per-target metrics CSV to: {target_csv_path}", flush=True)

    # Print Authoritative Evaluation Matrix
    print("\n" + "=" * 115)
    print("                              PHASE D5 AUTHORITATIVE EVALUATION MATRIX")
    print("=" * 115)
    hdr = f"{'Experiment ID':<22} | {'Rem AUC':<8} | {'Rem Wtd':<8} | {'Rem PR':<8} | {'Leg AUC':<8} | {'Leg Wtd':<8} | {'Leg PR':<8} | {'Δ vs D3 Rem':<11} | {'Δ vs D3 Leg':<11}"
    print(hdr)
    print("-" * 115)

    d3_rem_auc = eval_summaries["D5-A_STANDARD"]["remediated"]["global_macro_roc_auc_unweighted"]
    d3_leg_auc = eval_summaries["D5-A_STANDARD"]["legacy"]["global_macro_roc_auc_unweighted"]

    for exp_id in ["D5-A_STANDARD", "D5-B_FLIP_TTA", "D5-C_RANKNORM", "D5-D_B2_PRIMARY", "D5-D_B2_ALT_RANK_AVG", "D5-DIAG_FLIP_ALONE"]:
        r_eval = eval_summaries[exp_id]["remediated"]
        l_eval = eval_summaries[exp_id]["legacy"]
        r_auc = r_eval["global_macro_roc_auc_unweighted"]
        r_wtd = r_eval["global_macro_roc_auc_weighted"]
        r_pr = r_eval["global_macro_pr_auc"]
        l_auc = l_eval["global_macro_roc_auc_unweighted"]
        l_wtd = l_eval["global_macro_roc_auc_weighted"]
        l_pr = l_eval["global_macro_pr_auc"]
        delta_rem = r_auc - d3_rem_auc
        delta_leg = l_auc - d3_leg_auc
        print(f"{exp_id:<22} | {r_auc:<8.4f} | {r_wtd:<8.4f} | {r_pr:<8.4f} | {l_auc:<8.4f} | {l_wtd:<8.4f} | {l_pr:<8.4f} | {delta_rem:+11.4f} | {delta_leg:+11.4f}")

    print("=" * 115)

    # Print Per-Target Table
    print("\n" + "=" * 100)
    print("               PHASE D5 PRIMARY D5-B2 PER-TARGET BREAKDOWN (REMEDIATED GT)")
    print("=" * 100)
    pt_hdr = f"{'Target':<18} | {'Pos':<4} | {'D3 AUC':<7} | {'D5-B2 AUC':<9} | {'Δ AUC':<7} | {'D3 PR':<7} | {'D5-B2 PR':<9} | {'Δ PR':<7}"
    print(pt_hdr)
    print("-" * 100)
    for r in target_rows:
        d_auc = f"{r['Delta_B2_vs_D3_AUC']:+.4f}" if r['Delta_B2_vs_D3_AUC'] is not None else "N/A"
        d_pr = f"{r['Delta_B2_vs_D3_PRAUC']:+.4f}" if r['Delta_B2_vs_D3_PRAUC'] is not None else "N/A"
        print(f"{r['Target']:<18} | {r['Positive_Count']:<4} | {r['D3_Baseline_AUC']:<7.4f} | {r['D5_B2_Primary_AUC']:<9.4f} | {d_auc:<7} | {r['D3_Baseline_PRAUC']:<7.4f} | {r['D5_B2_Primary_PRAUC']:<9.4f} | {d_pr:<7}")
    print("=" * 100)

    total_time = time.time() - start_time
    print(f"\nPhase D5 inference & evaluation complete in {total_time:.1f}s ({total_time/60.0:.2f} min).", flush=True)


if __name__ == "__main__":
    main()
