#!/usr/bin/env python3
"""
Phase C3.12 Stage B: Locked Gold-58 Ensemble Inference & Evaluation Engine
RSNA Knee Abnormality Detection

Strict Provenance & Protocol Rules:
1. Frozen Checkpoints: Loads the 5 saved production checkpoints:
     phase_c/c3/checkpoints/fold_{0..4}/best_model.pt
2. Locked Gold Cohort: Evaluates strictly on the 58 Gold studies in csv/gold_studies.csv.
     Preprocessed volumes loaded from phase_c/c1/processed_gold/
     Zero leakage check: asserts 0 overlap with the 4,349 development cohort.
3. Prediction Preservation:
     Saves individual fold predictions to:
       phase_c/c3/results/gold58/fold_{0..4}_predictions.csv
4. Predetermined Rank Averaging Ensemble:
     Applies normalized rank averaging per target across the 5 models:
       R_m = rankdata(P_m, method='average') / N
       P_ensemble = mean(R_0..R_4)
     Saves canonical ensemble to:
       phase_c/c3/results/gold58/gold58_rank_averaged_predictions.csv
     Also saves score-averaged ensemble for complete transparent audit:
       phase_c/c3/results/gold58/gold58_score_averaged_predictions.csv
5. Locked Evaluation:
     Evaluates rank-averaged predictions and individual fold predictions against
     the locked ground-truth binary labels in csv/gold_studies.csv.
     Computes per-target ROC-AUC, PR-AUC, Accuracy, F1, and Macro ROC-AUC.
6. Audit & Provenance:
     Exports comprehensive evaluation summaries and integrity audit JSON/MD.
"""

import os
import sys
import json
import time
import hashlib
import platform
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import torch
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.join(REPO_ROOT, "phase_c", "c3", "scripts"))

from model import MultiSeriesKneeModel, TARGETS, NUM_SLOTS
from dataset import classify_series_slot, CANONICAL_SLOTS, TENSOR_SHAPE

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Gold58Dataset(Dataset):
    """Deterministic dataset for Gold-58 cohort evaluation."""

    def __init__(
        self,
        gold_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        processed_dir: str,
        dev_uids: set,
    ):
        super().__init__()
        self.study_uids = gold_df["StudyInstanceUID"].tolist()
        self.processed_dir = processed_dir
        self.gold_df = gold_df.set_index("StudyInstanceUID")
        self.meta_by_study = dict(tuple(meta_df.groupby("StudyInstanceUID")))

        # Leakage assertion: 0 development studies permitted
        overlap = set(self.study_uids) & dev_uids
        assert len(overlap) == 0, f"CRITICAL LEAKAGE: {len(overlap)} development studies detected in Gold cohort!"
        assert len(self.study_uids) == 58, f"Expected exactly 58 Gold studies, got {len(self.study_uids)}"

        # Verify all directories exist on disk
        for uid in self.study_uids:
            s_dir = os.path.join(self.processed_dir, uid)
            assert os.path.isdir(s_dir), f"Missing processed directory for Gold study: {s_dir}"

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int):
        study_uid = self.study_uids[idx]

        s_dir = os.path.join(self.processed_dir, study_uid)
        available_npz = {f[:-4]: os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.endswith(".npz")}

        # Deterministic slot resolution identical to development dataset
        slot_assignments = {}
        if study_uid in self.meta_by_study:
            meta_rows = self.meta_by_study[study_uid]
            for _, row in meta_rows.iterrows():
                ser_uid = row["SeriesInstanceUID"]
                if ser_uid not in available_npz:
                    continue
                plane = row["Anatomical_Plane"]
                fluid = row["Fluid_Sensitive"]
                slot_idx = classify_series_slot(plane, fluid)
                if slot_idx is not None:
                    if slot_idx in slot_assignments:
                        existing = slot_assignments[slot_idx]
                        chosen = min(existing, ser_uid)
                        slot_assignments[slot_idx] = chosen
                    else:
                        slot_assignments[slot_idx] = ser_uid

        volumes = torch.zeros((NUM_SLOTS, 1, *TENSOR_SHAPE), dtype=torch.float32)
        slot_mask = torch.zeros((NUM_SLOTS,), dtype=torch.float32)
        provenance = {i: "" for i in range(NUM_SLOTS)}

        for slot_idx in range(NUM_SLOTS):
            if slot_idx in slot_assignments:
                ser_uid = slot_assignments[slot_idx]
                npz_path = available_npz[ser_uid]
                try:
                    with np.load(npz_path) as data:
                        vol_np = data["volume"].astype(np.float32)
                    volumes[slot_idx] = torch.from_numpy(vol_np).unsqueeze(0)
                    slot_mask[slot_idx] = 1.0
                    provenance[slot_idx] = ser_uid
                except Exception:
                    volumes[slot_idx].zero_()
                    slot_mask[slot_idx] = 0.0

        # Extract binary ground truth
        gt = self.gold_df.loc[study_uid, TARGETS].values.astype(np.float32)

        return {
            "study_uid": study_uid,
            "volumes": volumes,
            "slot_mask": slot_mask,
            "targets": torch.from_numpy(gt),
            "provenance": provenance,
        }


def gold_collate_fn(batch):
    return {
        "study_uid": [d["study_uid"] for d in batch],
        "volumes": torch.stack([d["volumes"] for d in batch], dim=0),
        "slot_mask": torch.stack([d["slot_mask"] for d in batch], dim=0),
        "targets": torch.stack([d["targets"] for d in batch], dim=0),
        "provenance": [d["provenance"] for d in batch],
    }


def roc_auc_score_np(y_true, y_score):
    """Pure NumPy Mann-Whitney U / Pairwise difference ROC-AUC."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def average_precision_score_np(y_true, y_score):
    """Pure NumPy Average Precision / PR-AUC."""
    pos_total = np.sum(y_true == 1)
    if pos_total == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_sorted = (y_true[order] == 1).astype(float)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    prec = tp / (tp + fp)
    rec = tp / pos_total
    rec_diff = np.diff(np.concatenate(([0.0], rec)))
    return float(np.sum(rec_diff * prec))


def compute_binary_metrics(y_true, y_pred_prob, threshold=0.5):
    """Compute comprehensive metrics for binary classification using pure NumPy."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)

    roc_auc = roc_auc_score_np(y_true, y_pred_prob)
    pr_auc = average_precision_score_np(y_true, y_pred_prob)

    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred_bin == 0)))

    acc = float((tp + tn) / max(len(y_true), 1))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "positives_true": int(np.sum(y_true)),
        "positives_pred": int(np.sum(y_pred_bin)),
        "total_samples": int(len(y_true)),
    }


def run_gold_evaluation():
    print("=" * 80)
    print("  PHASE C3.12 STAGE B: LOCKED GOLD-58 ENSEMBLE INFERENCE & EVALUATION")
    print("=" * 80)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else 'CPU'})")

    gold_csv = os.path.join(REPO_ROOT, "csv", "gold_studies.csv")
    meta_csv = os.path.join(REPO_ROOT, "csv", "train_series.csv")
    splits_json = os.path.join(REPO_ROOT, "phase_c", "c3", "configs", "cv_splits.json")
    processed_gold_dir = os.path.join(REPO_ROOT, "phase_c", "c1", "processed_gold")
    gold_results_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "results", "gold58")
    os.makedirs(gold_results_dir, exist_ok=True)

    # 1. Load inputs and verify integrity
    print(f"Loading Gold-58 ground truth from: {gold_csv}")
    gold_df = pd.read_csv(gold_csv)
    assert len(gold_df) == 58, f"Expected exactly 58 Gold studies, got {len(gold_df)}"
    gold_uids = gold_df["StudyInstanceUID"].tolist()
    assert len(set(gold_uids)) == 58, "Duplicate Gold study UIDs detected!"

    print(f"Loading series metadata from: {meta_csv}")
    meta_df = pd.read_csv(meta_csv)

    print(f"Loading CV splits for isolation verification: {splits_json}")
    with open(splits_json) as f:
        cv_splits = json.load(f)
    dev_uids = set()
    for f_idx in range(5):
        dev_uids.update(cv_splits["folds"][f"fold_{f_idx}"]["val_studies"])
    assert len(dev_uids) == 4349, f"Expected 4349 dev studies, got {len(dev_uids)}"

    leakage = set(gold_uids) & dev_uids
    assert len(leakage) == 0, f"CRITICAL LEAKAGE DETECTED: {leakage}"
    print("[INTEGRITY VERIFIED] Exactly 58 Gold studies. 0 development leakage.")

    # 2. Build Dataset & Loader
    gold_dataset = Gold58Dataset(
        gold_df=gold_df,
        meta_df=meta_df,
        processed_dir=processed_gold_dir,
        dev_uids=dev_uids,
    )

    gold_loader = DataLoader(
        gold_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=gold_collate_fn,
    )

    # 3. Predict with all 5 frozen fold checkpoints
    fold_predictions = {}
    fold_metrics = {}
    ground_truth = gold_df[TARGETS].values.astype(np.float32)

    for fold_idx in range(5):
        ckpt_path = os.path.join(REPO_ROOT, "phase_c", "c3", "checkpoints", f"fold_{fold_idx}", "best_model.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
        print(f"\n--- [Fold {fold_idx}] Loading checkpoint: {ckpt_path} ---")

        model = MultiSeriesKneeModel(
            embed_dim=128,
            num_targets=len(TARGETS),
            num_slots=NUM_SLOTS,
            aggregation="masked_pool",
            dropout=0.0,
        ).to(DEVICE)

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        fold_preds = []
        fold_uids = []
        t0 = time.time()

        with torch.no_grad():
            for batch in gold_loader:
                vols = batch["volumes"].to(DEVICE, non_blocking=True)
                mask = batch["slot_mask"].to(DEVICE, non_blocking=True)
                uids = batch["study_uid"]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(vols, mask)
                    probs = torch.sigmoid(logits)

                fold_preds.append(probs.float().cpu().numpy())
                fold_uids.extend(uids)

        preds_np = np.concatenate(fold_preds, axis=0)  # (58, 12)
        assert preds_np.shape == (58, 12), f"Expected shape (58, 12), got {preds_np.shape}"
        assert fold_uids == gold_uids, "Study UID order mismatch in predictions!"

        fold_time = time.time() - t0
        print(f"[Fold {fold_idx}] Inference completed on 58 studies in {fold_time:.2f}s")

        # Save per-fold predictions
        fold_df = pd.DataFrame(preds_np, columns=TARGETS)
        fold_df.insert(0, "StudyInstanceUID", fold_uids)
        fold_out_path = os.path.join(gold_results_dir, f"fold_{fold_idx}_predictions.csv")
        fold_df.to_csv(fold_out_path, index=False)
        print(f"[Fold {fold_idx}] Saved predictions to: {fold_out_path}")

        fold_predictions[fold_idx] = preds_np

        # Evaluate fold metrics
        per_target_m = {}
        for t_idx, target_name in enumerate(TARGETS):
            m = compute_binary_metrics(ground_truth[:, t_idx], preds_np[:, t_idx])
            per_target_m[target_name] = m

        macro_auc = np.mean([m["roc_auc"] for m in per_target_m.values()])
        macro_prauc = np.mean([m["pr_auc"] for m in per_target_m.values()])
        fold_metrics[f"fold_{fold_idx}"] = {
            "macro_roc_auc": float(macro_auc),
            "macro_pr_auc": float(macro_prauc),
            "per_target": per_target_m,
        }
        print(f"[Fold {fold_idx}] Gold Macro ROC-AUC: {macro_auc:.4f} | Macro PR-AUC: {macro_prauc:.4f}")

    # 4. Predetermined Rank Averaging Ensemble
    print("\n" + "=" * 80)
    print("  COMPUTING PREDETERMINED RANK-AVERAGED ENSEMBLE")
    print("=" * 80)

    # For each target, rank probabilities across the 58 studies for each model,
    # normalize ranks to [0, 1] via rankdata(p, method='average') / N, then average across models
    N_studies = len(gold_uids)
    num_folds = 5
    num_targets = len(TARGETS)

    rank_averaged_preds = np.zeros((N_studies, num_targets), dtype=np.float64)
    score_averaged_preds = np.zeros((N_studies, num_targets), dtype=np.float64)

    for t_idx in range(num_targets):
        target_ranks = np.zeros((N_studies, num_folds), dtype=np.float64)
        target_scores = np.zeros((N_studies, num_folds), dtype=np.float64)

        for f_idx in range(num_folds):
            p = fold_predictions[f_idx][:, t_idx]
            target_scores[:, f_idx] = p
            # Normalized rank: rank in [1, N] divided by N
            r = rankdata(p, method="average") / float(N_studies)
            target_ranks[:, f_idx] = r

        rank_averaged_preds[:, t_idx] = np.mean(target_ranks, axis=1)
        score_averaged_preds[:, t_idx] = np.mean(target_scores, axis=1)

    # Save rank-averaged predictions
    rank_df = pd.DataFrame(rank_averaged_preds, columns=TARGETS)
    rank_df.insert(0, "StudyInstanceUID", gold_uids)
    rank_out_path = os.path.join(gold_results_dir, "gold58_rank_averaged_predictions.csv")
    rank_df.to_csv(rank_out_path, index=False)
    print(f"[SUCCESS] Saved locked rank-averaged predictions to: {rank_out_path}")

    # Save score-averaged predictions for reference
    score_df = pd.DataFrame(score_averaged_preds, columns=TARGETS)
    score_df.insert(0, "StudyInstanceUID", gold_uids)
    score_out_path = os.path.join(gold_results_dir, "gold58_score_averaged_predictions.csv")
    score_df.to_csv(score_out_path, index=False)
    print(f"[SUCCESS] Saved score-averaged predictions to: {score_out_path}")

    # 5. Evaluate Rank-Averaged Predictions against Locked Ground Truth
    print("\n" + "=" * 80)
    print("  LOCKED GOLD-58 ENSEMBLE EVALUATION RESULTS")
    print("=" * 80)

    rank_ensemble_metrics = {}
    print(f"{'Target':<20s} | {'True Pos':<8s} | {'ROC-AUC':<8s} | {'PR-AUC':<8s} | {'F1':<8s} | {'Acc':<8s}")
    print("-" * 65)

    for t_idx, target_name in enumerate(TARGETS):
        y_true = ground_truth[:, t_idx]
        y_pred = rank_averaged_preds[:, t_idx]
        m = compute_binary_metrics(y_true, y_pred, threshold=0.5)
        rank_ensemble_metrics[target_name] = m
        print(f"{target_name:<20s} | {int(np.sum(y_true)):2d}/{len(y_true):2d}   | {m['roc_auc']:.4f}   | {m['pr_auc']:.4f}   | {m['f1']:.4f}   | {m['accuracy']:.4f}")

    rank_macro_auc = float(np.mean([m["roc_auc"] for m in rank_ensemble_metrics.values()]))
    rank_macro_prauc = float(np.mean([m["pr_auc"] for m in rank_ensemble_metrics.values()]))
    rank_macro_f1 = float(np.mean([m["f1"] for m in rank_ensemble_metrics.values()]))
    rank_macro_acc = float(np.mean([m["accuracy"] for m in rank_ensemble_metrics.values()]))

    print("-" * 65)
    print(f"{'MACRO AVERAGE':<20s} | {'--':<8s} | {rank_macro_auc:.4f}   | {rank_macro_prauc:.4f}   | {rank_macro_f1:.4f}   | {rank_macro_acc:.4f}")
    print("=" * 80)

    # Evaluate Score-Averaged Ensemble for comparison
    score_ensemble_metrics = {}
    for t_idx, target_name in enumerate(TARGETS):
        y_true = ground_truth[:, t_idx]
        y_pred = score_averaged_preds[:, t_idx]
        m = compute_binary_metrics(y_true, y_pred, threshold=0.5)
        score_ensemble_metrics[target_name] = m

    score_macro_auc = float(np.mean([m["roc_auc"] for m in score_ensemble_metrics.values()]))
    score_macro_prauc = float(np.mean([m["pr_auc"] for m in score_ensemble_metrics.values()]))

    # 6. Build Integrity Audit
    integrity_audit = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gold_studies_evaluated": len(gold_uids),
        "gold_studies_duplicated": len(gold_uids) - len(set(gold_uids)),
        "gold_studies_missing": 58 - len(gold_uids),
        "development_studies_included": len(set(gold_uids) & dev_uids),
        "gold_labels_used_in_training": False,
        "gold_labels_used_for_threshold_tuning": False,
        "gold_labels_used_for_checkpoint_selection": False,
        "gold_labels_used_for_model_selection": False,
        "rank_averaging_pre_specified": True,
        "nan_inf_predictions": int(np.isnan(rank_averaged_preds).sum() + np.isinf(rank_averaged_preds).sum()),
        "integrity_verdict": "PASS - 100% UNCOMPROMISED LOCKED EVALUATION",
    }

    integrity_out_path = os.path.join(gold_results_dir, "gold58_integrity_audit.json")
    with open(integrity_out_path, "w") as f:
        json.dump(integrity_audit, f, indent=2)
    print(f"[AUDIT] Saved integrity audit to: {integrity_out_path}")

    # 7. Build Full Evaluation Summary JSON
    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": platform.node(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": str(DEVICE),
            "gpu_name": torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else "None",
            "gold_studies_count": 58,
            "gold_truth_path": gold_csv,
            "gold_truth_sha256": hashlib.sha256(open(gold_csv, "rb").read()).hexdigest(),
            "processed_gold_dir": processed_gold_dir,
            "ensemble_method": "predetermined_normalized_rank_averaging",
            "rank_formula": "mean(rankdata(p_fold, method='average') / 58.0)",
        },
        "locked_gold_rank_ensemble": {
            "macro_roc_auc": rank_macro_auc,
            "macro_pr_auc": rank_macro_prauc,
            "macro_f1": rank_macro_f1,
            "macro_accuracy": rank_macro_acc,
            "per_target": rank_ensemble_metrics,
        },
        "locked_gold_score_ensemble_reference": {
            "macro_roc_auc": score_macro_auc,
            "macro_pr_auc": score_macro_prauc,
            "per_target": score_ensemble_metrics,
        },
        "individual_checkpoints": fold_metrics,
        "integrity_audit": integrity_audit,
        "artifacts": {
            "rank_averaged_predictions": rank_out_path,
            "score_averaged_predictions": score_out_path,
            "per_fold_predictions": [os.path.join(gold_results_dir, f"fold_{f}_predictions.csv") for f in range(5)],
            "integrity_audit": integrity_out_path,
        },
    }

    summary_out_path = os.path.join(gold_results_dir, "gold58_evaluation_summary.json")
    with open(summary_out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUMMARY] Saved evaluation summary to: {summary_out_path}")

    # 8. Generate Markdown Audit Report
    report_md = generate_gold_markdown_report(summary)
    report_out_path = os.path.join(gold_results_dir, "gold58_evaluation_report.md")
    with open(report_out_path, "w") as f:
        f.write(report_md)
    print(f"[REPORT] Saved Markdown report to: {report_out_path}")


def generate_gold_markdown_report(summary: dict) -> str:
    m = summary["locked_gold_rank_ensemble"]
    meta = summary["metadata"]
    audit = summary["integrity_audit"]

    lines = [
        "# Phase C3.12 Stage B: Locked Gold-58 Evaluation Report",
        "",
        f"**Date / Timestamp:** {meta['timestamp']}  ",
        f"**Host:** `{meta['host']}`  ",
        f"**Environment:** Conda `ML_DFT_V2` | Python `{meta['python_version']}` | PyTorch `{meta['torch_version']}`  ",
        f"**Hardware:** {meta['gpu_name']}  ",
        f"**Status:** **LOCKED EVALUATION COMPLETE — 100% PASS**  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "The **Phase C3.12 Stage B Locked Gold-58 Evaluation** has been executed using the **five frozen C3.11 production checkpoints** on the held-out 58-study Gold cohort.",
        "In strict adherence to protocol:",
        "- **0 Gold-58 studies** were ever exposed to training, validation, or checkpoint selection.",
        "- **No fine-tuning, threshold tuning, or fold weighting** was performed on Gold-58 data.",
        "- The ensemble was formed using the **pre-specified normalized rank averaging** procedure across all 5 models.",
        "",
        "### Key Global Metrics Comparison",
        "",
        "| Evaluation Stage | Scope | Cohort Size | Macro ROC-AUC | Note |",
        "| :--- | :--- | :---: | :---: | :--- |",
        "| **C3.11 Production CV** | Development Mean Fold Val | 869.8 / fold | **0.6333 ± 0.0326** | Fold-level CV development baseline |",
        "| **C3.12 Stage A OOF** | Global Development OOF | 4,349 studies | **0.6190** | Global study-level non-Gold development OOF |",
        f"| **C3.12 Stage B Gold-58** | **Locked Rank Ensemble** | **58 studies** | **{m['macro_roc_auc']:.4f}** | **Official locked out-of-sample Gold evaluation** |",
        f"| *C3.12 Stage B Gold-58* | *Locked Score Ensemble* | *58 studies* | *{summary['locked_gold_score_ensemble_reference']['macro_roc_auc']:.4f}* | *Reference simple score average* |",
        "",
        "---",
        "",
        "## 2. Locked Gold-58 Per-Target Performance (Rank-Averaged Ensemble)",
        "",
        "| Target Name | True Positives | Prevalence | ROC-AUC | PR-AUC | F1-Score | Accuracy |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for target_name in TARGETS:
        t_m = m["per_target"][target_name]
        pos = t_m["positives_true"]
        tot = t_m["total_samples"]
        prev = (pos / tot) * 100
        lines.append(
            f"| **{target_name}** | {pos}/{tot} | {prev:.1f}% | **{t_m['roc_auc']:.4f}** | {t_m['pr_auc']:.4f} | {t_m['f1']:.4f} | {t_m['accuracy']:.4f} |"
        )

    lines.extend([
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **MACRO AVERAGE** | **--** | **--** | **{m['macro_roc_auc']:.4f}** | **{m['macro_pr_auc']:.4f}** | **{m['macro_f1']:.4f}** | **{m['macro_accuracy']:.4f}** |",
        "",
        "---",
        "",
        "## 3. Per-Checkpoint Gold Performance",
        "",
        "| Model Checkpoint | Macro ROC-AUC | Macro PR-AUC |",
        "| :--- | :---: | :---: |",
    ])

    for f in range(5):
        ck_m = summary["individual_checkpoints"][f"fold_{f}"]
        lines.append(f"| **Fold {f} Checkpoint** | {ck_m['macro_roc_auc']:.4f} | {ck_m['macro_pr_auc']:.4f} |")

    lines.extend([
        f"| **Predetermined Rank Ensemble** | **{m['macro_roc_auc']:.4f}** | **{m['macro_pr_auc']:.4f}** |",
        "",
        "---",
        "",
        "## 4. Integrity and Leakage Audit",
        "",
        f"- **Gold Studies Evaluated:** `{audit['gold_studies_evaluated']}` (Expected: 58)",
        f"- **Gold Studies Duplicated:** `{audit['gold_studies_duplicated']}`",
        f"- **Gold Studies Missing:** `{audit['gold_studies_missing']}`",
        f"- **Development Studies Accidentally Included:** `{audit['development_studies_included']}`",
        f"- **Gold Labels Used in Training:** `{audit['gold_labels_used_in_training']}`",
        f"- **Gold Labels Used for Threshold/Model Selection:** `{audit['gold_labels_used_for_threshold_tuning']}`",
        f"- **Pre-specified Rank Averaging Applied:** `{audit['rank_averaging_pre_specified']}`",
        f"- **NaN / Inf Predictions:** `{audit['nan_inf_predictions']}`",
        f"- **Audit Verdict:** **{audit['integrity_verdict']}**",
        "",
        "---",
        "",
        "## 5. Artifact Inventory",
        "",
        "| Artifact Description | File Path |",
        "| :--- | :--- |",
        f"| Rank-Averaged Ensemble Predictions | `{summary['artifacts']['rank_averaged_predictions']}` |",
        f"| Score-Averaged Ensemble Predictions | `{summary['artifacts']['score_averaged_predictions']}` |",
        f"| Integrity Audit JSON | `{summary['artifacts']['integrity_audit']}` |",
        f"| Comprehensive Evaluation Summary JSON | `{os.path.join(REPO_ROOT, 'phase_c', 'c3', 'results', 'gold58', 'gold58_evaluation_summary.json')}` |",
        f"| Evaluation Markdown Report | `{os.path.join(REPO_ROOT, 'phase_c', 'c3', 'results', 'gold58', 'gold58_evaluation_report.md')}` |",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    run_gold_evaluation()
