"""
Phase D1.2-B: Horizontal-Flip Test-Time Augmentation (TTA) Runner.

Executes 5-fold OOF inference with horizontal flipping:
- For each study/slot: generates both P_orig and P_flip.
- Computes arithmetic probability mean: P_mean = 0.5 * P_orig + 0.5 * P_flip.
- Computes rank mean: P_rank = 0.5 * rank(P_orig) + 0.5 * rank(P_flip).
- Evaluates:
    1. D1.2-B1: Arithmetic Probability Mean
    2. D1.2-B2: Rank-Averaged Mean
    3. P_flip alone (diagnostic)
- Utilizes 2 x A100 GPUs in parallel (GPU 0: Folds 0, 1; GPU 1: Folds 2, 3, 4).
- Saves full prediction matrices and evaluation summaries.
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
from phase_d.d1.scripts.dataset_d1 import PhaseD1KneeDataset
from phase_d.d1.scripts.d1_2_eval_utils import evaluate_oof_df, PATHOLOGY_NAMES

CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
SLOT_MANIFEST_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/manifests/d1_slot_mapping_manifest.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "phase_c/c1/processed")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d1/checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "phase_d/d1/predictions/d1_2_b_hflip")


def run_hflip_worker(folds: List[int], gpu_id: int, out_queue: Queue):
    """Worker function processing a subset of folds on a specific GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    with open(CONFIG_CV) as f:
        cv_data = json.load(f)

    slot_manifest_df = pd.read_csv(SLOT_MANIFEST_CSV)
    labels_df = pd.read_csv(LABELS_CSV)

    for fold in folds:
        t0 = time.time()
        val_studies = cv_data["folds"][f"fold_{fold}"]["val_studies"]
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"fold_{fold}/best_model.pt")

        dataset = PhaseD1KneeDataset(
            study_uids=val_studies,
            slot_manifest_df=slot_manifest_df,
            labels_df=labels_df,
            processed_dir=PROCESSED_DIR,
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
        p_orig_list = []
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

                with torch.amp.autocast("cuda"):
                    # 1. Standard prediction
                    logits_orig, _ = model(images, slot_mask)
                    probs_orig = torch.sigmoid(logits_orig).cpu().numpy()

                    # 2. Horizontal flip along width axis (axis -1)
                    images_flip = torch.flip(images, dims=[-1])
                    logits_flip, _ = model(images_flip, slot_mask)
                    probs_flip = torch.sigmoid(logits_flip).cpu().numpy()

                p_orig_list.append(probs_orig)
                p_flip_list.append(probs_flip)

        p_orig = np.concatenate(p_orig_list, axis=0)
        p_flip = np.concatenate(p_flip_list, axis=0)
        p_mean = 0.5 * (p_orig + p_flip)
        y_true = np.concatenate(targets_list, axis=0)
        y_weights = np.concatenate(weights_list, axis=0)

        # Rank mean within fold
        N = len(uids)
        p_rank = np.zeros_like(p_orig)
        for c in range(12):
            r_orig = rankdata(p_orig[:, c], method="average") / N
            r_flip = rankdata(p_flip[:, c], method="average") / N
            p_rank[:, c] = 0.5 * (r_orig + r_flip)

        pred_dict = {"StudyInstanceUID": uids, "fold": fold}
        for idx, path_name in enumerate(PATHOLOGY_NAMES):
            pred_dict[f"pred_{path_name}"] = p_mean[:, idx]  # default to prob mean
            pred_dict[f"pred_orig_{path_name}"] = p_orig[:, idx]
            pred_dict[f"pred_flip_{path_name}"] = p_flip[:, idx]
            pred_dict[f"pred_rank_{path_name}"] = p_rank[:, idx]
            pred_dict[f"true_{path_name}"] = y_true[:, idx]
            pred_dict[f"weight_{path_name}"] = y_weights[:, idx]

        df_fold = pd.DataFrame(pred_dict)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fold_csv = os.path.join(OUTPUT_DIR, f"fold_{fold}_hflip_predictions.csv")
        df_fold.to_csv(fold_csv, index=False)
        t_fold = time.time() - t0
        print(f"[GPU {gpu_id}] Fold {fold} Complete: {len(df_fold)} studies in {t_fold:.1f}s")
        out_queue.put((fold, df_fold))


def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_queue = Queue()

    print("===================================================================")
    print("PHASE D1.2-B: HORIZONTAL-FLIP TTA PRODUCTION RUN (2 x A100 DUAL-GPU)")
    print("===================================================================")

    # Worker 0: Folds 0, 1 on GPU 0
    # Worker 1: Folds 2, 3, 4 on GPU 1
    p0 = Process(target=run_hflip_worker, args=([0, 1], 0, out_queue))
    p1 = Process(target=run_hflip_worker, args=([2, 3, 4], 1, out_queue))

    p0.start()
    p1.start()

    received_folds = {}
    for _ in range(5):
        fold_idx, df_fold = out_queue.get()
        received_folds[fold_idx] = df_fold

    p0.join()
    p1.join()

    total_runtime = round(time.time() - start_time, 1)
    print(f"\nAll 5 folds finished in {total_runtime}s!")

    # Sort folds 0..4
    ordered_dfs = [received_folds[i] for i in range(5)]
    global_df = pd.concat(ordered_dfs, ignore_index=True)
    global_csv = os.path.join(OUTPUT_DIR, "d1_2_b_hflip_global_oof.csv")
    global_df.to_csv(global_csv, index=False)
    print(f"Saved global table to: {global_csv} (Rows: {len(global_df)})")

    # Evaluate Variant 1: Arithmetic Probability Mean (pred_*)
    eval_prob_mean = evaluate_oof_df(global_df, experiment_id="D1.2-B1_HFLIP_PROB_MEAN")
    eval_prob_mean["runtime_seconds"] = total_runtime
    eval_prob_mean["method"] = "Horizontal Flip (Arithmetic Probability Mean: 0.5*Orig + 0.5*Flip)"

    # Evaluate Variant 2: Rank Mean (pred_rank_*)
    df_rank = global_df.copy()
    for t in PATHOLOGY_NAMES:
        df_rank[f"pred_{t}"] = df_rank[f"pred_rank_{t}"]
    eval_rank_mean = evaluate_oof_df(df_rank, experiment_id="D1.2-B2_HFLIP_RANK_MEAN")
    eval_rank_mean["runtime_seconds"] = total_runtime
    eval_rank_mean["method"] = "Horizontal Flip (Rank Mean: 0.5*Rank_Orig + 0.5*Rank_Flip)"

    # Evaluate Variant 3: Flipped Alone (pred_flip_*)
    df_flip_alone = global_df.copy()
    for t in PATHOLOGY_NAMES:
        df_flip_alone[f"pred_{t}"] = df_flip_alone[f"pred_flip_{t}"]
    eval_flip_alone = evaluate_oof_df(df_flip_alone, experiment_id="D1.2-B_DIAGNOSTIC_FLIP_ALONE")
    eval_flip_alone["runtime_seconds"] = total_runtime
    eval_flip_alone["method"] = "Horizontal Flip Alone (Diagnostic)"

    # Save summary JSON
    summary_path = os.path.join(OUTPUT_DIR, "d1_2_b_hflip_evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "D1.2-B1_PROB_MEAN": eval_prob_mean,
            "D1.2-B2_RANK_MEAN": eval_rank_mean,
            "D1.2-B_FLIP_ALONE": eval_flip_alone,
        }, f, indent=2)

    print("\n===================================================================")
    print("D1.2-B HORIZONTAL-FLIP TTA BENCHMARK RESULTS:")
    print(f"D1.1 Baseline OOF AUC:            0.7510 (Unweighted) | 0.7573 (Weighted)")
    print(f"D1.2-B1 (Prob Mean) OOF AUC:      {eval_prob_mean['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_prob_mean['delta_unweighted_vs_d1_1']:+.4f}) | Weighted: {eval_prob_mean['global_macro_roc_auc_weighted']:.4f}")
    print(f"D1.2-B2 (Rank Mean) OOF AUC:      {eval_rank_mean['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_rank_mean['delta_unweighted_vs_d1_1']:+.4f}) | Weighted: {eval_rank_mean['global_macro_roc_auc_weighted']:.4f}")
    print(f"Diagnostic Flip Alone OOF AUC:    {eval_flip_alone['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_flip_alone['delta_unweighted_vs_d1_1']:+.4f})")
    print(f"Total Runtime: {total_runtime}s across 2 x A100 GPUs")
    print("===================================================================\n")


if __name__ == "__main__":
    main()
