"""
Phase D1.2-C: Multi-Window / Slice-Window Test-Time Augmentation (TTA) Runner.

Investigates controlled variations in slice sampling windows:
- Baseline: slice_band = (0.20, 0.80) [Central 60%]
- Window 1 (Anterior Shift): slice_band = (0.15, 0.75)
- Window 2 (Posterior Shift): slice_band = (0.25, 0.85)
- Window 3 (Expanded Band): slice_band = (0.15, 0.85)

Evaluates:
- Each window candidate independently vs D1.1 baseline.
- 2-Window average (Baseline + Win1)
- 3-Window average (Baseline + Win1 + Win2)
- Rank-averaged vs arithmetic probability averaged combinations.

Uses parallel dual-GPU execution (GPU 0: Folds 0, 1; GPU 1: Folds 2, 3, 4).
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "phase_d/d1/predictions/d1_2_c_multiwindow")

WINDOWS = {
    "win_ant": (0.15, 0.75),
    "win_post": (0.25, 0.85),
    "win_exp": (0.15, 0.85),
}


def run_window_worker(folds: List[int], gpu_id: int, window_name: str, slice_band: Tuple[float, float], out_queue: Queue):
    """Worker function processing multi-window predictions on a specific GPU."""
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
            slice_band=slice_band,
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
        p_list = []
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
                    logits, _ = model(images, slot_mask)
                    probs = torch.sigmoid(logits).cpu().numpy()

                p_list.append(probs)

        probs_all = np.concatenate(p_list, axis=0)
        y_true = np.concatenate(targets_list, axis=0)
        y_weights = np.concatenate(weights_list, axis=0)

        pred_dict = {"StudyInstanceUID": uids, "fold": fold}
        for idx, path_name in enumerate(PATHOLOGY_NAMES):
            pred_dict[f"pred_{path_name}"] = probs_all[:, idx]
            pred_dict[f"true_{path_name}"] = y_true[:, idx]
            pred_dict[f"weight_{path_name}"] = y_weights[:, idx]

        df_fold = pd.DataFrame(pred_dict)
        t_fold = time.time() - t0
        print(f"[GPU {gpu_id} | {window_name}] Fold {fold} Complete: {len(df_fold)} studies in {t_fold:.1f}s")
        out_queue.put((fold, df_fold))


def evaluate_window_setting(window_name: str, slice_band: Tuple[float, float]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run full 5-fold evaluation for a specific slice window setting."""
    start_time = time.time()
    out_queue = Queue()
    win_dir = os.path.join(OUTPUT_DIR, window_name)
    os.makedirs(win_dir, exist_ok=True)

    print(f"\nEvaluating Window: {window_name} | Band: {slice_band} across 2 x A100 GPUs...")
    p0 = Process(target=run_window_worker, args=([0, 1], 0, window_name, slice_band, out_queue))
    p1 = Process(target=run_window_worker, args=([2, 3, 4], 1, window_name, slice_band, out_queue))

    p0.start()
    p1.start()

    received_folds = {}
    for _ in range(5):
        fold_idx, df_fold = out_queue.get()
        received_folds[fold_idx] = df_fold

    p0.join()
    p1.join()

    runtime = round(time.time() - start_time, 1)
    ordered_dfs = [received_folds[i] for i in range(5)]
    global_df = pd.concat(ordered_dfs, ignore_index=True)

    csv_path = os.path.join(win_dir, f"{window_name}_global_oof.csv")
    global_df.to_csv(csv_path, index=False)

    eval_results = evaluate_oof_df(global_df, experiment_id=f"D1.2-C_{window_name.upper()}")
    eval_results["runtime_seconds"] = runtime
    eval_results["slice_band"] = list(slice_band)
    eval_results["window_name"] = window_name

    json_path = os.path.join(win_dir, f"{window_name}_evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"[{window_name.upper()}] Done in {runtime}s | Unweighted AUC: {eval_results['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_results['delta_unweighted_vs_d1_1']:+.4f}) | Weighted: {eval_results['global_macro_roc_auc_weighted']:.4f}")
    return global_df, eval_results


def main():
    print("===================================================================")
    print("PHASE D1.2-C: MULTI-WINDOW SLICE TTA PRODUCTION EXPERIMENT")
    print("===================================================================")
    
    # 1. Run Window 1 (Anterior Shift: 0.15 to 0.75)
    df_ant, eval_ant = evaluate_window_setting("win_ant", (0.15, 0.75))

    # 2. Run Window 2 (Posterior Shift: 0.25 to 0.85)
    df_post, eval_post = evaluate_window_setting("win_post", (0.25, 0.85))

    # 3. Load D1.1 Baseline OOF
    base_csv = os.path.join(PROJECT_ROOT, "phase_d/d1/results/oof/d1_1_global_oof_predictions.csv")
    df_base = pd.read_csv(base_csv)

    # 4. Multi-Window Combinations
    uids = df_base["StudyInstanceUID"].values
    folds = df_base["fold"].values
    y_true = df_base[[f"true_{t}" for t in PATHOLOGY_NAMES]].values
    y_weights = df_base[[f"weight_{t}" for t in PATHOLOGY_NAMES]].values

    p_base = df_base[[f"pred_{t}" for t in PATHOLOGY_NAMES]].values
    p_ant = df_ant.set_index("StudyInstanceUID").reindex(uids)[[f"pred_{t}" for t in PATHOLOGY_NAMES]].values
    p_post = df_post.set_index("StudyInstanceUID").reindex(uids)[[f"pred_{t}" for t in PATHOLOGY_NAMES]].values

    # Multi-Window Combination 1: 3-Window Probability Mean (Baseline + Ant + Post)
    p_3win_prob = (p_base + p_ant + p_post) / 3.0
    df_3win_prob = pd.DataFrame({"StudyInstanceUID": uids, "fold": folds})
    for idx, t in enumerate(PATHOLOGY_NAMES):
        df_3win_prob[f"pred_{t}"] = p_3win_prob[:, idx]
        df_3win_prob[f"true_{t}"] = y_true[:, idx]
        df_3win_prob[f"weight_{t}"] = y_weights[:, idx]
    eval_3win_prob = evaluate_oof_df(df_3win_prob, experiment_id="D1.2-C_3WIN_PROB_MEAN")
    eval_3win_prob["method"] = "3-Window Arithmetic Probability Mean (0.20-0.80 + 0.15-0.75 + 0.25-0.85)"

    # Multi-Window Combination 2: 3-Window Rank Mean
    N = len(uids)
    r_base = np.zeros_like(p_base)
    r_ant = np.zeros_like(p_ant)
    r_post = np.zeros_like(p_post)
    for c in range(12):
        r_base[:, c] = rankdata(p_base[:, c], method="average") / float(N)
        r_ant[:, c] = rankdata(p_ant[:, c], method="average") / float(N)
        r_post[:, c] = rankdata(p_post[:, c], method="average") / float(N)
    p_3win_rank = (r_base + r_ant + r_post) / 3.0

    df_3win_rank = pd.DataFrame({"StudyInstanceUID": uids, "fold": folds})
    for idx, t in enumerate(PATHOLOGY_NAMES):
        df_3win_rank[f"pred_{t}"] = p_3win_rank[:, idx]
        df_3win_rank[f"true_{t}"] = y_true[:, idx]
        df_3win_rank[f"weight_{t}"] = y_weights[:, idx]
    eval_3win_rank = evaluate_oof_df(df_3win_rank, experiment_id="D1.2-C_3WIN_RANK_MEAN")
    eval_3win_rank["method"] = "3-Window Rank Mean (0.20-0.80 + 0.15-0.75 + 0.25-0.85)"

    # Save multi-window combined predictions
    comb_dir = os.path.join(OUTPUT_DIR, "combined")
    os.makedirs(comb_dir, exist_ok=True)
    df_3win_prob.to_csv(os.path.join(comb_dir, "d1_2_c_3win_prob_mean_oof.csv"), index=False)
    df_3win_rank.to_csv(os.path.join(comb_dir, "d1_2_c_3win_rank_mean_oof.csv"), index=False)

    summary_file = os.path.join(OUTPUT_DIR, "d1_2_c_multiwindow_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "WIN_ANT": eval_ant,
            "WIN_POST": eval_post,
            "3WIN_PROB_MEAN": eval_3win_prob,
            "3WIN_RANK_MEAN": eval_3win_rank,
        }, f, indent=2)

    print("\n===================================================================")
    print("D1.2-C MULTI-WINDOW TTA BENCHMARK RESULTS:")
    print(f"D1.1 Baseline OOF AUC:               0.7510 (Unw) | 0.7573 (Wgt)")
    print(f"Window Ant (0.15-0.75) OOF AUC:      {eval_ant['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_ant['delta_unweighted_vs_d1_1']:+.4f}) | Wgt: {eval_ant['global_macro_roc_auc_weighted']:.4f}")
    print(f"Window Post (0.25-0.85) OOF AUC:     {eval_post['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_post['delta_unweighted_vs_d1_1']:+.4f}) | Wgt: {eval_post['global_macro_roc_auc_weighted']:.4f}")
    print(f"3-Window Prob Mean OOF AUC:          {eval_3win_prob['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_3win_prob['delta_unweighted_vs_d1_1']:+.4f}) | Wgt: {eval_3win_prob['global_macro_roc_auc_weighted']:.4f}")
    print(f"3-Window Rank Mean OOF AUC:          {eval_3win_rank['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_3win_rank['delta_unweighted_vs_d1_1']:+.4f}) | Wgt: {eval_3win_rank['global_macro_roc_auc_weighted']:.4f}")
    print("===================================================================\n")


if __name__ == "__main__":
    main()

