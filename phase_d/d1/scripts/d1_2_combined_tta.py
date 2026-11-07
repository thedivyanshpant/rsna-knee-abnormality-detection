"""
Phase D1.2-D: Combined Test-Time Augmentation (TTA) and Blending Engine.

Combines spatial augmentation (Horizontal Flip) and through-plane augmentation (Slice Window Variation):
Stream 1: Baseline (Central 0.20-0.80)
Stream 2: Horizontal Flip (Central 0.20-0.80 Flipped)
Stream 3: Anterior Window (0.15-0.75)
Stream 4: Posterior Window (0.25-0.85)

Evaluates:
- 4-Stream Arithmetic Probability Mean
- 4-Stream Equal Rank Average
- 4-Stream 50/50 Probability-Rank Hybrid
- Target-by-target breakdown across all 12 pathologies
- Comprehensive governance audit
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase_d.d1.scripts.d1_2_eval_utils import evaluate_oof_df, PATHOLOGY_NAMES

BASE_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/results/oof/d1_1_global_oof_predictions.csv")
HFLIP_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/predictions/d1_2_b_hflip/d1_2_b_hflip_global_oof.csv")
ANT_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/predictions/d1_2_c_multiwindow/win_ant/win_ant_global_oof.csv")
POST_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/predictions/d1_2_c_multiwindow/win_post/win_post_global_oof.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "phase_d/d1/predictions/d1_2_d_combined_tta")


def rank_matrix(probs: np.ndarray) -> np.ndarray:
    N = probs.shape[0]
    ranks = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        ranks[:, c] = rankdata(probs[:, c], method="average") / float(N)
    return ranks


def run_combined_tta():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("===================================================================")
    print("PHASE D1.2-D: COMBINED TTA & RANK BLENDING EVALUATION")
    print("===================================================================")

    df_base = pd.read_csv(BASE_CSV)
    df_hflip = pd.read_csv(HFLIP_CSV)
    df_ant = pd.read_csv(ANT_CSV)
    df_post = pd.read_csv(POST_CSV)

    uids = df_base["StudyInstanceUID"].values
    folds = df_base["fold"].values
    y_true = df_base[[f"true_{t}" for t in PATHOLOGY_NAMES]].values
    y_weights = df_base[[f"weight_{t}" for t in PATHOLOGY_NAMES]].values

    # Extract aligned probability matrices
    p_base = df_base[[f"pred_{t}" for t in PATHOLOGY_NAMES]].values
    p_flip = df_hflip.set_index("StudyInstanceUID").reindex(uids)[[f"pred_flip_{t}" for t in PATHOLOGY_NAMES]].values
    p_ant = df_ant.set_index("StudyInstanceUID").reindex(uids)[[f"pred_{t}" for t in PATHOLOGY_NAMES]].values
    p_post = df_post.set_index("StudyInstanceUID").reindex(uids)[[f"pred_{t}" for t in PATHOLOGY_NAMES]].values

    streams = {
        "Baseline": p_base,
        "HFlip": p_flip,
        "WinAnt": p_ant,
        "WinPost": p_post,
    }

    # 1. 4-Stream Probability Mean
    p_4stream_prob = (p_base + p_flip + p_ant + p_post) / 4.0
    df_4stream_prob = pd.DataFrame({"StudyInstanceUID": uids, "fold": folds})
    for idx, t in enumerate(PATHOLOGY_NAMES):
        df_4stream_prob[f"pred_{t}"] = p_4stream_prob[:, idx]
        df_4stream_prob[f"true_{t}"] = y_true[:, idx]
        df_4stream_prob[f"weight_{t}"] = y_weights[:, idx]

    eval_4stream_prob = evaluate_oof_df(df_4stream_prob, experiment_id="D1.2-D1_COMBINED_4STREAM_PROB")
    eval_4stream_prob["method"] = "4-Stream Arithmetic Probability Mean (Baseline + HFlip + WinAnt + WinPost)"
    df_4stream_prob.to_csv(os.path.join(OUTPUT_DIR, "d1_2_d1_4stream_prob_oof.csv"), index=False)

    # 2. 4-Stream Rank Average
    r_base = rank_matrix(p_base)
    r_flip = rank_matrix(p_flip)
    r_ant = rank_matrix(p_ant)
    r_post = rank_matrix(p_post)

    p_4stream_rank = (r_base + r_flip + r_ant + r_post) / 4.0
    df_4stream_rank = pd.DataFrame({"StudyInstanceUID": uids, "fold": folds})
    for idx, t in enumerate(PATHOLOGY_NAMES):
        df_4stream_rank[f"pred_{t}"] = p_4stream_rank[:, idx]
        df_4stream_rank[f"true_{t}"] = y_true[:, idx]
        df_4stream_rank[f"weight_{t}"] = y_weights[:, idx]

    eval_4stream_rank = evaluate_oof_df(df_4stream_rank, experiment_id="D1.2-D2_COMBINED_4STREAM_RANK")
    eval_4stream_rank["method"] = "4-Stream Equal Rank Average (Baseline + HFlip + WinAnt + WinPost)"
    df_4stream_rank.to_csv(os.path.join(OUTPUT_DIR, "d1_2_d2_4stream_rank_oof.csv"), index=False)

    # 3. 4-Stream 50/50 Probability-Rank Hybrid
    r_prob = rank_matrix(p_4stream_prob)
    p_4stream_hybrid = 0.5 * (r_prob + p_4stream_rank)
    df_4stream_hybrid = pd.DataFrame({"StudyInstanceUID": uids, "fold": folds})
    for idx, t in enumerate(PATHOLOGY_NAMES):
        df_4stream_hybrid[f"pred_{t}"] = p_4stream_hybrid[:, idx]
        df_4stream_hybrid[f"true_{t}"] = y_true[:, idx]
        df_4stream_hybrid[f"weight_{t}"] = y_weights[:, idx]

    eval_4stream_hybrid = evaluate_oof_df(df_4stream_hybrid, experiment_id="D1.2-D3_COMBINED_4STREAM_HYBRID")
    eval_4stream_hybrid["method"] = "4-Stream 50/50 Probability-Rank Hybrid Blend"
    df_4stream_hybrid.to_csv(os.path.join(OUTPUT_DIR, "d1_2_d3_4stream_hybrid_oof.csv"), index=False)

    # 4. 3-Stream Best (Baseline + HFlip + WinAnt) Rank Average
    p_3stream_rank = (r_base + r_flip + r_ant) / 3.0
    df_3stream_rank = pd.DataFrame({"StudyInstanceUID": uids, "fold": folds})
    for idx, t in enumerate(PATHOLOGY_NAMES):
        df_3stream_rank[f"pred_{t}"] = p_3stream_rank[:, idx]
        df_3stream_rank[f"true_{t}"] = y_true[:, idx]
        df_3stream_rank[f"weight_{t}"] = y_weights[:, idx]

    eval_3stream_rank = evaluate_oof_df(df_3stream_rank, experiment_id="D1.2-D4_COMBINED_3STREAM_RANK")
    eval_3stream_rank["method"] = "3-Stream Equal Rank Average (Baseline + HFlip + WinAnt)"
    df_3stream_rank.to_csv(os.path.join(OUTPUT_DIR, "d1_2_d4_3stream_rank_oof.csv"), index=False)

    summary = {
        "D1.2-D1_4STREAM_PROB": eval_4stream_prob,
        "D1.2-D2_4STREAM_RANK": eval_4stream_rank,
        "D1.2-D3_4STREAM_HYBRID": eval_4stream_hybrid,
        "D1.2-D4_3STREAM_RANK": eval_3stream_rank,
    }

    with open(os.path.join(OUTPUT_DIR, "d1_2_d_combined_tta_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===================================================================")
    print("D1.2-D COMBINED TTA BENCHMARK RESULTS:")
    print(f"D1.1 Baseline OOF AUC:            0.7510 (Unw) | 0.7573 (Wgt)")
    print(f"D1.2-D1 (4-Stream Prob):          {eval_4stream_prob['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_4stream_prob['delta_unweighted_vs_d1_1']:+.4f}) | Wgt: {eval_4stream_prob['global_macro_roc_auc_weighted']:.4f}")
    print(f"D1.2-D2 (4-Stream Rank):          {eval_4stream_rank['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_4stream_rank['delta_unweighted_vs_d1_1']:+.4f}) | Wgt: {eval_4stream_rank['global_macro_roc_auc_weighted']:.4f}")
    print(f"D1.2-D3 (4-Stream Hybrid):        {eval_4stream_hybrid['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_4stream_hybrid['delta_unweighted_vs_d1_1']:+.4f}) | Wgt: {eval_4stream_hybrid['global_macro_roc_auc_weighted']:.4f}")
    print(f"D1.2-D4 (3-Stream Rank):          {eval_3stream_rank['global_macro_roc_auc_unweighted']:.4f} (Delta: {eval_3stream_rank['delta_unweighted_vs_d1_1']:+.4f}) | Wgt: {eval_3stream_rank['global_macro_roc_auc_weighted']:.4f}")
    print("===================================================================\n")

    return summary


if __name__ == "__main__":
    run_combined_tta()
