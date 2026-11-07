"""
Phase D1.2-E: Rank Transformations and Prediction Blending Engine.

Investigates controlled rank-based and probability-based prediction aggregations:
- Equal Probability Average (Arithmetic mean)
- Equal Rank Average (Percentile ranking within each prediction stream)
- Hybrid 50/50 Probability-Rank Blend
- Evaluates impact on Global Unweighted Macro ROC-AUC, Weighted Macro ROC-AUC, and PR-AUC.
- Strictly adheres to development-only protocol with zero Gold-58 interaction.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase_d.d1.scripts.d1_2_eval_utils import evaluate_oof_df, PATHOLOGY_NAMES

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "phase_d/d1/predictions/d1_2_e_blends")


def rank_transform_matrix(probs: np.ndarray) -> np.ndarray:
    """Convert probability matrix (N, C) to uniform percentile ranks [0, 1] per column."""
    N = probs.shape[0]
    ranks = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        ranks[:, c] = rankdata(probs[:, c], method="average") / float(N)
    return ranks


def blend_prediction_tables(
    tables_dict: Dict[str, pd.DataFrame],
    experiment_prefix: str = "D1.2-E",
    within_fold_rank: bool = True
) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Takes a dictionary of {name: oof_df} where each dataframe has identical StudyInstanceUIDs,
    targets, and weights.
    
    Evaluates:
      1. Arithmetic Probability Mean
      2. Rank Average (Within-Fold or Global)
      3. 50/50 Hybrid (Arithmetic + Rank)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = list(tables_dict.keys())
    assert len(names) >= 2, "Must provide at least 2 tables to blend"

    # Base reference
    ref_df = tables_dict[names[0]]
    uids = ref_df["StudyInstanceUID"].values
    folds = ref_df["fold"].values
    y_true = ref_df[[f"true_{t}" for t in PATHOLOGY_NAMES]].values
    y_weights = ref_df[[f"weight_{t}" for t in PATHOLOGY_NAMES]].values

    # Extract probability matrices: List of (N, 12)
    prob_matrices = []
    rank_matrices = []
    for name in names:
        df = tables_dict[name]
        df_aligned = df.set_index("StudyInstanceUID").reindex(uids).reset_index()
        p = df_aligned[[f"pred_{t}" for t in PATHOLOGY_NAMES]].values
        prob_matrices.append(p)

        if within_fold_rank:
            r = np.zeros_like(p)
            for f_idx in range(5):
                f_mask = (folds == f_idx)
                N_f = f_mask.sum()
                for c in range(12):
                    r[f_mask, c] = rankdata(p[f_mask, c], method="average") / float(N_f)
            rank_matrices.append(r)
        else:
            rank_matrices.append(rank_transform_matrix(p))

    # 1. Arithmetic Probability Mean
    p_arith = np.mean(prob_matrices, axis=0)
    df_arith = pd.DataFrame({"StudyInstanceUID": uids, "fold": folds})
    for idx, t in enumerate(PATHOLOGY_NAMES):
        df_arith[f"pred_{t}"] = p_arith[:, idx]
        df_arith[f"true_{t}"] = y_true[:, idx]
        df_arith[f"weight_{t}"] = y_weights[:, idx]

    eval_arith = evaluate_oof_df(df_arith, experiment_id=f"{experiment_prefix}_PROB_MEAN")
    eval_arith["method"] = f"Equal Probability Average of [{', '.join(names)}]"
    df_arith.to_csv(os.path.join(OUTPUT_DIR, f"{experiment_prefix.lower()}_prob_mean_oof.csv"), index=False)

    # 2. Equal Rank Average
    p_rank = np.mean(rank_matrices, axis=0)
    df_rank = pd.DataFrame({"StudyInstanceUID": uids, "fold": folds})
    for idx, t in enumerate(PATHOLOGY_NAMES):
        df_rank[f"pred_{t}"] = p_rank[:, idx]
        df_rank[f"true_{t}"] = y_true[:, idx]
        df_rank[f"weight_{t}"] = y_weights[:, idx]

    eval_rank = evaluate_oof_df(df_rank, experiment_id=f"{experiment_prefix}_RANK_MEAN")
    eval_rank["method"] = f"Equal Rank Average of [{', '.join(names)}]"
    df_rank.to_csv(os.path.join(OUTPUT_DIR, f"{experiment_prefix.lower()}_rank_mean_oof.csv"), index=False)

    # 3. 50/50 Hybrid (Arithmetic + Rank)
    # Rank-transform the arithmetic mean, then average with rank mean
    r_arith = rank_transform_matrix(p_arith)
    p_hybrid = 0.5 * (r_arith + p_rank)
    df_hybrid = pd.DataFrame({"StudyInstanceUID": uids, "fold": folds})
    for idx, t in enumerate(PATHOLOGY_NAMES):
        df_hybrid[f"pred_{t}"] = p_hybrid[:, idx]
        df_hybrid[f"true_{t}"] = y_true[:, idx]
        df_hybrid[f"weight_{t}"] = y_weights[:, idx]

    eval_hybrid = evaluate_oof_df(df_hybrid, experiment_id=f"{experiment_prefix}_HYBRID_BLEND")
    eval_hybrid["method"] = f"50/50 Hybrid (Arithmetic Rank + Pure Rank Average) of [{', '.join(names)}]"
    df_hybrid.to_csv(os.path.join(OUTPUT_DIR, f"{experiment_prefix.lower()}_hybrid_blend_oof.csv"), index=False)

    results = {
        "PROB_MEAN": (df_arith, eval_arith),
        "RANK_MEAN": (df_rank, eval_rank),
        "HYBRID_BLEND": (df_hybrid, eval_hybrid),
    }

    summary_file = os.path.join(OUTPUT_DIR, f"{experiment_prefix.lower()}_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "PROB_MEAN": eval_arith,
            "RANK_MEAN": eval_rank,
            "HYBRID_BLEND": eval_hybrid,
        }, f, indent=2)

    return results
