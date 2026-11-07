"""
Phase C3: 5-Fold Stratified Study-Level Cross-Validation Splitter.

Splits all 4,349 non-Gold development studies into 5 disjoint folds (seed 42).
Stratification is based on total positive pathology count per study to preserve
balanced multi-label disease prevalence across all 5 folds.
Strictly excludes Gold-58 holdout benchmark (0 leakage assertion).
"""
import os
import json
import numpy as np
import pandas as pd

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
C3_ROOT = os.path.join(REPO_ROOT, "phase_c", "c3")
CONFIGS_DIR = os.path.join(C3_ROOT, "configs")
LABELS_CSV = os.path.join(REPO_ROOT, "labels", "production", "phase_b_step3_remediated_training_corpus.csv")
GOLD58_CSV = os.path.join(REPO_ROOT, "labels", "gold58", "phase_b_step3_remediated_predictions_gold58.csv")
PROCESSED_DIR = os.path.join(REPO_ROOT, "phase_c", "c1", "processed")

SEED = 42
NUM_FOLDS = 5


def generate_cv_splits(num_folds: int = NUM_FOLDS, seed: int = SEED) -> dict:
    """Generate deterministic 5-fold stratified cross-validation splits."""
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    out_path = os.path.join(CONFIGS_DIR, "cv_splits.json")

    # 1. Load Gold-58 and verify
    gold_df = pd.read_csv(GOLD58_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"].dropna().unique())
    print(f"Loaded Gold-58 cohort: {len(gold_uids)} studies.")

    # 2. Load production labels and extract non-Gold studies
    labels_df = pd.read_csv(LABELS_CSV)
    all_studies = set(labels_df["StudyInstanceUID"].dropna().unique())
    dev_uids = sorted(list(all_studies - gold_uids))
    print(f"Total non-Gold development studies: {len(dev_uids)}")
    assert len(dev_uids) == 4349, f"Expected 4,349 non-Gold studies, got {len(dev_uids)}"

    # 3. Verify all development studies exist in processed directory
    on_disk_dirs = set(os.listdir(PROCESSED_DIR))
    missing_on_disk = set(dev_uids) - on_disk_dirs
    assert len(missing_on_disk) == 0, f"Missing {len(missing_on_disk)} studies in processed directory!"

    # 4. Stratification based on positive abnormality count per study
    dev_labels = labels_df[labels_df["StudyInstanceUID"].isin(dev_uids)]
    study_pos_counts = dev_labels.groupby("StudyInstanceUID")["final_predicted_label"].sum().to_dict()

    # Bin positive counts to [0, 1, 2, 3, 4, 5+]
    strat_labels = np.array([min(study_pos_counts.get(uid, 0), 5) for uid in dev_uids])
    print("Stratification distribution across non-Gold development studies:")
    unique, counts = np.unique(strat_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Positive pathologies {u}: {c} studies ({c / len(dev_uids) * 100:.1f}%)")

    # 5. Pure NumPy Stratified K-Fold
    rng = np.random.RandomState(seed)
    dev_array = np.array(dev_uids)
    fold_val_indices = [[] for _ in range(num_folds)]

    # Partition each stratum independently across the K folds
    for u in unique:
        stratum_indices = np.where(strat_labels == u)[0]
        rng.shuffle(stratum_indices)
        for i, idx in enumerate(stratum_indices):
            fold_val_indices[i % num_folds].append(idx)

    folds_data = {}
    seen_val_uids = set()

    for fold_idx in range(num_folds):
        val_idx = np.array(sorted(fold_val_indices[fold_idx]))
        train_idx = np.setdiff1d(np.arange(len(dev_array)), val_idx)
        train_studies = sorted(dev_array[train_idx].tolist())
        val_studies = sorted(dev_array[val_idx].tolist())

        # Strict checks
        train_set = set(train_studies)
        val_set = set(val_studies)

        assert len(train_set & val_set) == 0, f"Train and Val overlap in fold {fold_idx}!"
        assert len(val_set & seen_val_uids) == 0, f"Val studies overlap across folds in fold {fold_idx}!"
        assert len(train_set & gold_uids) == 0, f"Gold-58 leakage in train split of fold {fold_idx}!"
        assert len(val_set & gold_uids) == 0, f"Gold-58 leakage in val split of fold {fold_idx}!"

        seen_val_uids.update(val_set)

        folds_data[f"fold_{fold_idx}"] = {
            "fold_index": fold_idx,
            "train_studies": train_studies,
            "val_studies": val_studies,
            "train_count": len(train_studies),
            "val_count": len(val_studies),
        }
        print(f"Fold {fold_idx}: Train = {len(train_studies)}, Val = {len(val_studies)}")

    assert len(seen_val_uids) == 4349, f"Expected 4349 total validated studies, got {len(seen_val_uids)}"

    output_payload = {
        "metadata": {
            "total_studies": len(dev_uids),
            "num_folds": num_folds,
            "seed": seed,
            "gold58_overlap": 0,
            "stratification": "positive_pathology_count_binned_5",
        },
        "folds": folds_data,
    }

    with open(out_path, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nCV Splits successfully generated and saved to: {out_path}")
    return output_payload


if __name__ == "__main__":
    generate_cv_splits()
