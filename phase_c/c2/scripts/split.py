"""
Phase C2: Deterministic train/validation split with Gold-58 exclusion verification.

Creates an 80/20 stratified study-level split of the 80 C0 pilot studies.
Stratification is based on total positive label count per study to ensure
both splits see a representative mix of pathology prevalence.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    C0_PILOT_CSV, LABELS_CSV, GOLD_STUDIES_CSV,
    TARGETS, SEED, VAL_FRACTION, CONFIGS_DIR,
)


def create_split():
    """Create and save the deterministic train/val split."""
    # 1. Load pilot study UIDs
    pilot_df = pd.read_csv(C0_PILOT_CSV)
    pilot_uids = sorted(
        pilot_df[pilot_df["selected"] == True]["StudyInstanceUID"].tolist()
    )
    print(f"Pilot studies: {len(pilot_uids)}")

    # 2. Load Gold-58 and verify zero overlap
    gold_df = pd.read_csv(GOLD_STUDIES_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"])
    pilot_set = set(pilot_uids)
    overlap = pilot_set & gold_uids
    assert len(overlap) == 0, f"FATAL: {len(overlap)} Gold-58 studies in pilot set!"
    print(f"Gold-58 exclusion verified: 0 overlap with {len(gold_uids)} gold studies")

    # 3. Load labels for stratification
    labels_df = pd.read_csv(LABELS_CSV)
    pilot_labels = labels_df[labels_df["StudyInstanceUID"].isin(pilot_set)]

    # Compute total positive count per study for stratification
    study_pos_count = {}
    for uid in pilot_uids:
        study_rows = pilot_labels[pilot_labels["StudyInstanceUID"] == uid]
        pos_count = (study_rows["final_predicted_label"] == 1).sum()
        study_pos_count[uid] = pos_count

    # Create stratification bins (0, 1, 2, 3+ positives)
    strat_labels = np.array([min(study_pos_count.get(uid, 0), 3) for uid in pilot_uids])
    print(f"Stratification distribution: {dict(zip(*np.unique(strat_labels, return_counts=True)))}")

    # 4. Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    pilot_array = np.array(pilot_uids)
    train_idx, val_idx = next(sss.split(pilot_array, strat_labels))

    train_uids = sorted(pilot_array[train_idx].tolist())
    val_uids = sorted(pilot_array[val_idx].tolist())

    print(f"Train: {len(train_uids)} studies, Val: {len(val_uids)} studies")

    # 5. Verify no leakage
    assert len(set(train_uids) & set(val_uids)) == 0, "Train/val overlap!"
    assert len(set(train_uids) & gold_uids) == 0, "Train/gold overlap!"
    assert len(set(val_uids) & gold_uids) == 0, "Val/gold overlap!"

    # 6. Save split
    split_info = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "total_pilot_studies": len(pilot_uids),
        "train_count": len(train_uids),
        "val_count": len(val_uids),
        "gold58_excluded": len(gold_uids),
        "gold58_overlap_train": 0,
        "gold58_overlap_val": 0,
        "train_uids": train_uids,
        "val_uids": val_uids,
        "stratification_bins": {
            int(k): int(v) for k, v in zip(*np.unique(strat_labels, return_counts=True))
        },
    }

    os.makedirs(CONFIGS_DIR, exist_ok=True)
    split_path = os.path.join(CONFIGS_DIR, "train_val_split.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Split saved: {split_path}")

    return train_uids, val_uids


def load_split():
    """Load the saved train/val split."""
    split_path = os.path.join(CONFIGS_DIR, "train_val_split.json")
    with open(split_path) as f:
        split_info = json.load(f)
    return split_info["train_uids"], split_info["val_uids"]


if __name__ == "__main__":
    train_uids, val_uids = create_split()
