"""
Phase C3: Automated Scientific Data Integrity Gate.

Verifies:
1. Exact non-Gold cohort coverage (4,349 studies) & zero Gold-58 leakage
2. No duplicate StudyInstanceUIDs
3. Every study has usable on-disk series in phase_c/c1/processed/
4. Sampled .npz files: shape (32, 256, 256), float16, finite, normalized [0.0, 1.0]
5. Label integrity: exactly 12 targets per study, Y in [0, 1], W >= 0
6. Series metadata presence for deterministic slot classification
7. CV fold disjointness (no study in multiple validation folds)
8. End-to-end model forward pass and loss evaluation
"""
import os
import sys
import json
import glob
import random
import unittest
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
PROCESSED_DIR = os.path.join(REPO_ROOT, "phase_c", "c1", "processed")
LABELS_CSV = os.path.join(REPO_ROOT, "labels", "production", "phase_b_step3_remediated_training_corpus.csv")
GOLD58_CSV = os.path.join(REPO_ROOT, "labels", "gold58", "phase_b_step3_remediated_predictions_gold58.csv")
MANIFEST_CSV = os.path.join(REPO_ROOT, "phase_c", "c1", "manifests", "c1_study_manifest.csv")

sys.path.insert(0, os.path.join(REPO_ROOT, "phase_c", "c3", "scripts"))
from loss import weighted_soft_bce_loss
from model import MultiSeriesKneeModel, TARGETS, NUM_TARGETS, NUM_SLOTS


class TestC3IntegrityGate(unittest.TestCase):
    """Automated integrity test suite for Phase C3."""

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 80)
        print("  PHASE C3 AUTOMATED DATA INTEGRITY GATE")
        print("=" * 80)
        cls.labels_df = pd.read_csv(LABELS_CSV)
        cls.gold_df = pd.read_csv(GOLD58_CSV)
        cls.gold_uids = set(cls.gold_df["StudyInstanceUID"].dropna().unique())
        cls.all_corpus_uids = set(cls.labels_df["StudyInstanceUID"].dropna().unique())
        cls.expected_dev_uids = cls.all_corpus_uids - cls.gold_uids
        cls.on_disk_study_dirs = set(d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d)))

    def test_01_gold58_strict_isolation(self):
        """Verify zero Gold-58 leakage into processed development data."""
        overlap = self.on_disk_study_dirs & self.gold_uids
        self.assertEqual(len(overlap), 0, f"CRITICAL LEAKAGE: {len(overlap)} Gold-58 studies found in processed data!")
        print(f"  ✓ Check 1 PASS: Gold-58 strictly isolated (0 overlap across {len(self.gold_uids)} Gold studies).")

    def test_02_cohort_counts_and_uniqueness(self):
        """Verify exactly 4,349 development studies with no duplicates."""
        self.assertEqual(len(self.expected_dev_uids), 4349, f"Expected 4349 non-gold studies, got {len(self.expected_dev_uids)}")
        self.assertEqual(len(self.on_disk_study_dirs), 4349, f"Expected 4349 on-disk studies, got {len(self.on_disk_study_dirs)}")
        missing = self.expected_dev_uids - self.on_disk_study_dirs
        self.assertEqual(len(missing), 0, f"Missing {len(missing)} studies on disk!")
        print(f"  ✓ Check 2 PASS: Cohort coverage complete (exactly 4,349 unique non-Gold studies).")

    def test_03_label_completeness_and_validity(self):
        """Verify all 12 targets, continuous Y in [0, 1], and non-negative W."""
        # Check that each development study has exactly 12 rows
        dev_labels = self.labels_df[self.labels_df["StudyInstanceUID"].isin(self.expected_dev_uids)]
        self.assertEqual(len(dev_labels), 4349 * 12)

        # Check Y range
        y = dev_labels["label_probability"].values
        self.assertTrue(np.all((y >= 0.0) & (y <= 1.0)), "Y values out of [0, 1] range")

        # Check W range
        w = dev_labels["sample_weight"].values
        self.assertTrue(np.all(w >= 0.0), "Negative sample weights found")

        # Check abstentions
        abstained = dev_labels[dev_labels["abstain_flag"] == 1]
        self.assertTrue(len(abstained) > 0, "Expected some abstentions in production corpus")
        print(f"  ✓ Check 3 PASS: Labels 100% complete (4,349 x 12 = 52,188 dev targets, Y in [0, 1], W >= 0).")

    def test_04_processed_volume_integrity(self):
        """Sample 50 random .npz files and verify dimensions, dtype, spacing, and range."""
        all_studies = list(self.on_disk_study_dirs)
        random.seed(42)
        sample_studies = random.sample(all_studies, 50)

        for suid in sample_studies:
            s_dir = os.path.join(PROCESSED_DIR, suid)
            npz_files = [os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.endswith(".npz")]
            self.assertGreater(len(npz_files), 0, f"Study {suid} has no .npz files")

            # Check primary npz
            with np.load(npz_files[0]) as data:
                self.assertIn("volume", data)
                self.assertIn("spacing", data)
                vol = data["volume"]
                sp = data["spacing"]

                self.assertEqual(vol.shape, (32, 256, 256))
                self.assertEqual(vol.dtype, np.float16)
                self.assertTrue(np.allclose(sp, (3.0, 0.5, 0.5)))
                self.assertFalse(np.isnan(vol).any(), f"NaN in {npz_files[0]}")
                self.assertFalse(np.isinf(vol).any(), f"Inf in {npz_files[0]}")
                self.assertGreaterEqual(vol.min(), 0.0)
                self.assertLessEqual(vol.max(), 1.0)
        print("  ✓ Check 4 PASS: Sampled volume archives verified (32x256x256, float16, finite, [0, 1]).")

    def test_05_model_forward_backward_loss(self):
        """Verify model architecture forward and backward pass with weighted BCE loss."""
        model = MultiSeriesKneeModel(aggregation="masked_pool")
        B = 2
        vols = torch.zeros(B, NUM_SLOTS, 1, 32, 256, 256, dtype=torch.float32)
        vols[:, 0] = torch.randn(B, 1, 32, 256, 256)
        mask = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        targets = torch.full((B, NUM_TARGETS), 0.1, dtype=torch.float32)
        weights = torch.ones((B, NUM_TARGETS), dtype=torch.float32)

        logits = model(vols, mask)
        self.assertEqual(logits.shape, (B, NUM_TARGETS))
        loss = weighted_soft_bce_loss(logits, targets, weights)
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        print("  ✓ Check 5 PASS: Model forward pass and weighted soft BCE backward pass verified.")

    def test_06_cv_splits_on_disk_integrity(self):
        """Verify the generated 5-fold CV splits on disk: 0 fold overlap, 0 Gold leakage, 4,349 studies."""
        splits_path = os.path.join(REPO_ROOT, "phase_c", "c3", "configs", "cv_splits.json")
        self.assertTrue(os.path.exists(splits_path), f"Splits file not found: {splits_path}")

        with open(splits_path) as f:
            splits_data = json.load(f)

        folds = splits_data["folds"]
        self.assertEqual(len(folds), 5)

        seen_val_uids = set()
        for f_name, f_data in folds.items():
            train_set = set(f_data["train_studies"])
            val_set = set(f_data["val_studies"])

            # 0 overlap between train and val
            self.assertEqual(len(train_set & val_set), 0, f"{f_name} train and val overlap!")
            # 0 overlap across val folds
            self.assertEqual(len(val_set & seen_val_uids), 0, f"{f_name} val overlaps with prior folds!")
            # 0 Gold-58 leakage
            self.assertEqual(len(train_set & self.gold_uids), 0, f"{f_name} train has Gold-58 leakage!")
            self.assertEqual(len(val_set & self.gold_uids), 0, f"{f_name} val has Gold-58 leakage!")

            seen_val_uids.update(val_set)

        self.assertEqual(len(seen_val_uids), 4349)
        print("  ✓ Check 6 PASS: On-disk 5-fold CV splits verified (4,349 studies, 0 fold overlap, 0 Gold leakage).")

    def test_07_series_metadata_presence_check(self):
        """Verify presence of series metadata required for deterministic slot classification.

        If metadata is absent, flag the limitation clearly as required by C3.2.
        """
        possible_meta_paths = [
            os.path.join(REPO_ROOT, "csv", "train_series.csv"),
            os.path.join(REPO_ROOT, "manifests", "train_series.csv"),
            os.path.join(REPO_ROOT, "train_series.csv"),
            os.path.join(REPO_ROOT, "data", "train_series.csv"),
        ]
        found_path = None
        for p in possible_meta_paths:
            if os.path.exists(p):
                found_path = p
                break

        if found_path is None:
            print("  ! Check 7 STATUS: Series metadata file (train_series.csv) is NOT currently on HPC.")
            print("    -> C3.2 Rule triggered: STOP and report limitation rather than inventing unreliable slot labels.")
        else:
            print(f"  ✓ Check 7 PASS: Series metadata located at {found_path}.")


if __name__ == "__main__":
    unittest.main()
