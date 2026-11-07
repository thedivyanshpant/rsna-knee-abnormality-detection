"""
Phase C2: Controlled four-way MRI preprocessing experiment.
Configuration module — single source of truth for all experimental parameters.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

# ============================================================================
# DYNAMIC PATHS
# ============================================================================
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
C2_ROOT = os.path.dirname(SCRIPTS_DIR)
PHASE_C_ROOT = os.path.dirname(C2_ROOT)
RSNA_CODE_ROOT = os.path.dirname(PHASE_C_ROOT)

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA_ROOT = os.environ.get("RSNA_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
TRAIN_SERIES_DIR = os.path.join(DATA_ROOT, "train_series")
TRAIN_SERIES_CSV = os.path.join(DATA_ROOT, "train_series.csv")
GOLD_STUDIES_CSV = os.path.join(DATA_ROOT, "gold_studies.csv")

C0_ROOT = os.path.join(PHASE_C_ROOT, "c0")
C0_PROCESSED = os.environ.get("RSNA_C0_PROCESSED", os.path.join(REPO_ROOT, "phase_c/c0/processed"))  # C0 pilot processed directory
C0_PILOT_CSV = os.path.join(C0_ROOT, "output", "pilot_selection.csv")

LABELS_CSV = os.path.join(RSNA_CODE_ROOT, "labels", "production", "phase_b_step3_remediated_training_corpus.csv")

CONFIGS_DIR = os.path.join(C2_ROOT, "configs")
LOGS_DIR = os.path.join(C2_ROOT, "logs")
CHECKPOINTS_DIR = os.path.join(C2_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(C2_ROOT, "results")

# ============================================================================
# TARGETS  (12 binary targets from Phase B)
# ============================================================================
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

# ============================================================================
# REPRESENTATION PIPELINES
# ============================================================================
PIPELINE_NAMES = ["raw", "pipeline_a", "pipeline_b", "pipeline_c"]

# Fixed spatial dimensions for model input (all pipelines resized to this)
MODEL_INPUT_SHAPE = (32, 256, 256)  # (D, H, W)

# ============================================================================
# EXPERIMENT HYPERPARAMETERS  (identical across all 4 runs)
# ============================================================================
SEED = 42
VAL_FRACTION = 0.2  # 80/20 stratified split of 80 pilot studies
EPOCHS = 30
BATCH_SIZE = 2  # 8 GB VRAM constraint with 3D volumes
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda:0"  # Quadro P4000 (8 GB)
PIN_MEMORY = True
MIXED_PRECISION = True  # Use AMP for VRAM efficiency

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
MODEL_NAME = "SmallResNet3D"
