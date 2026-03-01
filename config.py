"""
TED-LW Pipeline Configuration
==============================
Model registry, hyperparameters, and output paths.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Specification for a single test-subject model."""
    name: str               # Short display name for CSV / plots
    hf_id: str              # HuggingFace model identifier
    is_dense: bool = False  # True = dense model, False = MoE / sparse
    backend: str = "vllm"   # "vllm" or "hf" (fallback for unsupported archs)


MODELS: Dict[str, ModelSpec] = {
    "Qwen3.5-35B-A3B": ModelSpec(
        name="Qwen3.5-35B-A3B",
        hf_id="Qwen/Qwen3.5-35B-A3B",
        is_dense=False,
        backend="vllm",
    ),
    "Nanbeige4.1-3B": ModelSpec(
        name="Nanbeige4.1-3B",
        hf_id="Nanbeige/Nanbeige4.1-3B",
        is_dense=True,
        backend="vllm",
    ),
    "Nemotron-3-Nano-30B-A3B": ModelSpec(
        name="Nemotron-3-Nano-30B-A3B",
        hf_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        is_dense=False,
        backend="vllm",
    ),
    "LFM2-24B-A2B": ModelSpec(
        name="LFM2-24B-A2B",
        hf_id="LiquidAI/LFM2-24B-A2B",
        is_dense=False,
        backend="vllm",
    ),
}


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Smoothing window for Interference Gauge rolling average
IG_SMOOTHING_WINDOW: int = 50

# Generation constraints
MAX_NEW_TOKENS: int = 8000
TEMPERATURE: float = 0.0          # Greedy decoding

# Batch size (will auto-halve on OOM)
MAX_BATCH_SIZE: int = 16

# Checkpoint fractions of total layer depth (3 sparse hooks)
CHECKPOINT_FRACTIONS: List[float] = [0.25, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRACES_DIR = os.path.join(RESULTS_DIR, "traces")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CSV_PATH = os.path.join(RESULTS_DIR, "limo_latent_work_results.csv")

# Dataset
DATASET_NAME = "GAIR/LIMO"
DATASET_SPLIT = "train"
