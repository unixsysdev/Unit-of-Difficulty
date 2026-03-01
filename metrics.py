"""
TED-LW Metrics Module (v2 — Post-SVD Fix)
==========================================
Interference-based metrics for measuring residual stream saturation:
  - Inter-Checkpoint Velocity (ICV)
  - Interference Gauge (IG) — cosine similarity of consecutive updates
  - Cross-Segment Interference (CSI) — cosine similarity between model halves
  - Outlier Pressure (L∞ norm + outlier ratio)
  - Latent Work v2: ICV × (1 + IG)
  - Wander Ratio

All functions operate on CPU tensors/numpy arrays to prevent GPU memory leaks.
"""

import numpy as np
import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# Inter-Checkpoint Velocity (ICV)
# ---------------------------------------------------------------------------

def compute_icv(x_final: torch.Tensor, x_early: torch.Tensor) -> float:
    """
    Compute Inter-Checkpoint Velocity: ||x^(final) - x^(early)||₂

    Measures total residual displacement through the model at a single
    token step. Uses only 2 of the 3 hook points.

    Args:
        x_final: Residual stream at the final layer,  shape [d_model]
        x_early: Residual stream at the early layer,  shape [d_model]

    Returns:
        L2 norm of the difference (scalar float).
    """
    diff = x_final.float() - x_early.float()
    return torch.norm(diff, p=2).item()


# ---------------------------------------------------------------------------
# Interference Gauge (IG) — The core replacement for TED
# ---------------------------------------------------------------------------

def compute_interference(
    delta_current: torch.Tensor,
    delta_previous: torch.Tensor,
) -> float:
    """
    Compute Interference Gauge: cosine similarity between consecutive
    token-step update vectors.

    Update vector: Δ_t = x_t^(final) - x_t^(early)

    When IG ≈ 0: updates are orthogonal → model has room to maneuver.
    When IG → 1.0: updates collapse into a narrow cone → interference.

    Args:
        delta_current:  Update vector at step t,   shape [d_model]
        delta_previous: Update vector at step t-1, shape [d_model]

    Returns:
        Cosine similarity in [-1, 1]. Clamped to [0, 1] for the gauge
        since we only care about positive alignment (collapse).
    """
    a = delta_current.float()
    b = delta_previous.float()

    norm_a = torch.norm(a, p=2)
    norm_b = torch.norm(b, p=2)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    cos_sim = torch.dot(a, b) / (norm_a * norm_b)
    # Clamp to [0, 1] — we only care about the degree of alignment
    return max(0.0, cos_sim.item())


# ---------------------------------------------------------------------------
# Cross-Segment Interference (CSI)
# ---------------------------------------------------------------------------

def compute_csi(
    delta_early_to_mid: torch.Tensor,
    delta_mid_to_final: torch.Tensor,
) -> float:
    """
    Compute Cross-Segment Interference: cosine similarity between the
    early→mid and mid→final update segments at the same token step.

    When CSI → 1.0: early and late processing are stuck in the same
    subspace — the model is just repeating the same transformation.

    Args:
        delta_early_to_mid:  x^(mid) - x^(early),  shape [d_model]
        delta_mid_to_final:  x^(final) - x^(mid),   shape [d_model]

    Returns:
        Cosine similarity in [0, 1].
    """
    a = delta_early_to_mid.float()
    b = delta_mid_to_final.float()

    norm_a = torch.norm(a, p=2)
    norm_b = torch.norm(b, p=2)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    cos_sim = torch.dot(a, b) / (norm_a * norm_b)
    return max(0.0, cos_sim.item())


# ---------------------------------------------------------------------------
# Outlier Pressure (L∞ + Outlier Ratio)
# ---------------------------------------------------------------------------

def compute_outlier_pressure(
    residual: torch.Tensor,
    outlier_multiplier: float = 3.0,
) -> Tuple[float, float]:
    """
    Compute outlier dimension pressure from the residual stream.

    Transformers route excess capacity into "outlier dimensions" —
    specific coordinates with extreme magnitudes that serve as
    global memory aggregators. When these explode, the model's
    representational buffer is saturated.

    Args:
        residual:            Residual stream vector, shape [d_model]
        outlier_multiplier:  Threshold multiplier over median (default 3×)

    Returns:
        (l_inf, outlier_ratio):
            l_inf: Maximum absolute value of any coordinate.
            outlier_ratio: Fraction of coordinates exceeding
                           outlier_multiplier × median(|x|).
    """
    x = residual.float().abs()
    l_inf = x.max().item()

    median_val = x.median().item()
    if median_val < 1e-12:
        # All values near zero — no outliers
        return l_inf, 0.0

    threshold = outlier_multiplier * median_val
    outlier_ratio = (x > threshold).float().mean().item()

    return l_inf, outlier_ratio


# ---------------------------------------------------------------------------
# Latent Work v2
# ---------------------------------------------------------------------------

def compute_latent_work(icv: float, ig: float) -> float:
    """
    Compute Latent Work for a single token step (revised formula):
        LW_t = ICV_t × (1 + IG_t)

    When the model works hard (high ICV) AND its updates are collapsing
    into a narrow cone (high IG), latent work spikes.

    Args:
        icv: Inter-Checkpoint Velocity at step t.
        ig:  Interference Gauge at step t.

    Returns:
        Latent Work scalar (float).
    """
    return icv * (1.0 + ig)


def compute_total_lw(lw_per_token: list) -> float:
    """
    Compute Total Latent Work across the entire generation:
        Total_LW = Σ_t LW_t

    Args:
        lw_per_token: list of per-token LW values.

    Returns:
        Total Latent Work (float).
    """
    return sum(lw_per_token)


# ---------------------------------------------------------------------------
# Wander Ratio
# ---------------------------------------------------------------------------

def compute_wander_ratio(
    velocity_sum: float,
    first_residual: torch.Tensor,
    last_residual: torch.Tensor,
) -> float:
    """
    Compute Wander Ratio: path length / displacement.

        Wander = (Σ ||x_t - x_{t-1}||₂) / ||x_N - x_0||₂

    A Wander Ratio of 1.0 means a perfectly straight path.
    Higher values indicate wandering / backtracking.

    Args:
        velocity_sum:   Sum of token-to-token L2 norms at the final layer.
        first_residual: Final-layer residual at t=0, shape [d_model].
        last_residual:  Final-layer residual at t=N, shape [d_model].

    Returns:
        Wander ratio (float). Returns float('inf') if displacement is ~0.
    """
    displacement = torch.norm(
        last_residual.float() - first_residual.float(), p=2
    ).item()

    if displacement < 1e-12:
        return float("inf")

    return velocity_sum / displacement
