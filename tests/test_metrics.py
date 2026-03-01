"""
Unit tests for TED-LW metrics (v2 — interference-based).
Tests: ICV, IG, CSI, outlier pressure, LW v2, Wander Ratio.
"""

import numpy as np
import torch
import pytest

from metrics import (
    compute_icv,
    compute_interference,
    compute_csi,
    compute_outlier_pressure,
    compute_latent_work,
    compute_total_lw,
    compute_wander_ratio,
)


class TestICV:
    """Inter-Checkpoint Velocity tests."""

    def test_zero_displacement(self):
        v = torch.randn(128)
        assert compute_icv(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_known_displacement(self):
        a = torch.zeros(4); a[0] = 1.0
        b = torch.zeros(4); b[1] = 1.0
        assert compute_icv(a, b) == pytest.approx(np.sqrt(2), rel=1e-4)

    def test_scalar_output(self):
        assert isinstance(compute_icv(torch.randn(64), torch.randn(64)), float)


class TestInterference:
    """Interference Gauge (IG) tests."""

    def test_orthogonal_updates(self):
        """Orthogonal updates → IG ≈ 0."""
        a = torch.zeros(100); a[0] = 1.0
        b = torch.zeros(100); b[50] = 1.0
        assert compute_interference(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_parallel_updates(self):
        """Parallel updates → IG ≈ 1.0."""
        a = torch.randn(100)
        b = a * 2.0  # Same direction, different magnitude
        assert compute_interference(a, b) == pytest.approx(1.0, abs=1e-4)

    def test_antiparallel_clamped(self):
        """Antiparallel updates → clamped to 0 (we only track positive alignment)."""
        a = torch.randn(100)
        b = -a
        assert compute_interference(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector(self):
        a = torch.randn(100)
        b = torch.zeros(100)
        assert compute_interference(a, b) == pytest.approx(0.0)


class TestCSI:
    """Cross-Segment Interference tests."""

    def test_independent_segments(self):
        """Orthogonal early/late updates → CSI ≈ 0."""
        early = torch.zeros(100); early[0] = 1.0
        late = torch.zeros(100); late[50] = 1.0
        assert compute_csi(early, late) == pytest.approx(0.0, abs=1e-6)

    def test_redundant_segments(self):
        """Same direction early/late → CSI ≈ 1.0."""
        v = torch.randn(100)
        assert compute_csi(v, v * 3.0) == pytest.approx(1.0, abs=1e-4)


class TestOutlierPressure:
    """L∞ norm and outlier ratio tests."""

    def test_known_l_inf(self):
        x = torch.tensor([1.0, 2.0, 3.0, -5.0, 0.5])
        l_inf, _ = compute_outlier_pressure(x)
        assert l_inf == pytest.approx(5.0)

    def test_outlier_ratio(self):
        """One extreme outlier in 100 uniform values."""
        x = torch.ones(100)
        x[0] = 100.0  # Clear outlier
        _, ratio = compute_outlier_pressure(x, outlier_multiplier=3.0)
        assert ratio == pytest.approx(0.01, abs=0.02)  # ~1 out of 100

    def test_uniform_no_outliers(self):
        x = torch.ones(100)
        _, ratio = compute_outlier_pressure(x)
        assert ratio == pytest.approx(0.0)


class TestLatentWork:
    """Latent Work v2 tests."""

    def test_basic_lw(self):
        """LW = ICV × (1 + IG)."""
        assert compute_latent_work(10.0, 0.5) == pytest.approx(15.0)

    def test_zero_interference(self):
        """No interference → LW = ICV."""
        assert compute_latent_work(7.0, 0.0) == pytest.approx(7.0)

    def test_full_interference(self):
        """IG = 1.0 → LW = 2 × ICV."""
        assert compute_latent_work(5.0, 1.0) == pytest.approx(10.0)

    def test_total_lw(self):
        assert compute_total_lw([1.0, 2.0, 3.0]) == pytest.approx(6.0)

    def test_empty_total(self):
        assert compute_total_lw([]) == pytest.approx(0.0)


class TestWanderRatio:
    """Wander Ratio tests."""

    def test_straight_path(self):
        first = torch.tensor([0.0, 0.0])
        last = torch.tensor([3.0, 0.0])
        assert compute_wander_ratio(3.0, first, last) == pytest.approx(1.0)

    def test_backtracking(self):
        first = torch.tensor([0.0, 0.0])
        last = torch.tensor([1.0, 0.0])
        assert compute_wander_ratio(5.0, first, last) == pytest.approx(5.0)

    def test_zero_displacement(self):
        v = torch.tensor([1.0, 2.0])
        assert compute_wander_ratio(10.0, v, v) == float("inf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
