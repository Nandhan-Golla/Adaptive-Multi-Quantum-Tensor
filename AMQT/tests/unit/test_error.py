"""
Unit tests for the error/fidelity module.

Tests cover:
- state_fidelity: identical states, orthogonal states, superpositions
- trace_distance: relationship F + T² = 1 for pure states
- bures_distance: range and relationship to fidelity
- fidelity_from_density_matrices: pure-state reduction
- ErrorBudget: accumulation, tolerance, reset
"""
from __future__ import annotations

import numpy as np
import pytest

from amqt.error.fidelity import (
    state_fidelity,
    trace_distance,
    bures_distance,
    infidelity,
    fidelity_from_density_matrices,
)
from amqt.error.propagation import ErrorBudget

_EPS = 1e-10


# ---------------------------------------------------------------------------
# state_fidelity
# ---------------------------------------------------------------------------

class TestStateFidelity:
    def test_identical_states_fidelity_1(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        assert abs(state_fidelity(sv, sv) - 1.0) < _EPS

    def test_orthogonal_states_fidelity_0(self):
        s0 = np.array([1.0, 0.0], dtype=complex)
        s1 = np.array([0.0, 1.0], dtype=complex)
        assert abs(state_fidelity(s0, s1)) < _EPS

    def test_bell_vs_product(self):
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        prod = np.array([1, 0, 0, 0], dtype=complex)
        f = state_fidelity(bell, prod)
        assert abs(f - 0.5) < _EPS

    def test_global_phase_invariance(self):
        sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
        sv_phased = sv * np.exp(1j * np.pi / 3)
        assert abs(state_fidelity(sv, sv_phased) - 1.0) < _EPS

    def test_fidelity_range(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            a = rng.standard_normal(8) + 1j * rng.standard_normal(8)
            b = rng.standard_normal(8) + 1j * rng.standard_normal(8)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            f = state_fidelity(a, b)
            assert 0.0 <= f <= 1.0 + _EPS

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            state_fidelity(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    def test_zero_norm_raises(self):
        with pytest.raises(ValueError):
            state_fidelity(np.zeros(2, dtype=complex), np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# trace_distance
# ---------------------------------------------------------------------------

class TestTraceDistance:
    def test_identical_states_distance_0(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        assert abs(trace_distance(sv, sv)) < _EPS

    def test_orthogonal_states_distance_1(self):
        s0 = np.array([1.0, 0.0], dtype=complex)
        s1 = np.array([0.0, 1.0], dtype=complex)
        assert abs(trace_distance(s0, s1) - 1.0) < _EPS

    def test_trace_distance_fidelity_relation(self):
        """T² + F = 1 for pure states."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            b = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            F = state_fidelity(a, b)
            T = trace_distance(a, b)
            assert abs(T ** 2 + F - 1.0) < 1e-9

    def test_range(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            a = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            b = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            T = trace_distance(a, b)
            assert 0.0 <= T <= 1.0 + _EPS


# ---------------------------------------------------------------------------
# bures_distance
# ---------------------------------------------------------------------------

class TestBuresDistance:
    def test_identical_states(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        assert abs(bures_distance(sv, sv)) < _EPS

    def test_orthogonal_states(self):
        s0 = np.array([1.0, 0.0], dtype=complex)
        s1 = np.array([0.0, 1.0], dtype=complex)
        # d_B = sqrt(2 - 2*sqrt(0)) = sqrt(2)
        assert abs(bures_distance(s0, s1) - np.sqrt(2)) < _EPS

    def test_range(self):
        rng = np.random.default_rng(2)
        for _ in range(10):
            a = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            b = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            d = bures_distance(a, b)
            assert 0.0 <= d <= np.sqrt(2) + _EPS

    def test_bures_fidelity_relation(self):
        """d_B² = 2 - 2√F for pure states."""
        rng = np.random.default_rng(3)
        for _ in range(10):
            a = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            b = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            F = state_fidelity(a, b)
            d = bures_distance(a, b)
            assert abs(d ** 2 - (2 - 2 * np.sqrt(F))) < 1e-9


# ---------------------------------------------------------------------------
# fidelity_from_density_matrices
# ---------------------------------------------------------------------------

class TestDensityMatrixFidelity:
    def test_pure_state_consistency(self):
        """Density-matrix fidelity agrees with statevector fidelity for pure states."""
        sv1 = np.array([1, 1], dtype=complex) / np.sqrt(2)
        sv2 = np.array([1, 0], dtype=complex)
        rho1 = np.outer(sv1, sv1.conj())
        rho2 = np.outer(sv2, sv2.conj())
        F_sv = state_fidelity(sv1, sv2)
        F_dm = fidelity_from_density_matrices(rho1, rho2)
        assert abs(F_sv - F_dm) < 1e-9

    def test_identical_density_matrices(self):
        sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(sv, sv.conj())
        assert abs(fidelity_from_density_matrices(rho, rho) - 1.0) < 1e-9

    def test_orthogonal_density_matrices(self):
        rho0 = np.diag([1.0, 0.0]).astype(complex)
        rho1 = np.diag([0.0, 1.0]).astype(complex)
        assert abs(fidelity_from_density_matrices(rho0, rho1)) < 1e-9

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            fidelity_from_density_matrices(np.eye(2), np.eye(4))


# ---------------------------------------------------------------------------
# ErrorBudget
# ---------------------------------------------------------------------------

class TestErrorBudget:
    def test_initial_state(self):
        budget = ErrorBudget(tolerance=1e-3)
        assert budget.cumulative == 0.0
        assert budget.is_within_budget is True
        assert budget.n_steps == 0

    def test_record_accumulates(self):
        budget = ErrorBudget(tolerance=1e-2)
        budget.record(1e-4, source="svd_truncation")
        budget.record(2e-4)
        assert abs(budget.cumulative - 3e-4) < 1e-15
        assert budget.n_steps == 2
        assert budget.is_within_budget is True

    def test_exceeds_tolerance(self):
        budget = ErrorBudget(tolerance=1e-3)
        budget.record(2e-3, source="mps_trunc")
        assert budget.is_within_budget is False

    def test_reset(self):
        budget = ErrorBudget(tolerance=1e-3)
        budget.record(5e-3)
        budget.reset()
        assert budget.cumulative == 0.0
        assert budget.n_steps == 0
        assert budget.is_within_budget is True

    def test_history_recorded(self):
        budget = ErrorBudget()
        budget.record(1e-5, source="gate_a")
        budget.record(2e-5, source="gate_b")
        hist = budget.history
        assert len(hist) == 2
        assert hist[0].source == "gate_a"
        assert abs(hist[1].cumulative - 3e-5) < 1e-20

    def test_negative_error_raises(self):
        budget = ErrorBudget()
        with pytest.raises(ValueError):
            budget.record(-1e-5)

    def test_summary_str(self):
        budget = ErrorBudget(tolerance=1e-2)
        budget.record(1e-4, source="test")
        summary = budget.summary()
        assert "cumulative" in summary
        assert "within_budget" in summary

    def test_mps_error_integration(self):
        """Verify that MPS approximation_error feeds cleanly into ErrorBudget."""
        from amqt import QuantumTensor, H, CNOT
        state = QuantumTensor(5, initial_representation="mps", auto_switch=False)
        budget = ErrorBudget(tolerance=1e-3)
        for i in range(4):
            state.apply(H, i)
            state.apply(CNOT, i, i + 1)
            meta = state.metadata
            if meta.approximation_error > 0:
                budget.record(meta.approximation_error, source=f"gate_{i}")
        # No assertion on budget value (depends on circuit), but should not crash
        assert budget.cumulative >= 0.0
