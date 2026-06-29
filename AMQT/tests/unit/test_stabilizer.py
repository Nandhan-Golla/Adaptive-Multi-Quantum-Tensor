"""
Unit tests for StabilizerRepresentation.

Tests verify:
- Clifford gate correctness vs Dense (Bell, GHZ, Hadamard, CNOT, CZ, SWAP)
- Norm preservation through Clifford circuits
- Statevector agreement with Dense for all supported Clifford gates
- Non-Clifford fallback (T gate) gives correct result
- Metadata reports stabilizer type and O(n²) memory
- Representation consistency via QuantumTensor API
"""
from __future__ import annotations

import numpy as np
import pytest

from amqt import (
    QuantumTensor,
    H, X, Y, Z, S, T, I, CNOT, CZ, SWAP,
    DenseRepresentation,
    StabilizerRepresentation,
)

_EPS = 1e-10
_CLIFFORD_EPS = 1e-10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dense_sv(n: int, ops) -> np.ndarray:
    """Run ops on a fresh DenseRepresentation and return the statevector."""
    rep = DenseRepresentation(n)
    for gate, qubits in ops:
        rep.apply_gate(gate.matrix, list(qubits))
    return rep.get_statevector()


def _stab_sv(n: int, ops) -> np.ndarray:
    rep = StabilizerRepresentation(n)
    for gate, qubits in ops:
        rep.apply_gate(gate.matrix, list(qubits))
    return rep.get_statevector()


def _run_both(n: int, ops):
    return _dense_sv(n, ops), _stab_sv(n, ops)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInitialisation:
    def test_zero_state_1qubit(self):
        rep = StabilizerRepresentation(1)
        sv = rep.get_statevector()
        assert sv.shape == (2,)
        assert abs(sv[0] - 1.0) < _EPS

    def test_zero_state_4qubit(self):
        rep = StabilizerRepresentation(4)
        sv = rep.get_statevector()
        assert abs(sv[0] - 1.0) < _EPS
        assert np.allclose(sv[1:], 0, atol=_EPS)

    def test_norm_at_init(self):
        for n in [1, 2, 3, 5]:
            rep = StabilizerRepresentation(n)
            assert abs(np.linalg.norm(rep.get_statevector()) - 1.0) < _EPS

    def test_is_clifford_only_at_init(self):
        rep = StabilizerRepresentation(3)
        assert rep.is_clifford_only is True

    def test_metadata_type(self):
        rep = StabilizerRepresentation(4)
        meta = rep.get_metadata()
        assert meta.representation_name == "stabilizer"
        assert meta.n_qubits == 4
        assert meta.approximation_error == 0.0

    def test_memory_is_polynomial(self):
        # Stabilizer memory should be O(n²), far less than O(2^n) for large n
        for n in [4, 8, 16]:
            rep = StabilizerRepresentation(n)
            meta = rep.get_metadata()
            # Dense would need 2^n * 16 bytes; stabilizer uses ~(2n)*(2n+1) bytes
            dense_bytes = (1 << n) * 16
            assert meta.memory_bytes < dense_bytes, (
                f"n={n}: stabilizer ({meta.memory_bytes}B) >= dense ({dense_bytes}B)"
            )


# ---------------------------------------------------------------------------
# Single-qubit Clifford gates
# ---------------------------------------------------------------------------

class TestSingleQubitCliffordGates:
    def test_H(self):
        ops = [(H, (0,))]
        d, s = _run_both(1, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)

    def test_X(self):
        ops = [(X, (0,))]
        d, s = _run_both(1, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)

    def test_Y(self):
        ops = [(Y, (0,))]
        d, s = _run_both(1, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)

    def test_Z(self):
        ops = [(H, (0,)), (Z, (0,))]
        d, s = _run_both(1, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)

    def test_S(self):
        ops = [(H, (0,)), (S, (0,))]
        d, s = _run_both(1, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)

    def test_H_squared_identity(self):
        ops = [(H, (0,)), (H, (0,))]
        d, s = _run_both(1, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)
        # Should recover |0>
        assert abs(s[0] - 1.0) < _CLIFFORD_EPS

    def test_X_squared_identity(self):
        ops = [(X, (0,)), (X, (0,))]
        _, s = _run_both(1, ops)
        assert abs(s[0] - 1.0) < _CLIFFORD_EPS

    def test_S4_identity(self):
        ops = [(S, (0,))] * 4  # S^4 = I
        _, s = _run_both(1, ops)
        assert abs(s[0] - 1.0) < _CLIFFORD_EPS

    def test_Z_equals_S_squared(self):
        ops_z = [(H, (0,)), (Z, (0,))]
        ops_ss = [(H, (0,)), (S, (0,)), (S, (0,))]
        sz = _stab_sv(1, ops_z)
        sss = _stab_sv(1, ops_ss)
        assert np.allclose(sz, sss, atol=_CLIFFORD_EPS)

    def test_norm_preserved_single_qubit(self):
        n = 3
        ops = [(H, (0,)), (S, (1,)), (X, (2,)), (Y, (0,)), (Z, (1,)), (H, (2,))]
        rep = StabilizerRepresentation(n)
        for gate, qubits in ops:
            rep.apply_gate(gate.matrix, list(qubits))
            sv = rep.get_statevector()
            assert abs(np.linalg.norm(sv) - 1.0) < _CLIFFORD_EPS


# ---------------------------------------------------------------------------
# Two-qubit Clifford gates
# ---------------------------------------------------------------------------

class TestTwoQubitCliffordGates:
    def test_bell_state(self):
        ops = [(H, (0,)), (CNOT, (0, 1))]
        d, s = _run_both(2, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)
        # |00> + |11> / sqrt(2)
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[3] = 1 / np.sqrt(2)
        assert np.allclose(s, expected, atol=_CLIFFORD_EPS)

    def test_CNOT_on_zero(self):
        ops = [(CNOT, (0, 1))]
        d, s = _run_both(2, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)

    def test_CNOT_flips_target(self):
        ops = [(X, (0,)), (CNOT, (0, 1))]
        d, s = _run_both(2, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)
        # |10> -> |11>: index 3
        assert abs(s[3] - 1.0) < _CLIFFORD_EPS

    def test_CZ(self):
        ops = [(H, (0,)), (H, (1,)), (CZ, (0, 1))]
        d, s = _run_both(2, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)

    def test_SWAP(self):
        ops = [(X, (0,)), (SWAP, (0, 1))]
        d, s = _run_both(2, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)
        # X|0>⊗|0> = |10>; SWAP = |01> = index 1
        assert abs(s[1] - 1.0) < _CLIFFORD_EPS

    def test_ghz_3qubit(self):
        n = 3
        ops = [(H, (0,)), (CNOT, (0, 1)), (CNOT, (1, 2))]
        d, s = _run_both(n, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)
        # GHZ: (|000> + |111>) / sqrt(2) -> indices 0 and 7
        assert abs(s[0] - 1 / np.sqrt(2)) < _CLIFFORD_EPS
        assert abs(s[7] - 1 / np.sqrt(2)) < _CLIFFORD_EPS

    def test_ghz_5qubit(self):
        n = 5
        ops = [(H, (0,))] + [(CNOT, (i, i + 1)) for i in range(n - 1)]
        d, s = _run_both(n, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)


# ---------------------------------------------------------------------------
# Random Clifford circuits
# ---------------------------------------------------------------------------

class TestRandomClifford:
    def test_random_clifford_vs_dense_n4_d20(self):
        """Random 4-qubit Clifford circuit, depth 20 — stabilizer must match dense."""
        from amqt.utils.circuit import random_clifford_circuit
        n, depth = 4, 20
        circuit = random_clifford_circuit(n, depth=depth, seed=7)
        d = _dense_sv(n, circuit)
        s = _stab_sv(n, circuit)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS), (
            f"Max diff: {np.max(np.abs(d - s)):.2e}"
        )

    def test_random_clifford_norm_preserved(self):
        from amqt.utils.circuit import random_clifford_circuit
        n, depth = 5, 15
        circuit = random_clifford_circuit(n, depth=depth, seed=99)
        rep = StabilizerRepresentation(n)
        for gate, qubits in circuit:
            rep.apply_gate(gate.matrix, list(qubits))
        sv = rep.get_statevector()
        assert abs(np.linalg.norm(sv) - 1.0) < _CLIFFORD_EPS

    def test_clifford_only_flag_stays_true(self):
        from amqt.utils.circuit import random_clifford_circuit
        circuit = random_clifford_circuit(4, depth=10, seed=0)
        rep = StabilizerRepresentation(4)
        for gate, qubits in circuit:
            rep.apply_gate(gate.matrix, list(qubits))
        assert rep.is_clifford_only is True


# ---------------------------------------------------------------------------
# Non-Clifford fallback
# ---------------------------------------------------------------------------

class TestNonCliffordFallback:
    def test_T_gate_fallback_correct(self):
        """T gate triggers Dense fallback; result must still match Dense."""
        ops = [(H, (0,)), (T, (0,))]
        d, s = _run_both(1, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)

    def test_clifford_flag_cleared_after_T(self):
        rep = StabilizerRepresentation(1)
        rep.apply_gate(H.matrix, [0])
        assert rep.is_clifford_only is True
        rep.apply_gate(T.matrix, [0])
        assert rep.is_clifford_only is False

    def test_gates_after_T_still_correct(self):
        """After T-gate fallback, subsequent gates must still be applied correctly."""
        ops = [(H, (0,)), (T, (0,)), (H, (0,)), (X, (1,)), (CNOT, (0, 1))]
        d, s = _run_both(2, ops)
        assert np.allclose(d, s, atol=_CLIFFORD_EPS)


# ---------------------------------------------------------------------------
# QuantumTensor API integration
# ---------------------------------------------------------------------------

class TestQuantumTensorIntegration:
    def test_stabilizer_rep_via_api(self):
        state = QuantumTensor(2, initial_representation="stabilizer", auto_switch=False)
        state.apply(H, 0).apply(CNOT, 0, 1)
        sv = state.statevector()
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[3] = 1 / np.sqrt(2)
        assert np.allclose(sv, expected, atol=_CLIFFORD_EPS)

    def test_representation_name(self):
        state = QuantumTensor(3, initial_representation="stabilizer", auto_switch=False)
        assert state.representation_name == "stabilizer"

    def test_chaining(self):
        state = QuantumTensor(2, initial_representation="stabilizer")
        result = state.apply(H, 0).apply(CNOT, 0, 1)
        assert result is state

    def test_norm_via_api(self):
        state = QuantumTensor(4, initial_representation="stabilizer", auto_switch=False)
        for i in range(4):
            state.apply(H, i)
        for i in range(3):
            state.apply(CNOT, i, i + 1)
        assert abs(state.norm() - 1.0) < _CLIFFORD_EPS
