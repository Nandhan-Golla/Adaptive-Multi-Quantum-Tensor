"""
Standard quantum circuit builders.

Each function returns a list of (gate, qubit_indices) pairs that can be
applied to a QuantumTensor in sequence:

    for gate, qubits in ghz_circuit(n):
        state.apply(gate, *qubits)

All functions produce circuits that act on qubits 0..n-1.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple

from amqt.core.gates.standard import (
    Gate, H, X, CNOT, S, CZ, SWAP, RZ, RX,
)

Circuit = List[Tuple[Gate, Tuple[int, ...]]]


def ghz_circuit(n: int) -> Circuit:
    """GHZ state preparation: H on qubit 0, then CNOT(i, i+1) for i=0..n-2.

    Final state: (|0...0⟩ + |1...1⟩) / √2
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    ops: Circuit = [(H, (0,))]
    for i in range(n - 1):
        ops.append((CNOT, (i, i + 1)))
    return ops


def qft_circuit(n: int) -> Circuit:
    """Quantum Fourier Transform on n qubits (in-place, standard ordering).

    Implements the standard QFT decomposition:
      For each qubit k from 0 to n-1:
        - Apply H on qubit k
        - Apply controlled-Phase(π/2^j) with control k+j on qubit k, j=1..n-1-k
      Finally, apply SWAP pairs to reverse the output qubit order.

    This is an exact (non-approximate) implementation.  All gates are unitary.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    ops: Circuit = []
    for k in range(n):
        ops.append((H, (k,)))
        for j in range(1, n - k):
            angle = np.pi / (2 ** j)
            # Controlled-RZ: implement as CZ-like but we use the Phase gate
            # We use the fact that CPhase(θ) = diag(1,1,1,e^{iθ})
            # and apply it as a pair of RZ gates + CNOT (standard decomposition)
            # For simplicity we use a custom 2-qubit gate via the dense path
            cphase = _controlled_phase(angle)
            ops.append((_MatrixGate(cphase, 2, f"CP({angle:.3f})"), (k, k + j)))
    # Bit-reversal: swap qubits 0..n//2-1 with n-1..n//2
    for i in range(n // 2):
        ops.append((SWAP, (i, n - 1 - i)))
    return ops


def random_clifford_circuit(n: int, depth: int, seed: int = 0) -> Circuit:
    """Random Clifford circuit of given depth on n qubits.

    Each layer randomly applies single-qubit Clifford gates (H, S, X) and
    two-qubit CNOT gates on randomly chosen pairs.  The resulting circuit
    is exactly Clifford (no T or rotation gates), making it ideal for
    benchmarking the Stabilizer representation.

    Parameters
    ----------
    n:
        Number of qubits.
    depth:
        Number of layers.  Each layer applies n single-qubit gates and
        n//2 CNOT gates.
    seed:
        Random seed for reproducibility.
    """
    if n < 1 or depth < 0:
        raise ValueError("n >= 1 and depth >= 0 required")
    rng = np.random.default_rng(seed)
    single_gates = [H, S, X]
    ops: Circuit = []
    for _ in range(depth):
        # Single-qubit layer
        for q in range(n):
            g = single_gates[rng.integers(3)]
            ops.append((g, (q,)))
        # Two-qubit layer (random non-overlapping pairs)
        qubits = list(rng.permutation(n))
        for i in range(0, len(qubits) - 1, 2):
            ops.append((CNOT, (qubits[i], qubits[i + 1])))
    return ops


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _controlled_phase(theta: float) -> np.ndarray:
    """Return the 4×4 controlled-Phase(theta) matrix."""
    mat = np.eye(4, dtype=np.complex128)
    mat[3, 3] = np.exp(1j * theta)
    return mat


class _MatrixGate(Gate):
    """Inline gate with an arbitrary matrix — used internally by circuit builders."""

    def __init__(self, mat: np.ndarray, n_qubits: int, name: str = "MatGate") -> None:
        self._mat = np.asarray(mat, dtype=np.complex128)
        self.n_qubits = n_qubits
        self.name = name

    @property
    def matrix(self) -> np.ndarray:
        return self._mat.copy()
