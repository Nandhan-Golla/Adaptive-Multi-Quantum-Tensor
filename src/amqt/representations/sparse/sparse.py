"""
Sparse statevector representation.

Stores only the amplitudes whose magnitude exceeds a configurable
threshold.  Ideal for states with many zero amplitudes (e.g., early
in a circuit before entanglement builds up, or stabilizer-like states).

Memory: O(nnz) where nnz = number of non-zero amplitudes.
Gate cost: O(2^k * nnz) per k-qubit gate — can be much cheaper than
dense when nnz << 2^n, but degrades to dense-like cost for entangled states.
"""
from __future__ import annotations

from typing import Sequence, Dict

import numpy as np

from amqt.representations.base import QuantumRepresentation
from amqt.core.state.metadata import StateMetadata

_DEFAULT_THRESHOLD = 1e-10


class SparseRepresentation(QuantumRepresentation):
    """Quantum state stored as a dict mapping basis index → complex amplitude."""

    def __init__(
        self,
        n_qubits: int,
        amplitudes: Dict[int, complex] | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._n = n_qubits
        self._threshold = threshold
        if amplitudes is None:
            # |0...0>
            self._data: Dict[int, complex] = {0: complex(1.0, 0.0)}
        else:
            self._data = {
                k: complex(v)
                for k, v in amplitudes.items()
                if abs(v) > threshold
            }

    # ------------------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def nnz(self) -> int:
        return len(self._data)

    def apply_gate(self, gate_matrix: np.ndarray, target_qubits: Sequence[int]) -> None:
        """Apply gate via explicit enumeration of non-zero amplitudes.

        For each non-zero basis state |b>, the gate maps it to a
        superposition of 2^k new basis states (one per output of the gate
        column).  We accumulate contributions into a new dict.
        """
        k = len(target_qubits)
        gate = np.asarray(gate_matrix, dtype=np.complex128).reshape(1 << k, 1 << k)
        targets = list(target_qubits)
        n = self._n

        new_data: Dict[int, complex] = {}

        for basis_idx, amp in self._data.items():
            if abs(amp) <= self._threshold:
                continue
            # Extract the bits at target positions
            target_bits = _extract_bits(basis_idx, targets, n)
            # Compute gate output column index
            # gate[:, target_bits] gives the output amplitudes
            col = gate[:, target_bits]           # shape (2^k,)
            # Each output row corresponds to a new combination of target bits
            for out_row in range(1 << k):
                new_amp = col[out_row] * amp
                if abs(new_amp) <= self._threshold:
                    continue
                new_basis = _set_bits(basis_idx, targets, out_row, n)
                if new_basis in new_data:
                    new_data[new_basis] += new_amp
                else:
                    new_data[new_basis] = new_amp

        # Prune near-zero entries after accumulation
        self._data = {
            k: v for k, v in new_data.items() if abs(v) > self._threshold
        }

    def get_statevector(self) -> np.ndarray:
        sv = np.zeros(1 << self._n, dtype=np.complex128)
        for idx, amp in self._data.items():
            sv[idx] = amp
        return sv

    def get_metadata(self) -> StateMetadata:
        dim = 1 << self._n
        nnz = self.nnz
        sparsity = 1.0 - nnz / dim
        sv = self.get_statevector()
        entropy = _entropy_estimate(sv, self._n)
        mem = nnz * (8 + 16)  # int key (8B) + complex128 value (16B)
        return StateMetadata(
            n_qubits=self._n,
            representation_name="sparse",
            memory_bytes=mem,
            sparsity=sparsity,
            entropy_estimate=entropy,
            approximation_error=0.0,
            bond_dimension=0,
        )

    def to_dense(self):
        from amqt.representations.dense.dense import DenseRepresentation
        return DenseRepresentation(self._n, self.get_statevector())


# ---------------------------------------------------------------------------
# Bit manipulation helpers
# ---------------------------------------------------------------------------

def _extract_bits(basis_idx: int, target_qubits: list[int], n: int) -> int:
    """Extract bits at *target_qubits* positions from *basis_idx*.

    Qubit 0 = MSB (bit n-1 of the integer).
    Returns a sub-index with qubit target_qubits[0] as the MSB of the result.
    """
    result = 0
    for i, q in enumerate(target_qubits):
        bit_pos = n - 1 - q          # bit position in the integer (0=LSB)
        bit = (basis_idx >> bit_pos) & 1
        result = (result << 1) | bit
    return result


def _set_bits(basis_idx: int, target_qubits: list[int], value: int, n: int) -> int:
    """Return a new basis index with bits at *target_qubits* replaced by *value*."""
    result = basis_idx
    k = len(target_qubits)
    for i, q in enumerate(target_qubits):
        bit_pos = n - 1 - q
        # Extract bit i of value (MSB first)
        v_bit = (value >> (k - 1 - i)) & 1
        # Clear the bit then set
        result = (result & ~(1 << bit_pos)) | (v_bit << bit_pos)
    return result


def _entropy_estimate(sv: np.ndarray, n: int) -> float:
    if n < 2:
        return 0.0
    n_a = n // 2
    n_b = n - n_a
    rho = sv.reshape(1 << n_a, 1 << n_b)
    s = np.linalg.svd(rho, compute_uv=False)
    lam = s ** 2
    lam = lam[lam > 1e-15]
    entropy = float(-np.sum(lam * np.log2(lam)))
    max_entropy = float(min(n_a, n_b))
    return entropy / max_entropy if max_entropy > 0 else 0.0
