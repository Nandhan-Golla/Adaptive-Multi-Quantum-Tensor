
from __future__ import annotations

from typing import Sequence

import numpy as np

from amqt.representations.base import QuantumRepresentation
from amqt.core.state.metadata import StateMetadata


_ZERO_THRESHOLD = 1e-10

_BACKEND = "numpy"
_rust_core = None
_nb_1q = None
_nb_2q = None

try:
    import amqt_core as _rust_core 
    _BACKEND = "rust"
except ImportError:
    pass

if _BACKEND == "numpy":
    try:
        from amqt.kernels.numba_kernels import (  
            apply_gate_1q as _nb_1q,
            apply_gate_2q as _nb_2q,
            NUMBA_AVAILABLE,
        )
        if NUMBA_AVAILABLE:
            _BACKEND = "numba"
    except (ImportError, Exception):
        pass


class DenseRepresentation(QuantumRepresentation):
    """Full statevector stored as a rank-1 complex128 numpy array."""

    def __init__(self, n_qubits: int, statevector: np.ndarray | None = None) -> None:
        self._n = n_qubits
        dim = 1 << n_qubits
        if statevector is None:
            self._sv = np.zeros(dim, dtype=np.complex128)
            self._sv[0] = 1.0
        else:
            sv = np.asarray(statevector, dtype=np.complex128).ravel()
            if sv.shape[0] != dim:
                raise ValueError(
                    f"statevector length {sv.shape[0]} != 2^{n_qubits} = {dim}"
                )
            self._sv = sv.copy()

    # ------------------------------------------------------------------
    # QuantumRepresentation interface
    # ------------------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        return self._n

    def apply_gate(self, gate_matrix: np.ndarray, target_qubits: Sequence[int]) -> None:
        """Apply *gate_matrix* to *target_qubits* using the fastest available backend."""
        k = len(target_qubits)
        gate = np.asarray(gate_matrix, dtype=np.complex128)

        if _BACKEND == "rust" and k <= 2:
            gate_c = np.ascontiguousarray(gate.reshape(1 << k, 1 << k))
            if k == 1:
                _rust_core.apply_gate_1q(self._sv, gate_c, target_qubits[0], self._n)
            else:
                _rust_core.apply_gate_2q(self._sv, gate_c, target_qubits[0], target_qubits[1], self._n)
        elif _BACKEND == "numba" and k <= 2:
            gate_c = np.ascontiguousarray(gate.reshape(1 << k, 1 << k))
            if k == 1:
                _nb_1q(self._sv, gate_c, target_qubits[0], self._n)
            else:
                _nb_2q(self._sv, gate_c, target_qubits[0], target_qubits[1], self._n)
        else:
            self._apply_gate_numpy(gate, target_qubits, k)

    def _apply_gate_numpy(self, gate: np.ndarray, target_qubits: Sequence[int], k: int) -> None:
        """NumPy fallback — handles arbitrary k-qubit gates via tensor contraction."""
        n = self._n
        tensor = self._sv.reshape([2] * n)
        targets = list(target_qubits)
        other_axes = [i for i in range(n) if i not in targets]
        perm = targets + other_axes
        t = np.transpose(tensor, perm)
        shape_rest = t.shape[k:]
        t = t.reshape(1 << k, -1)
        gate_mat = gate.reshape(1 << k, 1 << k)
        t = gate_mat @ t
        t = t.reshape([2] * k + list(shape_rest))
        self._sv = np.transpose(t, np.argsort(perm)).reshape(-1)

    def get_statevector(self) -> np.ndarray:
        return self._sv.copy()

    def get_metadata(self) -> StateMetadata:
        sv = self._sv
        n_zero = int(np.sum(np.abs(sv) < _ZERO_THRESHOLD))
        dim = len(sv)
        sparsity = n_zero / dim
        entropy = _entropy_estimate(sv, self._n)

        return StateMetadata(
            n_qubits=self._n,
            representation_name="dense",
            memory_bytes=sv.nbytes,
            sparsity=sparsity,
            entropy_estimate=entropy,
            approximation_error=0.0,
            bond_dimension=0,
        )

    def to_dense(self) -> "DenseRepresentation":
        return DenseRepresentation(self._n, self._sv)



def _entropy_estimate(sv: np.ndarray, n: int) -> float:
    """Normalised bipartite entropy across the n//2 | n-n//2 cut."""
    if n < 2:
        return 0.0
    n_a = n // 2
    n_b = n - n_a
    dim_a = 1 << n_a
    dim_b = 1 << n_b
    rho = sv.reshape(dim_a, dim_b)
    s = np.linalg.svd(rho, compute_uv=False)
    lam = s ** 2
    lam = lam[lam > 1e-15]
    entropy = float(-np.sum(lam * np.log2(lam)))
    max_entropy = float(min(n_a, n_b))
    return entropy / max_entropy if max_entropy > 0 else 0.0
