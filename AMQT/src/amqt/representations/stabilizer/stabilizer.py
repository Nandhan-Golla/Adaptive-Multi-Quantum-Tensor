"""
Stabilizer representation using the Aaronson-Gottesman CHP tableau.

An n-qubit Clifford state is represented by a (2n × 2n+1) binary matrix:
  Rows 0..n-1   : destabilizer generators
  Rows n..2n-1  : stabilizer generators
  Cols 0..n-1   : X component for each qubit
  Cols n..2n-1  : Z component for each qubit
  Col  2n       : phase r ∈ {0,1} where 0→+1, 1→−1

The full Pauli operator for row h is:
  (-1)^{r_h} · ∏_j  i^{x_{h,j}·z_{h,j}} · X_j^{x_{h,j}} · Z_j^{z_{h,j}}

Gate update rules (Aaronson-Gottesman 2004, Table I):

  H on qubit a:
    r_h ^= x_{h,a} & z_{h,a}
    swap(x_{h,a}, z_{h,a})

  S on qubit a:
    r_h ^= x_{h,a} & z_{h,a}
    z_{h,a} ^= x_{h,a}

  CNOT(ctrl=a, tgt=b):
    r_h ^= x_{h,a} & z_{h,b} & (x_{h,b} ^ z_{h,a} ^ 1)
    x_{h,b} ^= x_{h,a}
    z_{h,a} ^= z_{h,b}

All other Clifford gates are derived from these three primitives.

Non-Clifford gates (T, RZ, etc.) trigger a transparent fallback to
DenseRepresentation; the stabilizer tableau is abandoned from that point.

Memory: O(n²) vs O(2^n) for Dense — exponential advantage for large n.
Gate cost: O(n) per Clifford gate (vs O(2^n) for Dense).
Statevector: O(depth · 2^n) via replay on Dense (computed on demand).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from amqt.representations.base import QuantumRepresentation
from amqt.core.state.metadata import StateMetadata

# ---------------------------------------------------------------------------
# Known Clifford gate matrices (for identification)
# ---------------------------------------------------------------------------

_I1 = np.eye(2, dtype=np.complex128)
_H_MAT = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_X_MAT = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y_MAT = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z_MAT = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_S_MAT = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_SDG_MAT = np.array([[1, 0], [0, -1j]], dtype=np.complex128)

_CNOT_MAT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
)
_CZ_MAT = np.diag([1, 1, 1, -1]).astype(np.complex128)
_SWAP_MAT = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
)

_SINGLE_CLIFFORD = [
    ("H",   _H_MAT),
    ("X",   _X_MAT),
    ("Y",   _Y_MAT),
    ("Z",   _Z_MAT),
    ("S",   _S_MAT),
    ("Sdg", _SDG_MAT),
    ("I",   _I1),
]
_TWO_CLIFFORD = [
    ("CNOT", _CNOT_MAT),
    ("CZ",   _CZ_MAT),
    ("SWAP", _SWAP_MAT),
]


def _identify_clifford(gate_matrix: np.ndarray, k: int) -> Optional[str]:
    """Return the gate name if *gate_matrix* is a known Clifford, else None."""
    G = np.asarray(gate_matrix, dtype=np.complex128)
    if k == 1:
        for name, ref in _SINGLE_CLIFFORD:
            if np.allclose(G, ref, atol=1e-10):
                return name
    elif k == 2:
        for name, ref in _TWO_CLIFFORD:
            if np.allclose(G, ref, atol=1e-10):
                return name
    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StabilizerRepresentation(QuantumRepresentation):
    """CHP tableau representation for Clifford circuits.

    For purely Clifford circuits this uses O(n²) memory instead of O(2^n).
    Non-Clifford gates trigger a transparent fallback to DenseRepresentation.

    Parameters
    ----------
    n_qubits:
        Number of qubits.  Initial state is |0...0>.
    """

    def __init__(self, n_qubits: int) -> None:
        self._n = n_qubits
        n = n_qubits
        # Tableau shape: (2n, 2n+1), uint8
        # Initial state |0...0>:
        #   Destabilizer i: X_i  → row i,   x[i,i]=1
        #   Stabilizer   i: Z_i  → row n+i, z[n+i, n+i]=1
        self._tab = np.zeros((2 * n, 2 * n + 1), dtype=np.uint8)
        for i in range(n):
            self._tab[i, i] = 1          # destabilizer: X_i
            self._tab[n + i, n + i] = 1  # stabilizer: Z_i

        # Gate history for statevector reconstruction (replay on Dense)
        self._history: List[Tuple[np.ndarray, List[int]]] = []
        # Fallback: set when a non-Clifford gate is encountered
        self._fallback: Optional[QuantumRepresentation] = None
        self._clifford_only: bool = True

    # ------------------------------------------------------------------
    # QuantumRepresentation interface
    # ------------------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def is_clifford_only(self) -> bool:
        """True if no non-Clifford gate has been applied yet."""
        return self._clifford_only

    def apply_gate(self, gate_matrix: np.ndarray, target_qubits: Sequence[int]) -> None:
        targets = list(target_qubits)
        k = len(targets)

        if self._fallback is not None:
            self._fallback.apply_gate(gate_matrix, target_qubits)
            self._history.append((np.asarray(gate_matrix, dtype=np.complex128), targets))
            return

        gate_id = _identify_clifford(gate_matrix, k)
        if gate_id is None:
            # Non-Clifford: switch to Dense and continue there
            self._clifford_only = False
            self._fallback = self._build_dense_from_history()
            self._fallback.apply_gate(gate_matrix, target_qubits)
            self._history.append((np.asarray(gate_matrix, dtype=np.complex128), targets))
            return

        self._apply_clifford(gate_id, targets)
        self._history.append((np.asarray(gate_matrix, dtype=np.complex128), targets))

    def get_statevector(self) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.get_statevector()
        return self._build_dense_from_history().get_statevector()

    def get_metadata(self) -> StateMetadata:
        n = self._n
        if self._fallback is not None:
            meta = self._fallback.get_metadata()
            meta.representation_name = "stabilizer(dense)"
            return meta

        # Memory is O(n²) for the tableau
        mem = self._tab.nbytes
        # Entropy estimate from tableau cross-terms
        entropy = self._entropy_estimate()
        # Sparsity: stabilizer states can be highly sparse; report 0 since
        # we don't reconstruct the statevector for this estimate
        return StateMetadata(
            n_qubits=n,
            representation_name="stabilizer",
            memory_bytes=mem,
            sparsity=0.0,          # updated lazily if needed
            entropy_estimate=entropy,
            approximation_error=0.0,
            bond_dimension=0,
        )

    def to_dense(self):
        from amqt.representations.dense.dense import DenseRepresentation
        return DenseRepresentation(self._n, self.get_statevector())

    # ------------------------------------------------------------------
    # CHP tableau gate updates
    # ------------------------------------------------------------------

    def _apply_h(self, a: int) -> None:
        n = self._n
        tab = self._tab
        xa = tab[:, a].copy()
        za = tab[:, n + a].copy()
        tab[:, 2 * n] ^= xa & za
        tab[:, a] = za
        tab[:, n + a] = xa

    def _apply_s(self, a: int) -> None:
        n = self._n
        tab = self._tab
        xa = tab[:, a]
        za = tab[:, n + a]
        tab[:, 2 * n] ^= xa & za
        tab[:, n + a] ^= xa

    def _apply_cnot(self, ctrl: int, tgt: int) -> None:
        n = self._n
        tab = self._tab
        xc = tab[:, ctrl]
        zt = tab[:, n + tgt]
        xt = tab[:, tgt]
        zc = tab[:, n + ctrl]
        tab[:, 2 * n] ^= xc & zt & ((xt ^ zc ^ 1) & 0x1)
        tab[:, tgt] ^= xc
        tab[:, n + ctrl] ^= zt

    def _apply_x(self, a: int) -> None:
        # X conjugation: X→X, Z→−Z → phase ^= z_{h,a}
        self._tab[:, 2 * self._n] ^= self._tab[:, self._n + a]

    def _apply_z(self, a: int) -> None:
        # Z conjugation: X→−X, Z→Z → phase ^= x_{h,a}
        self._tab[:, 2 * self._n] ^= self._tab[:, a]

    def _apply_y(self, a: int) -> None:
        # Y conjugation: X→−X, Z→−Z → phase ^= x ^ z
        n = self._n
        self._tab[:, 2 * n] ^= self._tab[:, a] ^ self._tab[:, n + a]

    def _apply_clifford(self, gate_id: str, targets: List[int]) -> None:
        if gate_id == "H":
            self._apply_h(targets[0])
        elif gate_id == "S":
            self._apply_s(targets[0])
        elif gate_id == "Sdg":
            # S† = S³
            a = targets[0]
            self._apply_s(a)
            self._apply_s(a)
            self._apply_s(a)
        elif gate_id == "X":
            self._apply_x(targets[0])
        elif gate_id == "Z":
            self._apply_z(targets[0])
        elif gate_id == "Y":
            self._apply_y(targets[0])
        elif gate_id == "I":
            pass
        elif gate_id == "CNOT":
            self._apply_cnot(targets[0], targets[1])
        elif gate_id == "CZ":
            # CZ = (I⊗H) · CNOT(a,b) · (I⊗H)
            a, b = targets[0], targets[1]
            self._apply_h(b)
            self._apply_cnot(a, b)
            self._apply_h(b)
        elif gate_id == "SWAP":
            # SWAP = CNOT(a,b) · CNOT(b,a) · CNOT(a,b)
            a, b = targets[0], targets[1]
            self._apply_cnot(a, b)
            self._apply_cnot(b, a)
            self._apply_cnot(a, b)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_dense_from_history(self):
        """Reconstruct the state by replaying gate history on a fresh Dense."""
        from amqt.representations.dense.dense import DenseRepresentation
        rep = DenseRepresentation(self._n)
        for mat, targets in self._history:
            rep.apply_gate(mat, targets)
        return rep

    def _entropy_estimate(self) -> float:
        """Estimate bipartite entropy from tableau cross-terms.

        Count stabilizer generators (rows n..2n-1) that have support on
        both the left half (qubits 0..n//2-1) and right half (n//2..n-1).
        Normalise by the maximum possible (n//2 generators with cross-support).
        """
        n = self._n
        if n < 2:
            return 0.0
        n_a = n // 2
        stab_rows = self._tab[n:, :]  # (n, 2n+1)
        # X and Z support on left half: cols 0..n_a-1 and n..n+n_a-1
        left_x = stab_rows[:, :n_a]
        left_z = stab_rows[:, n:n + n_a]
        left_support = np.any(left_x | left_z, axis=1)
        # X and Z support on right half
        right_x = stab_rows[:, n_a:n]
        right_z = stab_rows[:, n + n_a:2 * n]
        right_support = np.any(right_x | right_z, axis=1)
        cross = int(np.sum(left_support & right_support))
        return cross / n_a  # normalised to [0, 1]

    def __repr__(self) -> str:
        mode = "clifford" if self._clifford_only else "fallback→dense"
        return f"StabilizerRepresentation(n_qubits={self._n}, mode={mode})"
