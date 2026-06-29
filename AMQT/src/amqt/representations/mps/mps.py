"""
Matrix Product State (MPS) representation.

An n-qubit pure state is written as:

    |psi> = sum_{s_0,...,s_{n-1}} A^{s_0}[0] A^{s_1}[1] ... A^{s_{n-1}}[n-1] |s_0 ... s_{n-1}>

where each A^{s_i}[i] is a matrix of shape (chi_{i-1}, chi_i) and chi_i
is the bond dimension (chi_0 = chi_n = 1).

We use a left-canonical gauge: A^dag[i] A[i] = I for all i.

Gate application:
  Single-qubit gate G on site i:
      A'^{s_i}[i] = sum_{s} G[s_i, s] A^{s}[i]   (local contraction, no SVD needed)

  Two-qubit gate U on adjacent sites (i, i+1):
      1. Contract: Theta^{s_i, s_{i+1}}_{alpha, beta} =
             sum_{gamma} A^{s_i}_{alpha, gamma} A^{s_{i+1}}_{gamma, beta}
         Shape: (chi_left, 2, 2, chi_right)
      2. Apply gate: Theta' = U Theta  (reshape as needed)
      3. SVD: reshape Theta' to (chi_left * 2, 2 * chi_right),
             SVD -> U_svd S V_svd, truncate to max_bond.
      4. New tensors:
             A'[i]     = U_svd[:, :chi_new].reshape(chi_left, 2, chi_new)
             A'[i+1]   = (S_trunc @ V_svd_trunc).reshape(chi_new, 2, chi_right)
      5. Accumulate truncation error: || S_discarded ||_F

Non-adjacent two-qubit gates are handled by SWAP-routing the qubits
to adjacent positions (alternative: full statevector fallback).  Here
we use the SVD-routing approach.

Memory: O(n * chi^2 * 2) complex128 = O(32 n chi^2) bytes.
Gate cost: O(chi^3) per two-qubit gate (dominated by SVD).
"""
from __future__ import annotations

from typing import Sequence, List

import numpy as np

from amqt.representations.base import QuantumRepresentation
from amqt.core.state.metadata import StateMetadata

_DEFAULT_MAX_BOND = 64
_SVD_THRESHOLD = 1e-10


class MPSRepresentation(QuantumRepresentation):
    """Left-canonical MPS for an n-qubit pure state.

    Each site tensor has shape ``(chi_left, 2, chi_right)`` where
    the physical index is in the middle (axis 1).
    """

    def __init__(
        self,
        n_qubits: int,
        max_bond: int = _DEFAULT_MAX_BOND,
        statevector: np.ndarray | None = None,
    ) -> None:
        self._n = n_qubits
        self._max_bond = max_bond
        self._trunc_error: float = 0.0

        if statevector is not None:
            sv = np.asarray(statevector, dtype=np.complex128).ravel()
            if sv.shape[0] != 1 << n_qubits:
                raise ValueError("statevector length mismatch")
            self._tensors = _sv_to_mps(sv, n_qubits, max_bond)
        else:
            # |0...0> — all bond dimensions = 1
            self._tensors: List[np.ndarray] = []
            for _ in range(n_qubits):
                t = np.zeros((1, 2, 1), dtype=np.complex128)
                t[0, 0, 0] = 1.0
                self._tensors.append(t)

    # ------------------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def max_bond(self) -> int:
        return self._max_bond

    def apply_gate(self, gate_matrix: np.ndarray, target_qubits: Sequence[int]) -> None:
        k = len(target_qubits)
        targets = list(target_qubits)

        if k == 1:
            self._apply_single(gate_matrix, targets[0])
        elif k == 2:
            q0, q1 = targets[0], targets[1]
            if abs(q0 - q1) == 1:
                # Adjacent: apply directly
                left, right = (q0, q1) if q0 < q1 else (q1, q0)
                if q0 > q1:
                    # Reorder gate for swapped qubit order
                    gate_matrix = _swap_gate_qubits(gate_matrix)
                self._apply_two_adjacent(gate_matrix, left, right)
            else:
                # Non-adjacent: SWAP-route to adjacent, apply, unswap
                self._apply_two_nonadjacent(gate_matrix, q0, q1)
        else:
            # Fallback: convert to dense, apply, convert back
            from amqt.representations.dense.dense import DenseRepresentation
            dense = self.to_dense()
            dense.apply_gate(gate_matrix, target_qubits)
            new_mps = MPSRepresentation(
                self._n,
                max_bond=self._max_bond,
                statevector=dense.get_statevector(),
            )
            self._tensors = new_mps._tensors
            self._trunc_error += new_mps._trunc_error

    def get_statevector(self) -> np.ndarray:
        """Reconstruct the statevector by contracting all MPS tensors."""
        # Start with the first tensor: shape (1, 2, chi_1)
        result = self._tensors[0]  # (1, 2, chi_1)
        # Reshape to (2, chi_1) by squeezing left boundary
        result = result[0]  # (2, chi_1)

        for i in range(1, self._n):
            t = self._tensors[i]  # (chi_i, 2, chi_{i+1})
            # result: (2^i, chi_i);  t: (chi_i, 2, chi_{i+1})
            # contract over chi_i:  (2^i, chi_i) x (chi_i, 2 * chi_{i+1})
            chi_left, phys, chi_right = t.shape
            t2 = t.reshape(chi_left, phys * chi_right)  # (chi_i, 2*chi_{i+1})
            result = result @ t2                          # (2^i, 2*chi_{i+1})
            result = result.reshape(-1, chi_right)         # (2^(i+1), chi_{i+1})

        # result: (2^n, 1) -> flatten
        return result.reshape(-1)

    def get_metadata(self) -> StateMetadata:
        sv = self.get_statevector()
        n_zero = int(np.sum(np.abs(sv) < _SVD_THRESHOLD))
        dim = len(sv)
        sparsity = n_zero / dim
        entropy = _entropy_estimate(sv, self._n)
        max_bd = max(t.shape[0] for t in self._tensors)
        mem = sum(t.nbytes for t in self._tensors)
        return StateMetadata(
            n_qubits=self._n,
            representation_name="mps",
            memory_bytes=mem,
            sparsity=sparsity,
            entropy_estimate=entropy,
            approximation_error=self._trunc_error,
            bond_dimension=max_bd,
        )

    def to_dense(self):
        from amqt.representations.dense.dense import DenseRepresentation
        return DenseRepresentation(self._n, self.get_statevector())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_single(self, gate: np.ndarray, q: int) -> None:
        """Apply 2×2 gate *gate* to site *q*."""
        G = np.asarray(gate, dtype=np.complex128)
        t = self._tensors[q]          # (chi_l, 2, chi_r)
        # contract: new_t[chi_l, s', chi_r] = sum_s G[s', s] * t[chi_l, s, chi_r]
        # Use einsum: 'ij, ajk -> aik'  where a=chi_l, k=chi_r
        self._tensors[q] = np.einsum("ij,ajk->aik", G, t)

    def _apply_two_adjacent(self, gate: np.ndarray, left: int, right: int) -> None:
        """Apply 4×4 gate *gate* to adjacent sites (*left*, *left+1 = right*)."""
        assert right == left + 1
        A = self._tensors[left]    # (chi_l, 2, chi_m)
        B = self._tensors[right]   # (chi_m, 2, chi_r)

        chi_l = A.shape[0]
        chi_m = A.shape[2]
        chi_r = B.shape[2]

        # Contract A and B into a rank-4 tensor Theta
        # Theta[alpha, s_i, s_{i+1}, beta] = sum_gamma A[alpha, s_i, gamma] * B[gamma, s_{i+1}, beta]
        # Use einsum: 'isj, jsk -> isk' ... let's be explicit:
        Theta = np.einsum("isj,jtk->istk", A, B)  # (chi_l, 2, 2, chi_r)

        # Apply the gate: reshape Theta to (4, chi_l, chi_r) then apply gate (4x4)
        # gate[s'_i s'_{i+1}, s_i s_{i+1}]  (row = output, col = input)
        G = np.asarray(gate, dtype=np.complex128)  # (4, 4)
        # Reshape Theta: (chi_l, 2, 2, chi_r) -> (chi_l * chi_r, 4)... easier differently
        # Let's treat as (4, chi_l, chi_r):
        Theta_r = Theta.transpose(1, 2, 0, 3).reshape(4, chi_l * chi_r)  # (4, chi_l*chi_r)
        Theta_prime = G @ Theta_r                                           # (4, chi_l*chi_r)
        Theta_prime = Theta_prime.reshape(2, 2, chi_l, chi_r).transpose(2, 0, 1, 3)
        # Now Theta_prime: (chi_l, 2, 2, chi_r)

        # SVD decomposition: reshape to (chi_l * 2, 2 * chi_r)
        M = Theta_prime.reshape(chi_l * 2, 2 * chi_r)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        # U: (chi_l*2, r), S: (r,), Vh: (r, 2*chi_r)

        # Truncate to max_bond
        chi_new = min(len(S), self._max_bond)
        # Accumulate truncation error (Frobenius norm of discarded singular values)
        if chi_new < len(S):
            self._trunc_error += float(np.linalg.norm(S[chi_new:]))

        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]

        # New tensors
        new_A = U.reshape(chi_l, 2, chi_new)
        new_B = (np.diag(S) @ Vh).reshape(chi_new, 2, chi_r)

        self._tensors[left] = new_A
        self._tensors[right] = new_B

    def _apply_two_nonadjacent(self, gate: np.ndarray, q0: int, q1: int) -> None:
        """Apply two-qubit gate to non-adjacent qubits via SWAP routing."""
        # Move q0 and q1 adjacent using SWAPs, apply gate, undo SWAPs
        left, right = (q0, q1) if q0 < q1 else (q1, q0)
        swapped = q0 > q1

        SWAP = np.array(
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]],
            dtype=np.complex128,
        )

        # Bubble left qubit right until adjacent to right qubit
        for i in range(left, right - 1):
            self._apply_two_adjacent(SWAP, i, i + 1)

        # Now the original left qubit is at position right-1
        effective_left = right - 1
        effective_right = right

        if swapped:
            gate = _swap_gate_qubits(gate)

        self._apply_two_adjacent(gate, effective_left, effective_right)

        # Bubble back
        for i in range(right - 2, left - 1, -1):
            self._apply_two_adjacent(SWAP, i, i + 1)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sv_to_mps(sv: np.ndarray, n: int, max_bond: int) -> List[np.ndarray]:
    """Convert a statevector to MPS via successive SVDs (left-canonical)."""
    tensors: List[np.ndarray] = []
    chi_left = 1
    remainder = sv.reshape(chi_left, -1)  # (1, 2^n)

    for site in range(n - 1):
        dim_right = remainder.shape[1]
        phys = 2
        # reshape to (chi_left * 2, 2^{n-site-1})
        M = remainder.reshape(chi_left * phys, dim_right // phys)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        chi_new = min(len(S), max_bond)
        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]
        tensors.append(U.reshape(chi_left, phys, chi_new))
        chi_left = chi_new
        remainder = np.diag(S) @ Vh  # (chi_new, remaining_dim)

    # Last site
    tensors.append(remainder.reshape(chi_left, 2, 1))
    return tensors


def _swap_gate_qubits(gate: np.ndarray) -> np.ndarray:
    """Permute a 4x4 two-qubit gate for swapped qubit order."""
    # In our convention |q0 q1>, if we swap q0 and q1 we need to reorder
    # basis states: 00->00, 01->10, 10->01, 11->11
    perm = [0, 2, 1, 3]
    return gate[np.ix_(perm, perm)]


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
