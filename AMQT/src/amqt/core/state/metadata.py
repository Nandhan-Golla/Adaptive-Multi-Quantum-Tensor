"""
StateMetadata — per-state diagnostics used by the runtime dispatcher
to make representation-switching decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StateMetadata:
    """Lightweight snapshot of a quantum state's structural properties.

    All fields are estimates; they need not be exact.  The dispatcher
    uses them as heuristics, not as proofs.

    Attributes
    ----------
    n_qubits:
        Number of qubits in the system.
    representation_name:
        Human-readable name of the currently active representation
        (e.g. ``"dense"``, ``"sparse"``, ``"mps"``).
    memory_bytes:
        Approximate number of bytes currently consumed by the state data.
    sparsity:
        Fraction of basis amplitudes that are effectively zero
        (|amplitude| < threshold).  0.0 = fully dense, 1.0 = all zero.
    entropy_estimate:
        Estimate of the von-Neumann entanglement entropy across the
        first bipartition (qubits 0..n//2-1 | n//2..n-1), normalised
        to [0, 1] by dividing by log(2^(n//2)).
    approximation_error:
        Cumulative truncation / approximation error since the last
        exact initialisation.  For dense and sparse representations
        this is 0.0.  For MPS it accumulates SVD truncation norms.
    bond_dimension:
        Maximum bond dimension currently used (MPS only; 0 otherwise).
    """

    n_qubits: int
    representation_name: str = "dense"
    memory_bytes: int = 0
    sparsity: float = 0.0
    entropy_estimate: float = 0.0
    approximation_error: float = 0.0
    bond_dimension: int = 0

    # --- convenience helpers ------------------------------------------------

    def is_sparse(self, threshold: float = 0.90) -> bool:
        """Return True when the state is highly sparse."""
        return self.sparsity >= threshold

    def is_entangled(self, threshold: float = 0.3) -> bool:
        """Return True when the normalised entropy exceeds *threshold*."""
        return self.entropy_estimate >= threshold

    def copy(self) -> "StateMetadata":
        import copy
        return copy.copy(self)

    def __repr__(self) -> str:
        return (
            f"StateMetadata(n_qubits={self.n_qubits}, "
            f"rep={self.representation_name!r}, "
            f"sparsity={self.sparsity:.3f}, "
            f"entropy={self.entropy_estimate:.3f}, "
            f"error={self.approximation_error:.2e}, "
            f"mem={self.memory_bytes}B)"
        )
