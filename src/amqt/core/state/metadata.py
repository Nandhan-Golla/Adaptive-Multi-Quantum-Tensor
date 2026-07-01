"""
StateMetadata — per-state diagnostics used by the runtime dispatcher
to make representation-switching decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StateMetadata:
   

    n_qubits: int
    representation_name: str = "dense"
    memory_bytes: int = 0
    sparsity: float = 0.0
    entropy_estimate: float = 0.0
    approximation_error: float = 0.0
    bond_dimension: int = 0

    # --- convenience helpers ------------------------------------------------

    def is_sparse(self, threshold: float = 0.90) -> bool:

        return self.sparsity >= threshold

    def is_entangled(self, threshold: float = 0.3) -> bool:

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
