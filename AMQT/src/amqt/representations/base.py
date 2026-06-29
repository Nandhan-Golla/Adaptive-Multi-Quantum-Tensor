"""
Abstract base class for all quantum state representations.

Every concrete representation must implement this interface so the
dispatcher can treat them uniformly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from amqt.core.state.metadata import StateMetadata
    from amqt.representations.dense.dense import DenseRepresentation


class QuantumRepresentation(ABC):
    """Interface shared by all internal quantum state representations."""

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def apply_gate(self, gate_matrix: np.ndarray, target_qubits: Sequence[int]) -> None:
        """Apply *gate_matrix* (shape 2^k × 2^k) to *target_qubits*.

        Parameters
        ----------
        gate_matrix:
            Unitary matrix in the computational basis, complex128,
            shape ``(2**k, 2**k)`` where ``k = len(target_qubits)``.
        target_qubits:
            Ordered sequence of qubit indices the gate acts on.
            Qubit 0 is the most-significant (leftmost) index of the
            statevector tensor.
        """

    @abstractmethod
    def get_statevector(self) -> np.ndarray:
        """Return a dense statevector of shape ``(2**n_qubits,)``, complex128.

        The result must be normalised (||v||_2 == 1) up to floating-point
        precision.  For approximate representations (e.g. MPS with
        truncation) the returned vector is the best available approximation.
        """

    @abstractmethod
    def get_metadata(self) -> "StateMetadata":
        """Compute and return a fresh :class:`~amqt.core.state.metadata.StateMetadata`."""

    @abstractmethod
    def to_dense(self) -> "DenseRepresentation":
        """Convert *self* to a :class:`~amqt.representations.dense.dense.DenseRepresentation`.

        Used as an intermediate step during representation switching.
        """

    # ------------------------------------------------------------------
    # Optional helpers with default implementations
    # ------------------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        raise NotImplementedError

    def norm(self) -> float:
        """Return ||state||_2.  Subclasses may override for efficiency."""
        sv = self.get_statevector()
        return float(np.linalg.norm(sv))

    def probabilities(self) -> np.ndarray:
        """Return measurement probability for each basis state."""
        sv = self.get_statevector()
        return (sv.conj() * sv).real

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_qubits={self.n_qubits})"
