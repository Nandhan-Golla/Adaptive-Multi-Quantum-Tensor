"""
QuantumTensor — the primary user-facing API for AMQT.

Users interact only with this class.  Internal representation details
(dense statevector, sparse dict, MPS tensors) are completely hidden.

Example
-------
>>> from amqt import QuantumTensor, H, CNOT
>>> state = QuantumTensor(2)          # |00>
>>> state.apply(H, 0)                 # H on qubit 0  -> (|00> + |10>) / sqrt(2)
>>> state.apply(CNOT, 0, 1)           # CNOT           -> Bell state
>>> probs = state.probabilities()
>>> # probs[0] ≈ 0.5, probs[3] ≈ 0.5
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import numpy as np

from amqt.core.state.metadata import StateMetadata
from amqt.representations.dense.dense import DenseRepresentation
from amqt.runtime.dispatcher.dispatcher import Dispatcher

if TYPE_CHECKING:
    from amqt.core.gates.standard import Gate
    from amqt.representations.base import QuantumRepresentation


class QuantumTensor:
    """Adaptive quantum state container for *n_qubits* qubits.

    Parameters
    ----------
    n_qubits:
        Number of qubits.  Initial state is |0...0>.
    initial_representation:
        Name of the starting representation:  ``"dense"`` (default),
        ``"sparse"``, or ``"mps"``.
    max_bond:
        Maximum MPS bond dimension (only used when representation is or
        becomes ``"mps"``).
    auto_switch:
        If ``True`` (default) the runtime may transparently switch
        internal representations to optimise memory or speed.
    """

    def __init__(
        self,
        n_qubits: int,
        initial_representation: str = "dense",
        max_bond: int = 64,
        auto_switch: bool = True,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self._n = n_qubits
        self._max_bond = max_bond

        rep = _make_representation(n_qubits, initial_representation, max_bond)
        self._dispatcher = Dispatcher(rep, auto_switch=auto_switch)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def apply(self, gate: "Gate", *qubits: int) -> "QuantumTensor":
        """Apply *gate* to *qubits* and return *self* for chaining.

        Parameters
        ----------
        gate:
            A :class:`~amqt.core.gates.standard.Gate` instance (e.g. ``H``,
            ``CNOT``, ``RZ(theta)``).
        *qubits:
            Zero-indexed qubit indices the gate acts on.  Must match
            ``gate.n_qubits``.

        Returns
        -------
        QuantumTensor
            *self*, to allow chaining: ``state.apply(H, 0).apply(CNOT, 0, 1)``.
        """
        if len(qubits) != gate.n_qubits:
            raise ValueError(
                f"Gate {gate.name} expects {gate.n_qubits} qubit(s), "
                f"got {len(qubits)}: {qubits}"
            )
        for q in qubits:
            if not (0 <= q < self._n):
                raise ValueError(f"Qubit index {q} out of range [0, {self._n})")

        self._dispatcher.apply_gate(gate.matrix, qubits)
        return self

    # ------------------------------------------------------------------
    # Observables / inspection
    # ------------------------------------------------------------------

    def statevector(self) -> np.ndarray:
        """Return the full statevector as a complex128 array of length 2^n."""
        return self._dispatcher.representation.get_statevector()

    def probabilities(self) -> np.ndarray:
        """Return the measurement probability for each computational basis state."""
        sv = self.statevector()
        return (sv.conj() * sv).real

    def norm(self) -> float:
        """Return ||state||_2 (should be 1.0 up to floating-point error)."""
        return float(np.linalg.norm(self.statevector()))

    def expectation(self, observable: np.ndarray) -> complex:
        """Compute <psi|observable|psi>.

        Parameters
        ----------
        observable:
            Hermitian matrix of shape ``(2**n, 2**n)``, complex128.
        """
        sv = self.statevector()
        return complex(sv.conj() @ (observable @ sv))

    # ------------------------------------------------------------------
    # Metadata / introspection
    # ------------------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def metadata(self) -> StateMetadata:
        """Current :class:`~amqt.core.state.metadata.StateMetadata` snapshot."""
        return self._dispatcher.metadata

    @property
    def representation_name(self) -> str:
        """Human-readable name of the active internal representation."""
        return self._dispatcher.metadata.representation_name

    @property
    def switch_history(self):
        """List of :class:`~amqt.runtime.dispatcher.dispatcher.SwitchEvent` objects."""
        return self._dispatcher.switch_history

    # ------------------------------------------------------------------
    # Advanced / escape hatches
    # ------------------------------------------------------------------

    def force_representation(self, name: str) -> "QuantumTensor":
        """Force a specific internal representation (for benchmarking/research)."""
        current_sv = self.statevector()
        rep = _make_representation(self._n, name, self._max_bond, sv=current_sv)
        # Preserve auto_switch setting
        auto = self._dispatcher._auto_switch
        self._dispatcher = Dispatcher(rep, auto_switch=auto)
        return self

    def __repr__(self) -> str:
        meta = self.metadata
        return (
            f"QuantumTensor(n_qubits={self._n}, "
            f"rep={meta.representation_name!r}, "
            f"sparsity={meta.sparsity:.2f}, "
            f"entropy={meta.entropy_estimate:.2f})"
        )


# ---------------------------------------------------------------------------
# Internal factory
# ---------------------------------------------------------------------------

def _make_representation(
    n: int,
    name: str,
    max_bond: int,
    sv: np.ndarray | None = None,
) -> "QuantumRepresentation":
    if name == "dense":
        return DenseRepresentation(n, statevector=sv)
    if name == "sparse":
        from amqt.representations.sparse.sparse import SparseRepresentation
        if sv is not None:
            amps = {i: complex(sv[i]) for i in range(len(sv)) if abs(sv[i]) > 1e-10}
            return SparseRepresentation(n, amplitudes=amps)
        return SparseRepresentation(n)
    if name == "mps":
        from amqt.representations.mps.mps import MPSRepresentation
        return MPSRepresentation(n, max_bond=max_bond, statevector=sv)
    if name == "stabilizer":
        from amqt.representations.stabilizer.stabilizer import StabilizerRepresentation
        if sv is not None:
            raise ValueError(
                "StabilizerRepresentation cannot be initialised from an arbitrary statevector. "
                "Use 'dense', 'sparse', or 'mps' and call force_representation('stabilizer') "
                "only from the |0...0> state."
            )
        return StabilizerRepresentation(n)
    raise ValueError(
        f"Unknown representation {name!r}. "
        "Choose 'dense', 'sparse', 'mps', or 'stabilizer'."
    )
