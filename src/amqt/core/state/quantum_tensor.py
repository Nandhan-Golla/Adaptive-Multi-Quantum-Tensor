
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


    def apply(self, gate: "Gate", *qubits: int) -> "QuantumTensor":

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


    def statevector(self) -> np.ndarray:

        return self._dispatcher.representation.get_statevector()

    def probabilities(self) -> np.ndarray:
        sv = self.statevector()
        return (sv.conj() * sv).real

    def norm(self) -> float:
        return float(np.linalg.norm(self.statevector()))

    def expectation(self, observable: np.ndarray) -> complex:
        sv = self.statevector()
        return complex(sv.conj() @ (observable @ sv))

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def metadata(self) -> StateMetadata:
        return self._dispatcher.metadata

    @property
    def representation_name(self) -> str:
        return self._dispatcher.metadata.representation_name

    @property
    def switch_history(self):
        return self._dispatcher.switch_history

    def force_representation(self, name: str) -> "QuantumTensor":
        current_sv = self.statevector()
        rep = _make_representation(self._n, name, self._max_bond, sv=current_sv)
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
