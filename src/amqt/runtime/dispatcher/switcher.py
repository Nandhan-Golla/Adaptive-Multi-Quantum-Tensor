"""
RepresentationSwitcher — decides whether and how to switch representations.

The switching policy is purely heuristic and is deliberately separated
from the dispatcher so it can be swapped out independently (e.g. replaced
by a learned policy later).

Current heuristics (all thresholds are tunable class attributes):

+-------------------+----------------------------+---------------------------+
| Condition         | Current rep                | Action                    |
+-------------------+----------------------------+---------------------------+
| sparsity > 0.95   | dense                      | -> sparse                 |
| sparsity < 0.70   | sparse                     | -> dense                  |
| n >= 12 and       | dense or sparse            | -> mps                    |
|   entropy > 0.50  |                            |                           |
| entropy < 0.10    | mps                        | -> sparse (or dense)      |
| bond_dim reaches  | mps                        | stay mps (no better alt.) |
|   max_bond        |                            |                           |
+-------------------+----------------------------+---------------------------+

Stabilizer switching is handled separately: QuantumTensor passes
``initial_representation="stabilizer"`` explicitly.  The switcher does NOT
auto-promote to stabilizer because it cannot know ahead of time whether
all future gates will be Clifford.

All conversions go through ``to_dense()`` as a safe intermediate step.
"""
from __future__ import annotations

from amqt.representations.base import QuantumRepresentation
from amqt.core.state.metadata import StateMetadata


class RepresentationSwitcher:
    """Encapsulates representation-switching heuristics."""

    # --- Policy knobs -------------------------------------------------------
    SPARSE_THRESHOLD: float = 0.95    # sparsity above which Dense -> Sparse
    DENSE_THRESHOLD: float = 0.70     # sparsity below which Sparse -> Dense
    MPS_QUBIT_MIN: int = 12           # minimum n_qubits for MPS
    MPS_ENTROPY_MIN: float = 0.50     # minimum entropy for MPS
    MPS_EXIT_ENTROPY: float = 0.10    # entropy below which MPS -> Sparse/Dense
    # ------------------------------------------------------------------------

    def switch_if_beneficial(
        self,
        rep: QuantumRepresentation,
        metadata: StateMetadata,
    ) -> QuantumRepresentation:
        """Return a new representation if switching is beneficial, else *rep*.

        Parameters
        ----------
        rep:
            Current representation object.
        metadata:
            Freshly computed metadata for *rep*.

        Returns
        -------
        QuantumRepresentation
            Either the same object (if no switch warranted) or a new
            representation containing the same quantum state.
        """
        current = metadata.representation_name
        n = metadata.n_qubits

        # Dense -> Sparse: very high sparsity
        if current == "dense" and metadata.sparsity > self.SPARSE_THRESHOLD:
            return self._to_sparse(rep)

        # Sparse -> Dense: state became dense
        if current == "sparse" and metadata.sparsity < self.DENSE_THRESHOLD:
            return self._to_dense(rep)

        # Dense or Sparse -> MPS: large system with significant entanglement
        if (
            current in ("dense", "sparse")
            and n >= self.MPS_QUBIT_MIN
            and metadata.entropy_estimate > self.MPS_ENTROPY_MIN
        ):
            return self._to_mps(rep, n)

        # MPS -> Sparse/Dense: entanglement dropped (e.g., measurement-like gate)
        if current == "mps" and metadata.entropy_estimate < self.MPS_EXIT_ENTROPY:
            if metadata.sparsity > self.SPARSE_THRESHOLD:
                return self._to_sparse(rep)
            return self._to_dense(rep)

        return rep  # No switch warranted

    # ------------------------------------------------------------------

    @staticmethod
    def _to_dense(rep: QuantumRepresentation):
        return rep.to_dense()

    @staticmethod
    def _to_sparse(rep: QuantumRepresentation):
        from amqt.representations.sparse.sparse import SparseRepresentation
        sv = rep.get_statevector()
        import numpy as np
        amps = {
            i: complex(sv[i])
            for i in range(len(sv))
            if abs(sv[i]) > 1e-10
        }
        return SparseRepresentation(rep.n_qubits, amplitudes=amps)

    @staticmethod
    def _to_mps(rep: QuantumRepresentation, n: int):
        from amqt.representations.mps.mps import MPSRepresentation
        sv = rep.get_statevector()
        return MPSRepresentation(n, statevector=sv)
