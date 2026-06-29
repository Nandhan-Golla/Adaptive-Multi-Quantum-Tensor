"""
Dispatcher — orchestrates gate application and representation switching.

The Dispatcher sits between the public QuantumTensor API and the
concrete representation objects.  Its responsibilities are:

1. Delegate each gate application to the active representation.
2. Lazily recompute StateMetadata — only when accessed or when the
   switcher needs it (every ``metadata_interval`` gates).
3. Consult RepresentationSwitcher to decide if a switch is warranted.
4. If so, perform the switch transparently.
5. Maintain a history of switch events for introspection / research.

Performance note
----------------
The metadata computation (especially the SVD-based entropy estimate in
DenseRepresentation) is expensive.  Previous versions called
``get_metadata()`` after *every* gate, which made it the dominant
cost (≈ 46 % of total runtime for n=10 GHZ).  The current version
uses a lazy / interval-based policy:

* ``auto_switch=False``: metadata is never computed during ``apply_gate``;
  it is computed on-demand when the ``.metadata`` property is accessed.
* ``auto_switch=True``: metadata is recomputed once every
  ``metadata_interval`` gates (default 8) so the switcher has fresh
  information without paying the SVD cost on every single gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from amqt.representations.base import QuantumRepresentation
from amqt.core.state.metadata import StateMetadata
from amqt.runtime.dispatcher.switcher import RepresentationSwitcher


@dataclass
class SwitchEvent:
    """Records a single representation transition."""
    gate_index: int
    from_rep: str
    to_rep: str
    reason: str = ""


class Dispatcher:
    """Manages gate dispatch, metadata tracking, and representation switching.

    Parameters
    ----------
    representation:
        Initial representation to use.
    switcher:
        Policy object that decides when to switch.  If ``None``, a default
        :class:`RepresentationSwitcher` is used.
    auto_switch:
        Set to ``False`` to disable automatic representation switching
        (useful for benchmarking a fixed representation).
    metadata_interval:
        When ``auto_switch=True``, metadata is recomputed every this many
        gates.  Higher values reduce SVD overhead at the cost of less
        frequent switching decisions.  Default: 8.
    """

    def __init__(
        self,
        representation: QuantumRepresentation,
        switcher: RepresentationSwitcher | None = None,
        auto_switch: bool = True,
        metadata_interval: int = 8,
    ) -> None:
        self._rep = representation
        self._switcher = switcher or RepresentationSwitcher()
        self._auto_switch = auto_switch
        self._metadata_interval = metadata_interval
        self._gate_count: int = 0
        self._switch_history: List[SwitchEvent] = []
        # Metadata is computed lazily — only on first access or at switch-check intervals.
        self._metadata: Optional[StateMetadata] = None
        self._metadata_stale: bool = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply_gate(self, gate_matrix: np.ndarray, target_qubits: Sequence[int]) -> None:
        """Apply *gate_matrix* to *target_qubits*, then consider switching."""
        self._rep.apply_gate(gate_matrix, target_qubits)
        self._gate_count += 1
        self._metadata_stale = True

        # Only recompute metadata (expensive SVD) at switching checkpoints.
        if self._auto_switch and (self._gate_count % self._metadata_interval == 0):
            self._metadata = self._rep.get_metadata()
            self._metadata_stale = False
            self._maybe_switch()

    @property
    def representation(self) -> QuantumRepresentation:
        return self._rep

    @property
    def metadata(self) -> StateMetadata:
        """Return current metadata, computing it lazily if stale."""
        if self._metadata is None or self._metadata_stale:
            self._metadata = self._rep.get_metadata()
            self._metadata_stale = False
        return self._metadata

    @property
    def switch_history(self) -> List[SwitchEvent]:
        return list(self._switch_history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_switch(self) -> None:
        old_name = self._metadata.representation_name
        new_rep = self._switcher.switch_if_beneficial(self._rep, self._metadata)
        if new_rep is not self._rep:
            new_meta = new_rep.get_metadata()
            self._switch_history.append(
                SwitchEvent(
                    gate_index=self._gate_count,
                    from_rep=old_name,
                    to_rep=new_meta.representation_name,
                )
            )
            self._rep = new_rep
            self._metadata = new_meta
            self._metadata_stale = False
