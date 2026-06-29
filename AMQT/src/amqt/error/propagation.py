"""
Error budget tracking across a quantum circuit.

The ErrorBudget accumulates approximation errors from each gate step and
maintains an upper bound on the total trace distance from the exact state.

Use cases
---------
1. Track MPS truncation error across a long circuit.
2. Combine per-gate approximate errors into a circuit-level fidelity bound.
3. Flag when accumulated error exceeds a user-specified tolerance.

Mathematical guarantee
----------------------
For a sequence of unitary channels U_1, ..., U_d applied to initial state |ψ₀⟩,
where each step introduces error ε_i in trace distance, the total error satisfies:

    T(ρ_exact, ρ_approx) ≤ Σ_i ε_i   (triangle inequality)

This bound is tight in the worst case but often much looser in practice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ErrorEvent:
    """Records a single error contribution."""
    step: int
    source: str
    error: float
    cumulative: float


class ErrorBudget:
    """Accumulates and tracks approximation error across circuit execution.

    Parameters
    ----------
    tolerance:
        Alert threshold.  When cumulative error exceeds this value,
        ``is_within_budget`` returns False.
    """

    def __init__(self, tolerance: float = 1e-3) -> None:
        self._tolerance = tolerance
        self._cumulative: float = 0.0
        self._step: int = 0
        self._history: List[ErrorEvent] = []

    def record(self, error: float, source: str = "gate") -> None:
        """Add *error* to the cumulative error budget.

        Parameters
        ----------
        error:
            Non-negative approximation error from this step (trace distance
            or Frobenius norm of truncated singular values, depending on
            the representation).
        source:
            Human-readable label for where this error came from.
        """
        if error < 0:
            raise ValueError(f"Error must be non-negative, got {error}")
        self._cumulative += error
        event = ErrorEvent(
            step=self._step,
            source=source,
            error=error,
            cumulative=self._cumulative,
        )
        self._history.append(event)
        self._step += 1

    @property
    def cumulative(self) -> float:
        """Total accumulated error (upper bound on trace distance from exact state)."""
        return self._cumulative

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        if value <= 0:
            raise ValueError("tolerance must be positive")
        self._tolerance = value

    @property
    def is_within_budget(self) -> bool:
        """True if accumulated error has not exceeded the tolerance."""
        return self._cumulative <= self._tolerance

    @property
    def history(self) -> List[ErrorEvent]:
        return list(self._history)

    @property
    def n_steps(self) -> int:
        return self._step

    def reset(self) -> None:
        """Reset the budget (e.g., after a representation switch to exact form)."""
        self._cumulative = 0.0
        self._step = 0
        self._history.clear()

    def summary(self) -> str:
        lines = [
            f"ErrorBudget: cumulative={self._cumulative:.3e}, "
            f"tolerance={self._tolerance:.3e}, "
            f"steps={self._step}, "
            f"within_budget={self.is_within_budget}"
        ]
        if self._history:
            top = sorted(self._history, key=lambda e: e.error, reverse=True)[:3]
            lines.append("  Largest contributions:")
            for ev in top:
                lines.append(f"    step={ev.step} ({ev.source}): {ev.error:.3e}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ErrorBudget(cumulative={self._cumulative:.3e}, "
            f"tolerance={self._tolerance:.3e}, "
            f"within_budget={self.is_within_budget})"
        )
