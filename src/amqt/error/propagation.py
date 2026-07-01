
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ErrorEvent:
    step: int
    source: str
    error: float
    cumulative: float


class ErrorBudget:

    def __init__(self, tolerance: float = 1e-3) -> None:
        self._tolerance = tolerance
        self._cumulative: float = 0.0
        self._step: int = 0
        self._history: List[ErrorEvent] = []

    def record(self, error: float, source: str = "gate") -> None:
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
        return self._cumulative <= self._tolerance

    @property
    def history(self) -> List[ErrorEvent]:
        return list(self._history)

    @property
    def n_steps(self) -> int:
        return self._step

    def reset(self) -> None:
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
