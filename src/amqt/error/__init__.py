"""
AMQT error and fidelity tracking.

Every approximation in AMQT must be measurable.  This module provides:

  fidelity    — state fidelity and trace distance between quantum states
  propagation — error budget accumulation across a circuit
"""
from amqt.error.fidelity import state_fidelity, trace_distance, bures_distance
from amqt.error.propagation import ErrorBudget

__all__ = [
    "state_fidelity",
    "trace_distance",
    "bures_distance",
    "ErrorBudget",
]
