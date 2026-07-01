"""
AMQT — Adaptive Multi-Representation Quantum Tensor
====================================================

Top-level package exposing the primary user API.

Quick start
-----------
>>> from amqt import QuantumTensor, H, CNOT
>>> state = QuantumTensor(2)
>>> state.apply(H, 0).apply(CNOT, 0, 1)
>>> print(state.probabilities())  # [0.5, 0., 0., 0.5]
"""

from amqt.core.state.quantum_tensor import QuantumTensor
from amqt.core.state.metadata import StateMetadata
from amqt.core.gates.standard import (
    # Singletons
    H, X, Y, Z, S, T, I,
    CNOT, CX, CZ, SWAP, ISWAP, Toffoli, CCX,
    # Gate classes
    HGate, XGate, YGate, ZGate, SGate, TGate, IGate,
    CNOTGate, CZGate, SWAPGate,
    # Parameterised constructors
    RZ, RX, RY, Phase,
    RZGate, RXGate, RYGate, PhaseGate,
)
from amqt.representations.dense.dense import DenseRepresentation
from amqt.representations.sparse.sparse import SparseRepresentation
from amqt.representations.mps.mps import MPSRepresentation
from amqt.representations.stabilizer.stabilizer import StabilizerRepresentation
from amqt.error.fidelity import state_fidelity, trace_distance, bures_distance
from amqt.error.propagation import ErrorBudget

__version__ = "0.1.0"

__all__ = [
    # Main class
    "QuantumTensor",
    "StateMetadata",
    # Gates — singletons
    "H", "X", "Y", "Z", "S", "T", "I",
    "CNOT", "CX", "CZ", "SWAP", "ISWAP", "Toffoli", "CCX",
    # Gates — parameterised
    "RZ", "RX", "RY", "Phase",
    # Gate classes
    "HGate", "XGate", "YGate", "ZGate", "SGate", "TGate", "IGate",
    "CNOTGate", "CZGate", "SWAPGate",
    "RZGate", "RXGate", "RYGate", "PhaseGate",
    # Representations
    "DenseRepresentation",
    "SparseRepresentation",
    "MPSRepresentation",
    "StabilizerRepresentation",
    # Error / fidelity
    "state_fidelity",
    "trace_distance",
    "bures_distance",
    "ErrorBudget",
]
