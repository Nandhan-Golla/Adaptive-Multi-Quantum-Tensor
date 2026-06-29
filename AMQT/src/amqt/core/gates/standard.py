"""
Standard quantum gates as lightweight objects that carry their unitary matrix.

Each gate exposes:
  .matrix  — numpy array of shape (2**n_qubits, 2**n_qubits), complex128
  .n_qubits — number of qubits the gate acts on
  .name    — short human-readable label
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Gate:
    """Abstract base for all gate objects."""

    name: str = "Gate"
    n_qubits: int = 1
    is_clifford: bool = False  # True for gates in the Clifford group

    @property
    def matrix(self) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}(n_qubits={self.n_qubits})"


# ---------------------------------------------------------------------------
# Single-qubit gates
# ---------------------------------------------------------------------------

class _ConstantGate(Gate):
    """A gate whose matrix is a fixed class-level constant."""

    _matrix: np.ndarray  # override in subclass
    n_qubits: int = 1

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()


class HGate(_ConstantGate):
    name = "H"
    is_clifford = True
    _matrix = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


class XGate(_ConstantGate):
    name = "X"
    is_clifford = True
    _matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)


class YGate(_ConstantGate):
    name = "Y"
    is_clifford = True
    _matrix = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


class ZGate(_ConstantGate):
    name = "Z"
    is_clifford = True
    _matrix = np.array([[1, 0], [0, -1]], dtype=np.complex128)


class SGate(_ConstantGate):
    name = "S"
    is_clifford = True
    _matrix = np.array([[1, 0], [0, 1j]], dtype=np.complex128)


class TGate(_ConstantGate):
    name = "T"
    is_clifford = False  # T is non-Clifford; it breaks stabilizer simulation
    _matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)


class IGate(_ConstantGate):
    name = "I"
    is_clifford = True
    _matrix = np.eye(2, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Parameterised single-qubit gates
# ---------------------------------------------------------------------------

class RZGate(Gate):
    """Rotation about Z-axis by *theta* radians: diag(e^{-i theta/2}, e^{i theta/2})."""

    name = "RZ"
    n_qubits = 1

    def __init__(self, theta: float) -> None:
        self.theta = theta

    @property
    def matrix(self) -> np.ndarray:
        half = self.theta / 2
        return np.array(
            [[np.exp(-1j * half), 0], [0, np.exp(1j * half)]],
            dtype=np.complex128,
        )

    def __repr__(self) -> str:
        return f"RZ(theta={self.theta:.4f})"


class RXGate(Gate):
    """Rotation about X-axis by *theta* radians."""

    name = "RX"
    n_qubits = 1

    def __init__(self, theta: float) -> None:
        self.theta = theta

    @property
    def matrix(self) -> np.ndarray:
        half = self.theta / 2
        c, s = np.cos(half), np.sin(half)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    def __repr__(self) -> str:
        return f"RX(theta={self.theta:.4f})"


class RYGate(Gate):
    """Rotation about Y-axis by *theta* radians."""

    name = "RY"
    n_qubits = 1

    def __init__(self, theta: float) -> None:
        self.theta = theta

    @property
    def matrix(self) -> np.ndarray:
        half = self.theta / 2
        c, s = np.cos(half), np.sin(half)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    def __repr__(self) -> str:
        return f"RY(theta={self.theta:.4f})"


class PhaseGate(Gate):
    """Single-qubit phase gate: diag(1, e^{i phi})."""

    name = "Phase"
    n_qubits = 1

    def __init__(self, phi: float) -> None:
        self.phi = phi

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * self.phi)]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Two-qubit gates
# ---------------------------------------------------------------------------

class CNOTGate(_ConstantGate):
    name = "CNOT"
    n_qubits = 2
    is_clifford = True
    _matrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=np.complex128,
    )


class CZGate(_ConstantGate):
    name = "CZ"
    n_qubits = 2
    is_clifford = True
    _matrix = np.diag([1, 1, 1, -1]).astype(np.complex128)


class SWAPGate(_ConstantGate):
    name = "SWAP"
    n_qubits = 2
    is_clifford = True
    _matrix = np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
        dtype=np.complex128,
    )


class ISWAPGate(_ConstantGate):
    name = "ISWAP"
    n_qubits = 2
    _matrix = np.array(
        [[1, 0,  0,  0],
         [0, 0,  1j, 0],
         [0, 1j, 0,  0],
         [0, 0,  0,  1]],
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
# Three-qubit gates
# ---------------------------------------------------------------------------

class ToffoliGate(_ConstantGate):
    """Toffoli (CCX) gate."""
    name = "Toffoli"
    n_qubits = 3
    _matrix = np.eye(8, dtype=np.complex128)
    _matrix[6, 6] = 0
    _matrix[7, 7] = 0
    _matrix[6, 7] = 1
    _matrix[7, 6] = 1


# ---------------------------------------------------------------------------
# Pre-built singleton instances (convenient aliases)
# ---------------------------------------------------------------------------

H = HGate()
X = XGate()
Y = YGate()
Z = ZGate()
S = SGate()
T = TGate()
I = IGate()
CNOT = CNOTGate()
CX = CNOT  # alias
CZ = CZGate()
SWAP = SWAPGate()
ISWAP = ISWAPGate()
Toffoli = ToffoliGate()
CCX = Toffoli


def RZ(theta: float) -> RZGate:
    return RZGate(theta)


def RX(theta: float) -> RXGate:
    return RXGate(theta)


def RY(theta: float) -> RYGate:
    return RYGate(theta)


def Phase(phi: float) -> PhaseGate:
    return PhaseGate(phi)
