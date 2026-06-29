"""
Fidelity and distance measures between quantum states.

All functions accept statevectors (1-D complex arrays of length 2^n).
Density matrices are supported where noted.

Definitions
-----------
State fidelity (pure states):
    F(ψ, φ) = |⟨ψ|φ⟩|²   ∈ [0, 1]

Trace distance (pure states):
    T(ψ, φ) = (1/2) · ||ρ_ψ − ρ_φ||_1
             = √(1 − F(ψ, φ))   for pure states

Bures distance:
    d_B(ψ, φ) = √(2 − 2√F(ψ, φ))

These satisfy the relationships:
    T ≤ d_B  and  d_B² = 2T for pure states.

For approximate representations (MPS with truncation), the accumulated
approximation_error in StateMetadata is a lower bound on the trace distance
from the exact state.
"""
from __future__ import annotations

import numpy as np


def state_fidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """Return |⟨sv1|sv2⟩|² for two pure statevectors.

    Parameters
    ----------
    sv1, sv2:
        Complex statevectors of equal length.  Need not be normalised
        (fidelity is computed from the normalised versions).

    Returns
    -------
    float
        Fidelity in [0, 1].  1.0 means identical states (up to global phase).
    """
    a = np.asarray(sv1, dtype=np.complex128).ravel()
    b = np.asarray(sv2, dtype=np.complex128).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"Statevector shape mismatch: {a.shape} vs {b.shape}"
        )
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-15 or norm_b < 1e-15:
        raise ValueError("Cannot compute fidelity with a zero-norm statevector")
    overlap = np.dot(a.conj(), b) / (norm_a * norm_b)
    return float(np.clip((overlap.conj() * overlap).real, 0.0, 1.0))


def trace_distance(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """Return the trace distance T(ρ₁, ρ₂) = √(1 − F) for pure states.

    The trace distance equals the maximum probability of distinguishing the
    two states in any single-shot measurement.  Range: [0, 1].
    """
    F = state_fidelity(sv1, sv2)
    return float(np.sqrt(max(0.0, 1.0 - F)))


def bures_distance(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """Return the Bures distance d_B = √(2 − 2√F) for pure states.

    The Bures distance is a Riemannian metric on the space of quantum
    states.  Range: [0, √2].
    """
    F = state_fidelity(sv1, sv2)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * np.sqrt(F))))


def infidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """Return 1 − F(sv1, sv2) — the error probability under optimal measurement."""
    return 1.0 - state_fidelity(sv1, sv2)


def fidelity_from_density_matrices(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Return F(ρ, σ) = (Tr √(√ρ σ √ρ))² for mixed states.

    Parameters
    ----------
    rho, sigma:
        Hermitian positive-semidefinite density matrices of equal shape.

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    rho = np.asarray(rho, dtype=np.complex128)
    sigma = np.asarray(sigma, dtype=np.complex128)
    if rho.shape != sigma.shape or rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho and sigma must be square matrices of equal size")
    # √ρ via eigendecomposition (ρ is Hermitian PSD)
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)
    sqrt_rho = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
    M = sqrt_rho @ sigma @ sqrt_rho
    # Eigenvalues of M (Hermitian PSD)
    eigs = np.linalg.eigvalsh(M)
    eigs = np.clip(eigs, 0.0, None)
    return float(np.clip(np.sum(np.sqrt(eigs)) ** 2, 0.0, 1.0))
