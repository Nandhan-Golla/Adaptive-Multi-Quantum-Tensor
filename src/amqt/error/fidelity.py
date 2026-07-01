
from __future__ import annotations

import numpy as np


def state_fidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:

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

    F = state_fidelity(sv1, sv2)
    return float(np.sqrt(max(0.0, 1.0 - F)))


def bures_distance(sv1: np.ndarray, sv2: np.ndarray) -> float:

    F = state_fidelity(sv1, sv2)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * np.sqrt(F))))


def infidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:

    return 1.0 - state_fidelity(sv1, sv2)


def fidelity_from_density_matrices(rho: np.ndarray, sigma: np.ndarray) -> float:

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
