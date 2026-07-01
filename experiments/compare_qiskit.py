from __future__ import annotations


SPEED_QUBIT_SIZES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

PROB_BELL_N   = 2    # Bell state  (must stay 2)
PROB_GHZ_N    = 5    # GHZ state
PROB_QFT_N    = 4    # QFT
PROB_MIXED_N  = 6    # Mixed Clifford + rotation circuit
TIMING_REPEATS = 3

# ─────────────────────────────────────────────────────────────────────────────

import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from amqt import QuantumTensor, H, X, CNOT, CZ, SWAP, RZ, RX, state_fidelity
from amqt.utils.circuit import _MatrixGate, _controlled_phase


def qiskit_to_amqt_sv(sv: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(sv)
    for i in range(len(sv)):
        rev = int(f"{i:0{n}b}"[::-1], 2)
        out[rev] = sv[i]
    return out

def bell(n: int = 2):
    assert n == 2
    qc = QuantumCircuit(n)
    qc.h(0); qc.cx(0, 1)

    def amqt():
        st = QuantumTensor(n)
        st.apply(H, 0).apply(CNOT, 0, 1)
        return st

    return qc, amqt


def ghz(n: int):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    def amqt():
        st = QuantumTensor(n)
        st.apply(H, 0)
        for i in range(n - 1):
            st.apply(CNOT, i, i + 1)
        return st

    return qc, amqt


def qft(n: int):
    qc = QuantumCircuit(n)
    for k in range(n):
        qc.h(k)
        for j in range(1, n - k):
            qc.cp(np.pi / (2 ** j), k + j, k)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)

    def amqt():
        st = QuantumTensor(n)
        for k in range(n):
            st.apply(H, k)
            for j in range(1, n - k):
                angle = np.pi / (2 ** j)
                cp = _MatrixGate(_controlled_phase(angle), 2, f"CP{j}")
                st._dispatcher.representation.apply_gate(cp.matrix, [k, k + j])
        for i in range(n // 2):
            st.apply(SWAP, i, n - 1 - i)
        return st

    return qc, amqt


def mixed(n: int):
    rng = np.random.default_rng(1337)
    angles = rng.uniform(0, 2 * np.pi, (n,))

    qc = QuantumCircuit(n)
    for i in range(n): qc.h(i)
    for i in range(n): qc.rz(angles[i], i)
    for i in range(n): qc.cx(i, (i + 1) % n)
    for i in range(n): qc.rx(angles[(i + 1) % n], i)
    for i in range(0, n - 1, 2): qc.cz(i, i + 1)
    for i in range(n): qc.h(i)

    def amqt():
        st = QuantumTensor(n)
        for i in range(n): st.apply(H, i)
        for i in range(n): st.apply(RZ(angles[i]), i)
        for i in range(n): st.apply(CNOT, i, (i + 1) % n)
        for i in range(n): st.apply(RX(angles[(i + 1) % n]), i)
        for i in range(0, n - 1, 2): st.apply(CZ, i, i + 1)
        for i in range(n): st.apply(H, i)
        return st

    return qc, amqt



def time_both(qc: QuantumCircuit, amqt_fn: Callable, n: int, repeats: int = 5):
    q_times, a_times = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        Statevector(qc)
        q_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        amqt_fn()
        a_times.append((time.perf_counter() - t0) * 1000)

    return float(np.median(q_times)), float(np.median(a_times))


def plot_probabilities():
    fixed_circuits = [
        (f"Bell State  (n={PROB_BELL_N})\nH(0)·CNOT(0,1)",        PROB_BELL_N,  bell),
        (f"GHZ State  (n={PROB_GHZ_N})",                           PROB_GHZ_N,  ghz),
        (f"QFT on |0⟩^{PROB_QFT_N}  (n={PROB_QFT_N})",            PROB_QFT_N,  qft),
        (f"Mixed Circuit  (n={PROB_MIXED_N})\nClifford + rotations", PROB_MIXED_N, mixed),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AMQT vs Qiskit Aer — Probability Distributions", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax, (title, n, builder) in zip(axes, fixed_circuits):
        qc, amqt_fn = builder(n)
        qsv = qiskit_to_amqt_sv(Statevector(qc).data, n)
        asv = amqt_fn().statevector()
        fid = state_fidelity(qsv, asv)

        q_probs = (qsv.conj() * qsv).real
        a_probs = (asv.conj() * asv).real
        dim = len(q_probs)
        MAX = 32
        if dim <= MAX:
            idx = np.arange(dim)
            labels = [f"|{i:0{n}b}⟩" for i in idx]
            qp, ap = q_probs, a_probs
        else:
            half = MAX // 2
            idx = np.concatenate([np.arange(half), np.arange(dim - half, dim)])
            labels = [f"|{i:0{n}b}⟩" for i in idx]
            qp = q_probs[idx]
            ap = a_probs[idx]

        x = np.arange(len(idx))
        w = 0.4
        ax.bar(x - w/2, qp, w, label="Qiskit Aer", color="steelblue")
        ax.bar(x + w/2, ap, w, label="AMQT",        color="darkorange", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7, fontfamily="monospace")
        ax.set_ylabel("Probability")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.text(
            0.98, 0.97,
            f"Fidelity = {fid:.10f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.8),
        )

    plt.tight_layout()
    out = "experiments/prob_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()



def plot_speed():
    benchmarks = [
        ("GHZ Circuit",    ghz,   SPEED_QUBIT_SIZES),
        ("QFT Circuit",    qft,   SPEED_QUBIT_SIZES),
        ("Mixed Circuit",  mixed, SPEED_QUBIT_SIZES),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Execution Time vs Qubit Count — AMQT vs Qiskit Aer  (median of 5 runs)",
        fontsize=13, fontweight="bold",
    )

    for ax, (label, builder, sizes) in zip(axes, benchmarks):
        q_ms_list, a_ms_list = [], []

        print(f"\n  {label}")
        print(f"  {'n':>4}  {'Qiskit (ms)':>12}  {'AMQT (ms)':>12}  {'ratio':>8}")
        print("  " + "-" * 44)

        for n in sizes:
            qc, amqt_fn = builder(n)
            q_ms, a_ms = time_both(qc, amqt_fn, n, repeats=TIMING_REPEATS)
            q_ms_list.append(q_ms)
            a_ms_list.append(a_ms)
            ratio = q_ms / a_ms if a_ms > 0 else float("inf")
            print(f"  {n:>4}  {q_ms:>12.3f}  {a_ms:>12.3f}  {ratio:>7.2f}x")

        ax.plot(sizes, q_ms_list, "o-", label="Qiskit Aer", color="steelblue",    linewidth=2, markersize=6)
        ax.plot(sizes, a_ms_list, "s-", label="AMQT",        color="darkorange",   linewidth=2, markersize=6)

        if 10 in sizes:
            i10 = sizes.index(10)
            ax.annotate(
                f"{q_ms_list[i10]:.1f} ms",
                xy=(10, q_ms_list[i10]),
                xytext=(9.1, q_ms_list[i10] * 1.08),
                fontsize=8, color="steelblue",
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8),
            )
            ax.annotate(
                f"{a_ms_list[i10]:.1f} ms",
                xy=(10, a_ms_list[i10]),
                xytext=(9.1, a_ms_list[i10] * 0.6),
                fontsize=8, color="darkorange",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8),
            )

        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Time (ms)")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks(sizes)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = "experiments/speed_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close()


if __name__ == "__main__":
    print("=" * 55)
    print("  Plotting probability distributions …")
    print("=" * 55)
    plot_probabilities()

    print("\n" + "=" * 55)
    print("  Timing circuits from n=2 to n=10 …")
    print("=" * 55)
    plot_speed()

    print("\nDone.")
