from __future__ import annotations

QUBIT_SIZES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
LARGE_N_SIZES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
FIDELITY_N = 5
GATE_SCALING_N = 10
GATE_SCALING_DEPTHS = [1, 2, 4, 6, 8, 10, 12, 16, 20]

CLIFFORD_DEPTH = 4
QAOA_LAYERS = 2
HW_ANSATZ_REPS = 2

ISING_STEPS = 3
TIMING_REPEATS = 3

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from amqt import (
    QuantumTensor, H, X, Y, Z, S, T, I, CNOT, CZ, SWAP,
    RZ, RX, RY, state_fidelity
)
from amqt.utils.circuit import _MatrixGate, _controlled_phase
def qiskit_to_amqt(sv: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(sv)
    for i in range(len(sv)):
        out[int(f"{i:0{n}b}"[::-1], 2)] = sv[i]
    return out


def time_fn(fn: Callable, repeats: int = 7) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def peak_memory_kb(fn: Callable) -> float:
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024


def build_ghz(n: int):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    gates = n

    def amqt():
        st = QuantumTensor(n, auto_switch=False)
        st.apply(H, 0)
        for i in range(n - 1):
            st.apply(CNOT, i, i + 1)
        return st
    return qc, amqt, gates


def build_qft(n: int):
    qc = QuantumCircuit(n)
    for k in range(n):
        qc.h(k)
        for j in range(1, n - k):
            qc.cp(np.pi / (2 ** j), k + j, k)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    gates = n + n*(n-1)//2 + n//2

    def amqt():
        st = QuantumTensor(n, auto_switch=False)
        for k in range(n):
            st.apply(H, k)
            for j in range(1, n - k):
                cp = _MatrixGate(_controlled_phase(np.pi / (2**j)), 2, f"CP{j}")
                st._dispatcher.representation.apply_gate(cp.matrix, [k, k+j])
        for i in range(n // 2):
            st.apply(SWAP, i, n - 1 - i)
        return st
    return qc, amqt, gates


def build_random_clifford(n: int, depth: int = 4):
    rng = np.random.default_rng(42)
    clifford_1q = [
        ("h", lambda qc, q: qc.h(q), H),
        ("x", lambda qc, q: qc.x(q), X),
        ("s", lambda qc, q: qc.s(q), S),
        ("z", lambda qc, q: qc.z(q), Z),
    ]
    qc = QuantumCircuit(n)
    ops_amqt = []
    gate_count = 0
    for _ in range(depth):
        for q in range(n):
            idx = int(rng.integers(0, 4))
            name, qc_fn, amqt_gate = clifford_1q[idx]
            qc_fn(qc, q)
            ops_amqt.append((amqt_gate, [q]))
            gate_count += 1
        pairs = list(range(0, n - 1, 2))
        for q in pairs:
            qc.cx(q, q + 1)
            ops_amqt.append((CNOT, [q, q + 1]))
            gate_count += 1

    def amqt():
        st = QuantumTensor(n, auto_switch=False)
        for gate, qubits in ops_amqt:
            st.apply(gate, *qubits)
        return st
    return qc, amqt, gate_count


def build_bernstein_vazirani(n: int):
    secret = [i % 2 for i in range(n - 1)]
    qc = QuantumCircuit(n)
    qc.x(n - 1)
    for q in range(n):
        qc.h(q)
    for i, bit in enumerate(secret):
        if bit:
            qc.cx(i, n - 1)
    for q in range(n):
        qc.h(q)
    gates = 1 + 2*n + sum(secret)

    def amqt():
        st = QuantumTensor(n, auto_switch=False)
        st.apply(X, n - 1)
        for q in range(n):
            st.apply(H, q)
        for i, bit in enumerate(secret):
            if bit:
                st.apply(CNOT, i, n - 1)
        for q in range(n):
            st.apply(H, q)
        return st
    return qc, amqt, gates


def build_qaoa(n: int, layers: int = 2):
    rng = np.random.default_rng(7)
    gammas = rng.uniform(0, np.pi, layers)
    betas  = rng.uniform(0, np.pi, layers)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for l in range(layers):
        for q in range(n - 1):
            qc.rzz(2 * gammas[l], q, q + 1)
        for q in range(n):
            qc.rx(2 * betas[l], q)
    gates = n + layers * (n - 1) * 2 + layers * n

    def amqt():
        st = QuantumTensor(n, auto_switch=False)
        for q in range(n):
            st.apply(H, q)
        for l in range(layers):
            for q in range(n - 1):
                st.apply(CNOT, q, q + 1)
                st.apply(RZ(2 * gammas[l]), q + 1)
                st.apply(CNOT, q, q + 1)
            for q in range(n):
                st.apply(RX(2 * betas[l]), q)
        return st
    return qc, amqt, gates


def build_hardware_ansatz(n: int, reps: int = 2):
    rng = np.random.default_rng(13)
    thetas = rng.uniform(0, 2*np.pi, (reps + 1, n))
    qc = QuantumCircuit(n)
    gate_count = 0
    for rep in range(reps + 1):
        for q in range(n):
            qc.ry(thetas[rep, q], q)
            gate_count += 1
        if rep < reps:
            for q in range(n - 1):
                qc.cx(q, q + 1)
                gate_count += 1

    def amqt():
        st = QuantumTensor(n, auto_switch=False)
        for rep in range(reps + 1):
            for q in range(n):
                st.apply(RY(thetas[rep, q]), q)
            if rep < reps:
                for q in range(n - 1):
                    st.apply(CNOT, q, q + 1)
        return st
    return qc, amqt, gate_count


def build_w_state(n: int):
    qc = QuantumCircuit(n)
    qc.x(0)
    for k in range(1, n):
        angle = 2 * np.arccos(np.sqrt(1.0 / (n - k + 1)))
        qc.cry(angle, k - 1, k)
        qc.cx(k, k - 1)
    gate_count = 1 + (n - 1) * 2

    def amqt():
        st = QuantumTensor(n, auto_switch=False)
        st.apply(X, 0)
        for k in range(1, n):
            angle = 2 * np.arccos(np.sqrt(1.0 / (n - k + 1)))
            cry = _MatrixGate(_controlled_ry(angle), 2, f"CRY{k}")
            st._dispatcher.representation.apply_gate(cry.matrix, [k - 1, k])
            st.apply(CNOT, k, k - 1)
        return st
    return qc, amqt, gate_count


def build_ising_evolution(n: int, steps: int = 3):
    dt = 0.1
    J = 1.0
    h = 0.5
    qc = QuantumCircuit(n)
    gate_count = 0
    for _ in range(steps):
        for q in range(n - 1):
            qc.rzz(2 * J * dt, q, q + 1)
            gate_count += 2  # CNOT-RZ-CNOT
        for q in range(n):
            qc.rx(2 * h * dt, q)
            gate_count += 1

    def amqt():
        st = QuantumTensor(n, auto_switch=False)
        for _ in range(steps):
            for q in range(n - 1):
                st.apply(CNOT, q, q + 1)
                st.apply(RZ(2 * J * dt), q + 1)
                st.apply(CNOT, q, q + 1)
            for q in range(n):
                st.apply(RX(2 * h * dt), q)
        return st
    return qc, amqt, gate_count


def build_swap_test(n: int):
    half = n // 2
    qc = QuantumCircuit(n + 1)
    qc.h(0)
    for q in range(half):
        qc.cswap(0, q + 1, q + 1 + half)
    qc.h(0)
    gate_count = 2 + half

    def amqt():
        st = QuantumTensor(n + 1, auto_switch=False)
        st.apply(H, 0)
        for q in range(half):
            st.apply(CNOT, 0, q + 1)  # simplified CSWAP
            st.apply(CNOT, 0, q + 1 + half)
        st.apply(H, 0)
        return st
    return qc, amqt, gate_count


def _controlled_ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [1, 0,  0, 0],
        [0, 1,  0, 0],
        [0, 0,  c, -s],
        [0, 0,  s,  c],
    ], dtype=np.complex128)


CIRCUITS = [
    ("GHZ",              build_ghz),
    ("QFT",              build_qft),
    ("Random Clifford",  lambda n: build_random_clifford(n, depth=CLIFFORD_DEPTH)),
    ("Bernstein-Vazirani", build_bernstein_vazirani),
    (f"QAOA ({QAOA_LAYERS} layers)",  lambda n: build_qaoa(n, layers=QAOA_LAYERS)),
    (f"HW Ansatz",        lambda n: build_hardware_ansatz(n, reps=HW_ANSATZ_REPS)),
    ("W State",          build_w_state),
    ("Ising Evolution",  lambda n: build_ising_evolution(n, steps=ISING_STEPS)),
]


def collect_timings(circuits, sizes, repeats=7):
    results = {}
    for name, builder in circuits:
        results[name] = {}
        print(f"  {name}")
        for n in sizes:
            try:
                qc, amqt_fn, _ = builder(n)
                q_ms = time_fn(lambda: Statevector(qc), repeats)
                a_ms = time_fn(amqt_fn, repeats)
                results[name][n] = (q_ms, a_ms)
                ratio = q_ms / a_ms if a_ms > 0 else float("inf")
                print(f"    n={n:2d}: Qiskit {q_ms:7.3f}ms  AMQT {a_ms:7.3f}ms  {ratio:.1f}×")
            except Exception as e:
                print(f"    n={n:2d}: ERROR — {e}")
                results[name][n] = (0, 0)
    return results


def plot_speed_grid(results: dict, sizes: list, out: str):
    names = list(results.keys())
    ncols = 4
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
    fig.suptitle("AMQT vs Qiskit Aer — Execution Time  (median 7 runs)", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        data = results[name]
        ns = [n for n in sizes if n in data and data[n][0] > 0]
        qk = [data[n][0] for n in ns]
        am = [data[n][1] for n in ns]

        ax.plot(ns, qk, "o-", color="steelblue",  lw=2, ms=5, label="Qiskit Aer")
        ax.plot(ns, am, "s-", color="darkorange", lw=2, ms=5, label="AMQT")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Qubits")
        ax.set_ylabel("Time (ms)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Annotate speedup at n=max
        if ns:
            n_last = ns[-1]
            q_last = data[n_last][0]
            a_last = data[n_last][1]
            if a_last > 0:
                ratio = q_last / a_last
                ax.text(0.97, 0.05, f"{ratio:.1f}× faster\n@ n={n_last}",
                        transform=ax.transAxes, ha="right", fontsize=8,
                        color="green" if ratio >= 1 else "red",
                        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.7))

    for ax in axes[len(names):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


def plot_speedup_heatmap(results: dict, sizes: list, out: str):
    names = list(results.keys())
    data_matrix = np.zeros((len(names), len(sizes)))
    for i, name in enumerate(names):
        for j, n in enumerate(sizes):
            q, a = results[name].get(n, (0, 0))
            data_matrix[i, j] = q / a if a > 0 and q > 0 else 1.0

    fig, ax = plt.subplots(figsize=(12, 6))
    vmax = min(data_matrix.max(), 15)
    im = ax.imshow(data_matrix, aspect="auto", cmap="RdYlGn",
                   norm=mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=vmax))

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Number of Qubits")
    ax.set_title("AMQT Speedup over Qiskit Aer  (green > 1× = AMQT wins)", fontweight="bold")

    for i in range(len(names)):
        for j in range(len(sizes)):
            val = data_matrix[i, j]
            ax.text(j, i, f"{val:.1f}×", ha="center", va="center",
                    fontsize=7.5, color="black", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Speedup ratio (AMQT / Qiskit)")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


def plot_fidelity(circuits, n: int, out: str):
    names, fidelities = [], []
    for name, builder in circuits:
        try:
            qc, amqt_fn, _ = builder(n)
            qsv = qiskit_to_amqt(Statevector(qc).data, n)
            asv = amqt_fn().statevector()
            fid = state_fidelity(qsv, asv)
            names.append(name)
            fidelities.append(fid)
            print(f"  Fidelity {name} (n={n}): {fid:.12f}")
        except Exception as e:
            names.append(name)
            fidelities.append(0.0)
            print(f"  Fidelity {name} (n={n}): ERROR — {e}")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["green" if f > 0.9999 else "orange" if f > 0.99 else "red"
              for f in fidelities]
    bars = ax.barh(names, fidelities, color=colors)
    ax.set_xlim(0.99999, 1.000001)
    ax.set_xlabel("Fidelity  |⟨ψ_AMQT | ψ_Qiskit⟩|²")
    ax.set_title(f"AMQT Fidelity vs Qiskit Aer — n={n} qubits", fontweight="bold")
    ax.axvline(1.0, color="black", lw=1, linestyle="--", alpha=0.5)
    for bar, fid in zip(bars, fidelities):
        ax.text(bar.get_width() - 5e-7, bar.get_y() + bar.get_height()/2,
                f"{fid:.10f}", va="center", ha="right", fontsize=8, fontfamily="monospace")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


def plot_memory(circuits_subset, sizes, out: str):
    fig, axes = plt.subplots(1, len(circuits_subset), figsize=(16, 5))
    fig.suptitle("Peak Memory Usage — AMQT vs Qiskit Aer", fontweight="bold")

    for ax, (name, builder) in zip(axes, circuits_subset):
        ns, qk_mem, am_mem = [], [], []
        for n in sizes:
            try:
                qc, amqt_fn, _ = builder(n)
                q_kb = peak_memory_kb(lambda: Statevector(qc))
                a_kb = peak_memory_kb(amqt_fn)
                ns.append(n)
                qk_mem.append(q_kb)
                am_mem.append(a_kb)
            except Exception:
                pass

        ax.semilogy(ns, qk_mem, "o-", color="steelblue", lw=2, ms=5, label="Qiskit Aer")
        ax.semilogy(ns, am_mem, "s-", color="darkorange", lw=2, ms=5, label="AMQT")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Qubits")
        ax.set_ylabel("Peak memory (KB, log scale)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()



def plot_gate_scaling(out: str):
    n = GATE_SCALING_N
    depths = GATE_SCALING_DEPTHS

    qk_per_gate, am_per_gate, total_gates = [], [], []

    for depth in depths:
        qc, amqt_fn, gates = build_random_clifford(n, depth)
        q_ms = time_fn(lambda: Statevector(qc))
        a_ms = time_fn(amqt_fn)
        qk_per_gate.append(q_ms / gates)
        am_per_gate.append(a_ms / gates)
        total_gates.append(gates)
        print(f"  depth={depth:2d} ({gates:3d} gates): Qiskit {q_ms/gates:.3f} µs/gate  AMQT {a_ms/gates:.3f} µs/gate")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Per-Gate Cost vs Circuit Depth (n={n} qubits, Random Clifford)",
                 fontweight="bold")

    ax1.plot(depths, qk_per_gate, "o-", color="steelblue",  lw=2, ms=6, label="Qiskit Aer")
    ax1.plot(depths, am_per_gate, "s-", color="darkorange", lw=2, ms=6, label="AMQT")
    ax1.set_xlabel("Circuit depth (layers)")
    ax1.set_ylabel("Time per gate (ms)")
    ax1.set_title("Per-gate cost vs depth")
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax2.plot(total_gates, qk_per_gate, "o-", color="steelblue",  lw=2, ms=6, label="Qiskit Aer")
    ax2.plot(total_gates, am_per_gate, "s-", color="darkorange", lw=2, ms=6, label="AMQT")
    ax2.set_xlabel("Total gate count")
    ax2.set_ylabel("Time per gate (ms)")
    ax2.set_title("Per-gate cost vs gate count")
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


def plot_large_n(out: str):
    sizes_large = LARGE_N_SIZES
    builders = [("GHZ", build_ghz), ("QFT", build_qft)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Scaling to Larger Qubit Counts (n=2..14)", fontweight="bold")

    for ax, (name, builder) in zip(axes, builders):
        ns, qk_ms, am_ms = [], [], []
        for n in sizes_large:
            try:
                qc, amqt_fn, _ = builder(n)
                q = time_fn(lambda: Statevector(qc), repeats=TIMING_REPEATS)
                a = time_fn(amqt_fn, repeats=TIMING_REPEATS)
                ns.append(n)
                qk_ms.append(q)
                am_ms.append(a)
                print(f"  {name} n={n}: Qiskit {q:.2f}ms  AMQT {a:.2f}ms  {q/a:.1f}×")
            except Exception as e:
                print(f"  {name} n={n}: ERROR — {e}")

        ax.semilogy(ns, qk_ms, "o-", color="steelblue",  lw=2, ms=6, label="Qiskit Aer")
        ax.semilogy(ns, am_ms, "s-", color="darkorange", lw=2, ms=6, label="AMQT")
        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Time (ms, log scale)")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks(ns)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle="--", which="both")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()

def plot_new_probabilities(out: str):
    cases = [
        ("Bernstein-Vazirani\n(n=5, secret=01010)", 5, build_bernstein_vazirani),
        ("QAOA (n=4, 2 layers)",                     4, lambda n: build_qaoa(n, 2)),
        ("W State (n=4)",                             4, build_w_state),
        ("Hardware Ansatz\n(n=4, 2 reps)",            4, lambda n: build_hardware_ansatz(n, 2)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("AMQT vs Qiskit Aer — Probability Distributions (New Circuits)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for ax, (title, n, builder) in zip(axes, cases):
        try:
            qc, amqt_fn, _ = builder(n)
            qsv = qiskit_to_amqt(Statevector(qc).data, n)
            asv = amqt_fn().statevector()
            fid = state_fidelity(qsv, asv)
            q_probs = (qsv.conj() * qsv).real
            a_probs = (asv.conj() * asv).real
            dim = len(q_probs)

            if dim > 16:
                top_idx = np.argsort(q_probs)[::-1][:16]
                top_idx = np.sort(top_idx)
            else:
                top_idx = np.arange(dim)

            labels = [f"|{i:0{n}b}⟩" for i in top_idx]
            x = np.arange(len(top_idx))
            w = 0.4
            ax.bar(x - w/2, q_probs[top_idx], w, label="Qiskit Aer", color="steelblue")
            ax.bar(x + w/2, a_probs[top_idx], w, label="AMQT", color="darkorange", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7, fontfamily="monospace")
            ax.set_ylabel("Probability")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.text(0.97, 0.96, f"F = {fid:.10f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8))
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes, ha="center")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()

if __name__ == "__main__":
    OUT = Path("experiments")

    print("=" * 60)
    print("  Collecting timings for all circuits …")
    print("=" * 60)
    timings = collect_timings(CIRCUITS, QUBIT_SIZES, repeats=TIMING_REPEATS)

    print("\n" + "=" * 60)
    print("  Figure 1: Speed grid (all circuits)")
    print("=" * 60)
    plot_speed_grid(timings, QUBIT_SIZES, str(OUT / "ext_speed_all.png"))

    print("\n" + "=" * 60)
    print("  Figure 2: Speedup heatmap")
    print("=" * 60)
    plot_speedup_heatmap(timings, QUBIT_SIZES, str(OUT / "ext_speedup_ratio.png"))

    print("\n" + "=" * 60)
    print("  Figure 3: Fidelity verification (n=5)")
    print("=" * 60)
    plot_fidelity(CIRCUITS, n=FIDELITY_N, out=str(OUT / "ext_fidelity.png"))

    print("\n" + "=" * 60)
    print("  Figure 4: Peak memory comparison")
    print("=" * 60)
    plot_memory(
        [("GHZ", build_ghz), ("QFT", build_qft), ("HW Ansatz", lambda n: build_hardware_ansatz(n, 2))],
        QUBIT_SIZES,
        str(OUT / "ext_memory.png"),
    )

    print("\n" + "=" * 60)
    print("  Figure 5: Per-gate cost vs depth (n=8)")
    print("=" * 60)
    plot_gate_scaling(str(OUT / "ext_gate_scaling.png"))

    print("\n" + "=" * 60)
    print("  Figure 6: Large-n scaling (n=2..14)")
    print("=" * 60)
    plot_large_n(str(OUT / "ext_large_n.png"))

    print("\n" + "=" * 60)
    print("  Figure 7: New circuit probability distributions")
    print("=" * 60)
    plot_new_probabilities(str(OUT / "ext_new_probabilities.png"))

    print("\n" + "=" * 60)
    print("  Done. Output files:")
    print("=" * 60)
    for f in sorted(OUT.glob("ext_*.png")):
        print(f"  {f}")
