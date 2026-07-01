
from __future__ import annotations

import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
sys.path.insert(0, "src")

from amqt import QuantumTensor, H, CNOT, S, X
from amqt.utils.circuit import ghz_circuit, qft_circuit, random_clifford_circuit
from amqt.error.fidelity import state_fidelity
from amqt.representations.stabilizer.stabilizer import StabilizerRepresentation

Circuit = List[Tuple]

@dataclass
class BenchResult:
    rep: str
    circuit: str
    n_qubits: int
    n_gates: int
    wall_time_s: float
    peak_memory_kb: float
    fidelity: Optional[float]
    approx_error: float


def _run_circuit_on_rep(
    n: int,
    circuit: Circuit,
    rep_name: str,
    reference_sv: Optional[np.ndarray] = None,
) -> BenchResult:
    tracemalloc.start()
    t0 = time.perf_counter()

    if rep_name == "stabilizer":
        from amqt.representations.dense.dense import DenseRepresentation
        rep = StabilizerRepresentation(n)
        for gate, qubits in circuit:
            rep.apply_gate(gate.matrix, list(qubits))
        sv = rep.get_statevector()
        meta = rep.get_metadata()
        approx_error = 0.0
    else:
        state = QuantumTensor(n, initial_representation=rep_name, auto_switch=False)
        for gate, qubits in circuit:
            state.apply(gate, *qubits)
        sv = state.statevector()
        meta = state.metadata
        approx_error = meta.approximation_error

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    fid = None
    if reference_sv is not None:
        try:
            fid = state_fidelity(reference_sv, sv)
        except Exception:
            fid = None

    return BenchResult(
        rep=rep_name,
        circuit="",
        n_qubits=n,
        n_gates=len(circuit),
        wall_time_s=elapsed,
        peak_memory_kb=peak / 1024,
        fidelity=fid,
        approx_error=approx_error,
    )


WORKLOADS: Dict[str, Callable[[int], Circuit]] = {
    "GHZ": ghz_circuit,
    "RandomClifford(d=10)": lambda n: random_clifford_circuit(n, depth=10, seed=42),
    "QFT": qft_circuit,
}

REPRESENTATIONS = ["dense", "sparse", "mps", "stabilizer"]

def _fmt(v: Optional[float], width: int = 10, decimals: int = 4) -> str:
    if v is None:
        return " " * (width - 2) + "N/A"
    return f"{v:{width}.{decimals}f}"


def _print_table(results: List[BenchResult], circuit_name: str, n: int) -> None:
    header = f"\n{'─'*70}\n  Circuit: {circuit_name}   n_qubits={n}\n{'─'*70}"
    print(header)
    col = "{:<12} {:>10} {:>14} {:>10} {:>10}"
    print(col.format("Rep", "Time (s)", "Mem (KB)", "Fidelity", "ApxErr"))
    print("─" * 60)
    for r in results:
        fid_str = f"{r.fidelity:.6f}" if r.fidelity is not None else "    N/A   "
        print(
            f"  {r.rep:<10}  {r.wall_time_s:>10.4f}  "
            f"{r.peak_memory_kb:>12.1f}  "
            f"{fid_str:>10}  "
            f"{r.approx_error:>10.2e}"
        )
    print()

def run(qubit_sizes: List[int] = (4, 8, 12)) -> None:
    print("=" * 70)
    print("  AMQT Benchmark Suite")
    print("=" * 70)

    all_results: List[BenchResult] = []

    for circuit_name, circuit_fn in WORKLOADS.items():
        for n in qubit_sizes:
            circuit = circuit_fn(n)

            ref_result = _run_circuit_on_rep(n, circuit, "dense", reference_sv=None)
            state_ref = QuantumTensor(n, initial_representation="dense", auto_switch=False)
            for gate, qubits in circuit:
                state_ref.apply(gate, *qubits)
            ref_sv = state_ref.statevector()

            row_results = [ref_result]
            row_results[0].fidelity = 1.0 
            row_results[0].circuit = circuit_name

            for rep in REPRESENTATIONS[1:]:
                try:
                    r = _run_circuit_on_rep(n, circuit, rep, reference_sv=ref_sv)
                    r.circuit = circuit_name
                    row_results.append(r)
                except Exception as exc:
                    print(f"  [WARN] {rep} failed on {circuit_name} n={n}: {exc}")

            _print_table(row_results, circuit_name, n)
            all_results.extend(row_results)

    print("=" * 70)
    print("  Summary: memory advantage of Stabilizer vs Dense on Clifford circuits")
    print("─" * 70)
    for n in qubit_sizes:
        dense_kb = next(
            (r.peak_memory_kb for r in all_results
             if r.rep == "dense" and r.circuit == "GHZ" and r.n_qubits == n),
            None,
        )
        stab_kb = next(
            (r.peak_memory_kb for r in all_results
             if r.rep == "stabilizer" and r.circuit == "GHZ" and r.n_qubits == n),
            None,
        )
        if dense_kb and stab_kb and stab_kb > 0:
            ratio = dense_kb / stab_kb
            print(f"  n={n:>3}: Dense={dense_kb:.1f} KB  Stabilizer={stab_kb:.1f} KB  ratio={ratio:.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AMQT benchmark harness")
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=int,
        default=[4, 8, 12],
        help="Qubit counts to benchmark (default: 4 8 12)",
    )
    args = parser.parse_args()
    run(qubit_sizes=args.qubits)
