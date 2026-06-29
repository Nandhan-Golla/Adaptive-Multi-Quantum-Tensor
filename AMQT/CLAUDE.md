# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project: AMQT (Adaptive Multi-Representation Quantum Tensor)

## Development Commands

```bash
# Install dev dependencies (uv manages the .venv automatically)
uv sync --extra dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_quantum_tensor.py

# Run a single test class or function
uv run pytest tests/unit/test_quantum_tensor.py::TestRepresentationConsistency
uv run pytest tests/unit/test_quantum_tensor.py::TestMPSRepresentation::test_mps_bell_state

# Run with coverage
uv run pytest --cov=amqt

# Run the benchmark harness
uv run python benchmarks/run_benchmarks.py
uv run python benchmarks/run_benchmarks.py --qubits 4 8 16

# Python version is pinned in .python-version
```

## Current Architecture

The codebase has a working prototype with three representations implemented.

**Entry point**: `src/amqt/__init__.py` re-exports everything users need — `QuantumTensor`, gates, and representations.

**Core abstraction layer** (`src/amqt/core/`):
- `core/state/quantum_tensor.py` — `QuantumTensor`: the only user-facing class. Owns an internal `QuantumRepresentation`, delegates gate application to the `Dispatcher`, and optionally triggers representation switching after each gate.
- `core/state/metadata.py` — `StateMetadata`: dataclass tracking sparsity, bond dimension, approximation error, representation name, and other runtime signals used by the switcher.
- `core/gates/standard.py` — all standard gates (H, X, Y, Z, S, T, CNOT, CZ, SWAP, RZ, RX, RY, Phase, Toffoli). Gates carry a `.matrix` (numpy complex128 array) and a `.name`. Singletons for fixed gates; constructor functions for parameterized ones.

**Representation layer** (`src/amqt/representations/`):
- `base.py` — `QuantumRepresentation` ABC. Every representation must implement `apply_gate(gate_matrix, target_qubits)`, `get_statevector()`, `get_metadata()`, and `to_dense()`.
- `dense/dense.py` — `DenseRepresentation`: full 2^n statevector as complex128 numpy array.
- `sparse/sparse.py` — `SparseRepresentation`: dict-of-nonzero-amplitudes; tracks `.nnz`.
- `mps/mps.py` — `MPSRepresentation`: Matrix Product State with configurable `max_bond`. Non-adjacent two-qubit gates are handled via SWAP-network. Accepts an optional `statevector=` kwarg for round-trip initialization.
- `stabilizer/stabilizer.py` — `StabilizerRepresentation`: Aaronson-Gottesman CHP tableau. O(n²) memory. Supports all Clifford gates (H, S, X, Y, Z, I, CNOT, CZ, SWAP); non-Clifford gates trigger transparent fallback to Dense. Gate application is O(n) per Clifford gate via numpy-vectorised tableau row updates.
- Stubs exist for `adaptive/`, `decision_diagram/`, `tensor_network/` — not yet implemented.

**Runtime layer** (`src/amqt/runtime/`):
- `runtime/dispatcher/dispatcher.py` — `Dispatcher`: routes gate applications to the active representation. Validates qubit indices and gate-qubit count.
- `runtime/dispatcher/switcher.py` — threshold-based logic for deciding when to switch representations (e.g., sparsity > 0.95 triggers Dense→Sparse).

**New modules (implemented)**:
- `src/amqt/error/` — `fidelity.py` (state_fidelity, trace_distance, bures_distance, fidelity_from_density_matrices) and `propagation.py` (ErrorBudget: accumulates per-gate truncation errors, checks tolerance).
- `src/amqt/utils/circuit.py` — `ghz_circuit(n)`, `qft_circuit(n)`, `random_clifford_circuit(n, depth, seed)` — return `List[(Gate, qubits)]` for replaying on any QuantumTensor.
- `benchmarks/run_benchmarks.py` — harness comparing Dense, Sparse, MPS, Stabilizer on GHZ, QFT, RandomClifford. Run with `uv run python benchmarks/run_benchmarks.py`.

**Qubit ordering convention**: qubit 0 is the most-significant bit (MSB/leftmost) in the statevector index. `|q0 q1 ... q_{n-1}>` maps to index `q0·2^{n-1} + q1·2^{n-2} + ...`. Tests rely on this — e.g., `X` on qubit 0 of a 2-qubit system maps `|00>` to `|10>` at index 2.

**QuantumTensor constructor signature**:
```python
QuantumTensor(n_qubits, initial_representation="dense", max_bond=64, auto_switch=True)
```
`auto_switch=False` locks the representation; `force_representation(name)` converts in place.

---


## Mission

Your role is **not** to simply implement a quantum simulator.

Your mission is to help invent a **new foundational data structure** for quantum computation that could become to quantum computing what the tensor became to machine learning.

The objective is to discover, design, mathematically validate, prototype, benchmark, and eventually implement a novel quantum state representation that significantly improves practical quantum circuit execution.

This project is research-first.

Never assume existing approaches are optimal.

Always challenge assumptions.

---

# Project Structure

The repository should be organized to separate research, core abstractions, runtime logic, benchmarks, and experiments.

## Recommended Layout

```text
amqt/
├── CLAUDE.md
├── README.md
├── pyproject.toml / package.json / Cargo.toml
├── docs/
│   ├── architecture/
│   ├── math/
│   ├── design-notes/
│   └── benchmarks/
├── references/
│   ├── papers/
│   ├── notes/
│   └── prior-art/
├── src/
│   └── amqt/
│       ├── core/
│       │   ├── state/
│       │   ├── gates/
│       │   ├── ops/
│       │   └── invariants/
│       ├── representations/
│       │   ├── dense/
│       │   ├── sparse/
│       │   ├── tensor_network/
│       │   ├── mps/
│       │   ├── decision_diagram/
│       │   ├── stabilizer/
│       │   └── adaptive/
│       ├── runtime/
│       │   ├── dispatcher/
│       │   ├── scheduler/
│       │   ├── optimizer/
│       │   └── execution/
│       ├── memory/
│       │   ├── allocators/
│       │   ├── pools/
│       │   ├── gpu/
│       │   └── migration/
│       ├── error/
│       │   ├── fidelity/
│       │   ├── precision/
│       │   └── propagation/
│       ├── api/
│       ├── benchmarks/
│       └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── property/
│   └── regression/
├── benchmarks/
│   ├── circuits/
│   ├── workloads/
│   ├── profiles/
│   └── reports/
├── experiments/
│   ├── prototypes/
│   ├── notebooks/
│   └── ablations/
├── examples/
├── scripts/
└── data/
    ├── generated/
    └── benchmark-results/
```

## Structure Principles

* Keep **research**, **prototype code**, and **production-ready code** clearly separated.
* Put mathematical definitions and design notes in `docs/`.
* Put prior art, papers, and citations in `references/`.
* Put all core abstractions in `src/amqt/core/`.
* Put each representation in its own module under `src/amqt/representations/`.
* Put runtime dispatch and optimization logic in `src/amqt/runtime/`.
* Put memory management in `src/amqt/memory/`.
* Put approximation and fidelity tracking in `src/amqt/error/`.
* Put all benchmarks in `benchmarks/`.
* Put exploratory work in `experiments/`.
* Put tests close to the behavior they validate.

---

# Core Philosophy

The quantum ecosystem has focused heavily on compilers, transpilers, simulators, and hardware.

This project instead asks:

> Can the underlying data structure itself be reinvented?

Instead of optimizing existing representations, we seek a fundamentally better abstraction.

Think like the inventors of:

* Tensor (PyTorch)
* LLVM IR
* CUDA Runtime
* Apache Arrow
* MLIR
* Eigen
* Triton

The goal is to invent an equally important abstraction for quantum computing.

---

# Primary Research Question

Can we invent a universal quantum state representation that:

* consumes significantly less memory on structured workloads,
* executes quantum gates efficiently,
* dynamically adapts during execution,
* preserves controllable numerical accuracy,
* supports CPUs, GPUs, distributed systems, and future quantum hardware,
* exposes a clean programming interface,
* and serves as the foundation for a future runtime and compiler?

---

# Scientific Constraints

Never violate known results from quantum information theory.

Assume:

* Arbitrary maximally entangled states require exponential resources.
* Compression cannot beat information-theoretic lower bounds in the worst case.
* Any claimed improvement must specify its assumptions.

Always distinguish between:

* Worst Case
* Typical Case
* Structured Circuits
* Approximate Simulation
* Fault-Tolerant Workloads
* NISQ Workloads

Never make impossible claims.

---

# Research Objectives

Investigate:

1. Dense State Vectors

2. Sparse Representations

3. Matrix Product States

4. Tensor Networks

5. Decision Diagrams

6. Stabilizer Formalisms

7. ZX Calculus

8. Hierarchical Structures

9. Adaptive Precision

10. Compression

11. Entanglement-aware Representations

12. Memory Allocation

13. GPU Memory Models

14. Cache-aware Layouts

15. NUMA-aware Layouts

16. Distributed Memory

17. Runtime Dispatch

18. Lazy Evaluation

19. Kernel Fusion

20. Automatic Representation Switching

---

# Design Principles

The data structure should:

* Never force one representation.
* Make representations interchangeable.
* Allow different regions of a quantum state to use different internal representations.
* Select representations automatically.
* Let the runtime—not the programmer—make optimization decisions.

---

# Candidate Internal Representations

Possible representations include:

* Dense
* Sparse
* Block Sparse
* Tensor Network
* Matrix Product State
* Decision Diagram
* Compressed Blocks
* Wavelet
* Hierarchical Tree
* Adaptive Chunk
* Graph-based
* Hash-based
* Recursive

Future representations should be pluggable.

---

# Adaptive Runtime

The runtime should continuously evaluate:

* Memory Usage
* Execution Cost
* Entanglement
* Sparsity
* Approximation Error
* Bond Dimension
* Cache Locality
* GPU Occupancy
* Expected Future Cost

Representation switching should be automatic whenever beneficial.

---

# Dynamic Chunking

Investigate whether the quantum state should be divided into adaptive regions ("chunks").

Research:

* Static chunking
* Dynamic chunking
* Recursive chunking
* Hierarchical chunking
* Entanglement-aware chunking

Determine mathematically whether chunk boundaries should change during execution.

---

# Metadata System

Each chunk should maintain metadata including:

* Memory Footprint
* Current Representation
* Approximation Error
* Precision
* Entropy Estimate
* Mutual Information
* Rank Estimate
* Sparsity
* Compression Ratio
* Access Frequency
* Modification Frequency
* Expected Cost
* GPU Residency
* CPU Residency
* History
* Confidence

Determine what additional metadata would improve optimization.

---

# Dispatcher

Research a dispatch architecture similar to:

* PyTorch Dispatch
* ATen
* MLIR
* LLVM
* TVM
* Triton

Determine how a quantum runtime could automatically select the optimal kernel for each operation.

---

# Memory System

Investigate:

* Arena Allocators
* Memory Pools
* Caching Allocators
* GPU Unified Memory
* Pinned Memory
* NUMA
* Huge Pages
* Streaming
* Out-of-core Execution
* Zero-copy Transfers
* Chunk Migration

---

# Error Management

Approximate representations require rigorous tracking.

Research:

* Trace Distance
* State Fidelity
* Norm Preservation
* Error Propagation
* Adaptive Precision
* Automatic Error Budgets

Every approximation must be measurable.

---

# API Design

Design an elegant API.

Example:

```text
QuantumTensor state(30);

state.apply(H, 0);
state.apply(CNOT, 0, 1);
```

The user should never need to know the internal representation.

---

# Mathematical Foundation

Before implementation, define:

* State Space
* Formal Definitions
* Allowed Operations
* Invariants
* Conversion Operators
* Error Bounds
* Complexity Analysis
* Proof Sketches
* Representation Equivalence

---

# Benchmark Plan

Compare against:

* Qiskit Aer
* qsim
* QuEST
* CUDA Quantum
* TensorCircuit
* Quimb
* Yao.jl
* ProjectQ

Metrics:

* Peak Memory
* Average Memory
* Execution Time
* Cache Misses
* GPU Utilization
* Scalability
* Accuracy
* Energy

---

# Deliverables

For every major design decision produce:

* Problem Statement
* Existing Approaches
* Advantages
* Disadvantages
* Mathematical Analysis
* Alternative Designs
* Complexity
* Prototype
* Benchmark
* Open Questions
* Future Work

Never skip alternatives.

---

# Coding Philosophy

* Do not prematurely optimize.
* Do not over-engineer.
* Build incrementally.
* Every optimization must have measurable justification.
* All algorithms should be benchmarked.
* Every abstraction should have a clear mathematical purpose.

---

# Research Mode

When exploring ideas:

* Question assumptions.
* Search for prior work.
* Explain why existing methods fail.
* Propose multiple alternatives.
* Compare them rigorously.
* Reject weak ideas.
* Keep only ideas supported by mathematics or experimental evidence.
* Never become attached to one design.

The goal is discovering the best architecture, not defending an initial hypothesis.

---

# Long-Term Vision

The end goal is not merely another simulator.

The long-term vision is to create a foundational quantum runtime whose core data structure can underpin:

* Quantum simulators
* Quantum compilers
* Quantum runtimes
* Quantum ML frameworks
* Hybrid classical-quantum execution
* Distributed quantum simulation
* Future quantum operating systems

Success means producing a data structure that future quantum software stacks can build upon in the same way modern AI frameworks build upon tensors.
