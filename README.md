# Adaptive Multi-Representation Quantum Tensor (AMQT)

---

### 🎓 Research Acknowledgement
This project is part of an active research collaboration between:
* **TRIAC, VIT-AP University**
* **RBG Labs, Indian Institute of Technology Madras (IITM)**

---

<p align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img src="assests/WhatsApp%20Image%202026-07-03%20at%2022.25.51.jpeg" alt="AMQT Logo" width="100%"/>
      </td>
      <td align="center" width="50%">
        <img src="assests/Pasted%20image.png" alt="RBG Labs Logo" width="100%"/>
      </td>
    </tr>
  </table>
</p>

AMQT (Adaptive Multi-Representation Quantum Tensor) is a research-first, novel foundational data structure and simulation framework for quantum computation. Inspired by how the multi-dimensional tensor became the bedrock of modern machine learning frameworks (like PyTorch and JAX), AMQT aims to reinvent the underlying data structures of quantum computing. 

Rather than locking developers into a single state representation, AMQT provides a **unified, representation-agnostic programming interface** (`QuantumTensor`) that dynamically and adaptively switches between different state representations at runtime based on data-driven heuristics, such as sparsity, bipartite entanglement entropy, and gate types.

---

## 🗺️ Project Structure

The repository is structured to maintain a clean separation between core mathematical abstractions, concrete representations, runtime dispatching, and developer tests/benchmarks:

```text
Adaptive-Multi-Quantum-Tensor/
├── pyproject.toml              # Root Python project config (dependencies, pytest options)
├── uv.lock                     # Locked dependencies managed via uv
├── CLAUDE.md                   # Development guide and shortcuts
├── README.md                   # Project documentation (this file)
│
├── amqt-core/                  # Rust library for high-performance CPU kernels
│   ├── Cargo.toml              # Cargo manifest specifying pyo3 and rayon dependencies
│   ├── pyproject.toml          # Maturin configuration for compiling Rust module
│   └── src/
│       └── lib.rs              # Rust implementations for fast 1-qubit and 2-qubit gates
│
├── src/amqt/                   # Core Python package
│   ├── __init__.py             # Exposes QuantumTensor, representations, and gates
│   │
│   ├── core/                   # Basic definitions and abstractions
│   │   ├── state/
│   │   │   ├── quantum_tensor.py # User-facing QuantumTensor wrapper
│   │   │   └── metadata.py     # StateMetadata tracking (sparsity, entropy, memory)
│   │   └── gates/
│   │       └── standard.py     # Standard gate singletons (H, X, Y, Z, S, T, CNOT, CZ, SWAP)
│   │
│   ├── representations/        # Concrete quantum state representations
│   │   ├── base.py             # QuantumRepresentation Abstract Base Class (ABC)
│   │   ├── dense/              # Dense statevector (1D Complex128 array)
│   │   ├── sparse/             # Sparse dictionary (index -> complex amplitude)
│   │   ├── mps/                # Matrix Product State (tensor chain with SVD truncation)
│   │   └── stabilizer/         # Aaronson-Gottesman stabilizer tableau (CHP)
│   │
│   ├── runtime/                # Orchestration & switching logic
│   │   └── dispatcher/
│   │       ├── dispatcher.py   # Routes gates, handles lazy metadata, and queries switcher
│   │       └── switcher.py     # Heuristic switching thresholds and conversion routines
│   │
│   ├── error/                  # Metrics and precision propagation
│   │   ├── fidelity.py         # State fidelity, trace distance, and Bures metrics
│   │   └── propagation.py      # ErrorBudget for tracking cumulative truncation errors
│   │
│   └── utils/
│       └── circuit.py          # Benchmark circuit generators (GHZ, QFT, Random Clifford)
│
├── tests/                      # Unit testing suite
│   ├── unit/
│   │   ├── test_quantum_tensor.py # API validation, consistency, and switching
│   │   ├── test_stabilizer.py     # Tableau logic and fallback validation
│   │   └── test_error.py          # Verification of fidelity and ErrorBudget metrics
│   └── integration/
│
└── benchmarks/
    └── run_benchmarks.py       # Benchmark suite for comparing performance across backends
```

---

## 🧮 Quantum State Representations

An $n$-qubit pure quantum state $|\psi\rangle$ lives in the Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n} \cong \mathbb{C}^{2^n}$, expressed as:
$$|\psi\rangle = \sum_{x \in \{0, 1\}^n} \alpha_x |x\rangle, \quad \sum_x |\alpha_x|^2 = 1$$

AMQT supports four core representations, selecting the optimal one depending on the structure of the state:

### 1. Dense Statevector
* **Definition**: Explicitly stores all $2^n$ complex amplitudes $\alpha_x$ as a flat `complex128` array.
* **Qubit Ordering**: Qubit 0 is the most-significant bit (MSB/leftmost). E.g., $|q_0 q_1 \dots q_{n-1}\rangle$ maps to index $q_0 \cdot 2^{n-1} + q_1 \cdot 2^{n-2} + \dots + q_{n-1}$.
* **Gate Application**: Executes $k$-qubit gates via NumPy tensor contraction or high-performance parallelized Rust kernels (`apply_gate_1q`, `apply_gate_2q`).
* **Complexity**: 
  - Memory: $O(2^n)$ complex numbers ($2^n \times 16$ bytes).
  - Gate Cost: $O(2^k \cdot 2^n)$ FLOPs.
* **When Optimal**: Small $n$ ($n \le 20$), high entanglement, or when zero approximation error is required.

### 2. Sparse Representation
* **Definition**: Stores only non-zero amplitudes ($|\alpha_x| > \varepsilon$, where $\varepsilon = 10^{-10}$) using a dictionary mapping integer basis states to complex values.
* **Gate Application**: Iterates over non-zero amplitudes, applies the gate matrix to targets, and accumulates results in a new dictionary, pruning elements below the threshold $\varepsilon$.
* **Complexity**:
  - Memory: $O(\text{nnz})$ where $\text{nnz}$ is the number of non-zero amplitudes ($\text{nnz} \le 2^n$).
  - Gate Cost: $O(2^k \cdot \text{nnz})$ FLOPs.
* **When Optimal**: States with high sparsity ($\text{sparsity} > 95\%$), such as computational basis states, early circuit stages, and Clifford circuits with sparse non-Clifford insertions.

### 3. Matrix Product State (MPS)
* **Definition**: Represents $|\psi\rangle$ under open boundary conditions as a chain of 3-tensors:
  $$|\psi\rangle = \sum_{s_0, \dots, s_{n-1}} A^{s_0}[0] A^{s_1}[1] \cdots A^{s_{n-1}}[n-1] |s_0 \dots s_{n-1}\rangle$$
  where each tensor $A[i]$ has shape $(\chi_{i-1}, 2, \chi_i)$ with boundary conditions $\chi_0 = \chi_n = 1$. The virtual indices $\chi_i$ are the *bond dimensions*.
* **Gate Application**: Single-qubit gates are applied locally ($O(\chi^2)$). Adjacent two-qubit gates contract adjacent tensors, apply the gate, and decompose back via Singular Value Decomposition (SVD), truncating singular values to $\chi_{\text{max}}$. Non-adjacent gates are routed via SWAP networks.
* **Complexity**:
  - Memory: $O(n \cdot \chi^2)$ complex numbers.
  - Gate Cost: $O(\chi^3)$ for adjacent 2-qubit gates.
* **When Optimal**: Large $n$ ($n \ge 20$ up to $100+$ qubits) with low, localized entanglement.

### 4. Stabilizer Formalism (CHP Tableau)
* **Definition**: Represents Clifford states using the Aaronson-Gottesman stabilizer tableau ($2n \times (2n+1)$ binary matrix). It stores $n$ destabilizers, $n$ stabilizers, and their phases.
* **Gate Application**: Clifford gates ($H$, $S$, $CNOT$, $CZ$, $SWAP$) are applied exactly via binary operations on the tableau. Non-Clifford gates (e.g., $T$, $RZ$, $RX$) automatically trigger a transparent fallback to the `Dense` representation.
* **Complexity**:
  - Memory: $O(n^2)$ bits.
  - Gate Cost: $O(n)$ binary operations per gate.
* **When Optimal**: Purely Clifford circuits or Clifford circuits with very few non-Clifford gates.

---

## 🔄 Dynamic Representation Switching

AMQT orchestrates gates and switching transparently using a `Dispatcher` and a heuristic-based `RepresentationSwitcher`.

### 1. Metadata Signals
Decisions are guided by the `StateMetadata` object, which tracks:
* **Sparsity**: $\rho = 1 - \frac{\text{nnz}}{2^n}$.
* **Bipartite Entropy**: Bipartite von-Neumann entropy $S$ across the $\lfloor n/2 \rfloor$ bipartition:
  $$S = -\sum_i \lambda_i \log_2 \lambda_i$$
  where $\lambda_i$ are the squared singular values of the reshaped statevector. Normalized to $[0, 1]$.
* **Bond Dimension ($\chi$)**: Maximum bond dimension across the MPS chain.
* **Approximation Error**: Cumulative SVD truncation error: $\varepsilon_{\text{total}} \le \sum \sqrt{\sum_{\text{discarded}} \sigma_j^2}$.

### 2. Switching Heuristics
The default switcher enforces the following thresholds to transition between representations:

| Current Rep | Condition | Target Rep | Rationale |
| :--- | :--- | :--- | :--- |
| **Dense** | $\text{Sparsity} > 0.95$ | **Sparse** | Memory savings and faster gate enumeration. |
| **Sparse** | $\text{Sparsity} < 0.70$ | **Dense** | High amplitude density; avoids Python dictionary overhead. |
| **Dense** / **Sparse** | $n \ge 12$ AND $\text{Entropy} > 0.50$ | **MPS** | Large-scale state with structured entanglement. |
| **MPS** | $\text{Entropy} < 0.10$ | **Sparse** / **Dense** | Bounded entanglement has dissolved; cheaper overhead. |
| **Stabilizer** | Encounters non-Clifford gate | **Dense** | Stabilizer formalism breaks; falls back to statevector. |

```
                       ┌─────────────┐
        ┌─────────────►│ Stabilizer  │
        │              └──────┬──────┘
        │ (Explicit           │ (Non-Clifford
        │  Init)              ▼  Fallback)
        │              ┌─────────────┐
        │              │    Dense    ├───────────────┐
        │              └▲───────────┬┘               │
        │               │           │                │
        │      (Sparsity│           │(Sparsity       │(Large system
        │        < 0.70)│           │  > 0.95)       │ & Entropy > 0.50)
        │               │           ▼                ▼
  ┌─────┴─────┐        ┌┴───────────┐          ┌───────────┐
  │   State   │        │   Sparse   ├─────────►│    MPS    │
  └───────────┘        └▲───────────┘ (Entropy └─────┬─────┘
                        │              < 0.10 &      │ (Entropy < 0.10 &
                        └──────────────  Sparsity    │  Sparsity < 0.95)
                                         > 0.95)     ▼
                                               ┌───────────┐
                                               │   Dense   │
                                               └───────────┘
```

---

## ⚡ Execution Guide

### Prerequisites
* **Python**: `3.10` or higher
* **Rust compiler**: Required to build the speed-optimized `amqt-core` gate kernels
* **uv**: We recommend using the [`uv`](https://github.com/astral-sh/uv) fast package manager to manage environments and dependencies.

### 1. Installation
To sync and install the development environment (which automatically compiles the Rust backend and links python paths):

```bash
# Sync package and dev dependencies
uv sync --extra dev
```

### 2. Running Tests
Run the entire unit test suite containing verification for all representations:

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/unit/test_quantum_tensor.py
```

### 3. Running Benchmarks
Evaluate performance and memory across the different representations under GHZ, QFT, and random Clifford workloads:

```bash
# Run the benchmark suite (default sizes: 4, 8, 12 qubits)
uv run python benchmarks/run_benchmarks.py

# Run benchmarks on custom qubit counts
uv run python benchmarks/run_benchmarks.py --qubits 4 8 16
```

---

## 💻 Usage Example

Creating a state, applying gates, and observing automatic representation changes in action:

```python
import numpy as np
from amqt import QuantumTensor, H, CNOT, T

# 1. Initialize a 14-qubit state (defaults to "dense" representation)
# auto_switch=True activates dynamic switching heuristics.
state = QuantumTensor(n_qubits=14, initial_representation="dense", auto_switch=True)
print(f"Initial: {state.representation_name}")  # Output: dense

# 2. Sparsify the state (e.g. by applying identity/basis-like gates)
# Applying H on qubit 0 keeps it sparse.
state.apply(H, 0)
print(f"After H: {state.representation_name} (Sparsity: {state.metadata.sparsity:.2%})")

# 3. Build entanglement (starts switching when entropy threshold is crossed)
# After applying CNOTs, the bipartite entropy increases.
for i in range(13):
    state.apply(CNOT, i, i + 1)
print(f"Entangled: {state.representation_name}")  # Under high entropy, switches to MPS

# 4. Check results
# We can extract the final norm, statevector, or expectation values
print(f"State Norm: {state.norm():.6f}")
print(f"Metadata summary: {state.metadata}")

# 5. Chaining API Support
# We can also chain multiple gate operations
state.apply(H, 0).apply(T, 0).apply(CNOT, 0, 1)

# 6. Retrieve the full statevector (contracts/evaluates representation behind the scenes)
sv = state.statevector()
print(f"Amplitudes shape: {sv.shape}")
```

---

## 📊 Comparison with Prior Art

AMQT distinguishes itself from existing frameworks by focusing on **automatic, runtime, data-driven representation switching** behind a single API:

| Simulator / Framework | Dense SV | Sparse | MPS / TN | DD (Decision Diagrams) | Stabilizer | Adaptive Switching |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qiskit Aer** | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **QuEST** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **qsim** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **CUDA-Q** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **ITensor** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **TeNPy** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Quimb** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **MQT DDSIM** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Yao.jl** | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ (Manual only) |
| **Stim** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **AMQT** | ✅ | ✅ | ✅ | *Planned* | ✅ | **✅ (Automatic)** |

---

## 📜 License
This software is licensed under the **Apache-2.0 License**. See the project files for full terms.
