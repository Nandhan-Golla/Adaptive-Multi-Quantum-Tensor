# Prior Art: Quantum State Simulation

This document surveys the existing landscape of quantum circuit simulators
and state representations that AMQT builds upon, differentiates from, or
is inspired by.

---

## 1. Dense Statevector Simulators

### 1.1 Qiskit Aer — `AerSimulator` (statevector method)

**Origin**: IBM Quantum / Qiskit project.
**Approach**: Dense complex128 statevector stored as a flat array.
  Gate application via custom C++ kernels with AVX-512 SIMD and optional GPU
  acceleration via CUDA.
**Strengths**:
  - Highly optimised single-gate and batched gate kernels.
  - GPU support (cuQuantum backend).
  - Broad gate set and noise model support.
**Weaknesses**:
  - Memory fixed at O(2^n) regardless of state structure.
  - No adaptive representation switching.
  - No approximation / compression.
**What AMQT improves**: Adaptive representation, MPS fallback for large n,
  sparse optimisation for low-entanglement states.

**Reference**: https://github.com/Qiskit/qiskit-aer

---

### 1.2 QuEST (Quantum Exact Simulation Toolkit)

**Origin**: University of Oxford / ETH Zurich.
**Approach**: Dense statevector with MPI+OpenMP parallelism for distributed-
  memory simulation.  Also supports density matrices.
**Strengths**:
  - Scales to ~40 qubits across HPC clusters (2^40 amplitudes across nodes).
  - Highly portable (C99, no external dependencies).
  - Exact simulation with clean error propagation.
**Weaknesses**:
  - Dense only; no compression or sparsity exploitation.
  - No adaptive representation.
  - Requires exponentially growing cluster as n grows.
**What AMQT improves**: Replacing the hard-coded dense backend with an adaptive
  system that can exploit structure before resorting to full dense simulation.

**Reference**: https://github.com/QuEST-Kit/QuEST

---

### 1.3 qsim (Google)

**Origin**: Google Quantum AI.
**Approach**: Dense statevector optimised for gate fusion and AVX/FMA vectorisation.
  Used in Google's supremacy experiments.
**Strengths**:
  - State-of-the-art gate fusion (fuses adjacent gates before application).
  - XSim backend uses TPU/GPU.
  - Very low memory overhead per amplitude.
**Weaknesses**:
  - Dense only.
  - No MPS or sparse mode.
  - Tightly coupled to Google's circuit format.
**What AMQT improves**: AMQT is representation-agnostic; future gate fusion can be
  added at the Dispatcher layer without changing the representation API.

**Reference**: https://github.com/quantumlib/qsim

---

### 1.4 CUDA Quantum (NVIDIA)

**Origin**: NVIDIA.
**Approach**: GPU-native statevector simulation using cuStateVec.  Supports
  multi-GPU via cuQuantum.
**Strengths**:
  - Best-in-class GPU utilisation for dense sim.
  - Integrates with NVIDIA hardware ecosystem.
  - Supports hybrid quantum-classical programming.
**Weaknesses**:
  - Dense-only at the statevector level.
  - Tightly coupled to NVIDIA CUDA stack.
**What AMQT improves**: AMQT's memory abstraction layer (planned) would allow
  the same adaptive logic to dispatch to GPU-resident dense or MPS backends
  transparently.

**Reference**: https://github.com/NVIDIA/cuda-quantum

---

## 2. Tensor Network Simulators

### 2.1 ITensor

**Origin**: Flatiron Institute / Miles Stoudenmire.
**Approach**: General tensor network library for physics, with MPS/DMRG as primary
  use case.  Julia and C++ implementations.
**Strengths**:
  - Extremely flexible tensor index system with named, tagged indices.
  - Best-in-class DMRG implementation.
  - Active development, broad physics community.
**Weaknesses**:
  - General-purpose physics library, not a quantum circuit simulator.
  - No automatic circuit-to-MPS mapping; user must design the contraction.
  - Julia ecosystem; C++ API less ergonomic.
**What AMQT improves**: AMQT wraps MPS behind a standard gate-based API so users
  need not understand tensor network contraction ordering.

**Reference**: https://itensor.org

---

### 2.2 TeNPy (Tensor Network Python)

**Origin**: Munich Quantum Valley / Hauschild group.
**Approach**: Python-first tensor network library, primarily for condensed matter.
  MPS, TEBD, DMRG, VUMPS.
**Strengths**:
  - Pure Python / numpy, highly readable code.
  - Excellent for 1D physics simulations.
  - Good documentation and active community.
**Weaknesses**:
  - Not designed for arbitrary quantum circuits.
  - No adaptive representation switching.
  - Performance limited by Python overhead for small bond dimensions.
**What AMQT improves**: Integrates MPS as one representation among several,
  with automatic switching based on runtime diagnostics.

**Reference**: https://github.com/tenpy/tenpy

---

### 2.3 Quimb

**Origin**: Johnnie Gray.
**Approach**: Python tensor network library with automatic contraction ordering
  (cotengra).  Supports arbitrary tensor networks, not just MPS.
**Strengths**:
  - Best-in-class tensor network contraction ordering (cotengra optimiser).
  - Supports random circuit sampling more efficiently than dense for some topologies.
  - Works with JAX/PyTorch backends.
**Weaknesses**:
  - No adaptive switching between representation types.
  - Primarily oriented toward offline contraction of fixed circuits,
    not online (streaming) simulation.
**What AMQT improves**: AMQT targets *online* simulation (gate-by-gate) with
  adaptive representation; Quimb targets *offline* contraction.  The two are
  complementary — AMQT could eventually use Quimb's cotengra as a submodule
  for the tensor-network representation backend.

**Reference**: https://github.com/jcmgray/quimb

---

### 2.4 TensorCircuit

**Origin**: Tencent Quantum Lab.
**Approach**: Differentiable quantum circuit simulator built on JAX/TensorFlow/PyTorch.
  Uses tensor network contraction for simulation.
**Strengths**:
  - Native autodiff support (great for VQE/QAOA optimisation).
  - Multiple backend support (JAX, TF, Torch).
  - Clean Pythonic API.
**Weaknesses**:
  - Fixed backend per session; no adaptive switching.
  - Primarily designed for near-term (NISQ) small circuits.
**What AMQT improves**: AMQT focuses on the representation layer rather than
  the computation backend.  A future AMQT-JAX backend could integrate with
  TensorCircuit-style autodiff.

**Reference**: https://github.com/tencent-quantum-lab/tensorcircuit

---

## 3. Decision Diagram Simulators

### 3.1 QMDD (Quantum Multi-valued Decision Diagrams)

**Origin**: Technical University of Munich / Robert Wille group.
**Approach**: Represent the state (or unitary matrix) as a directed acyclic graph
  where shared sub-trees encode repeated structure.  Used in the MQT simulator suite.
**Strengths**:
  - Can achieve exponential compression for structured circuits (e.g. oracles, QFT).
  - Exact representation.
  - Gate application implemented as DAG manipulation.
**Weaknesses**:
  - Worst case (random circuits) is exponential in size.
  - Complex implementation; cache performance depends on graph structure.
  - Limited tooling for entangled states without much structure.
**What AMQT improves**: AMQT's `representations/decision_diagram/` module
  (planned) will add DD as a fourth representation option, selectable by the
  dispatcher when circuit structure suggests it.

**Reference**: https://github.com/cda-tum/mqt-ddsim

---

### 3.2 Yao.jl

**Origin**: QuEra Computing / Julia community.
**Approach**: Differentiable quantum circuit framework in Julia.  Supports
  multiple backends including dense array, MPS, and symbolic computation.
**Strengths**:
  - Multiple representation support from the start (similar vision to AMQT).
  - Native Julia performance.
  - Strong ecosystem for hybrid classical-quantum algorithms.
**Weaknesses**:
  - Switching policy is manual (user selects backend per circuit).
  - No automatic representation switching based on runtime diagnostics.
  - Julia-specific; not easily embeddable in Python/C++ stacks.
**What AMQT improves**: AMQT automates the representation selection that Yao.jl
  leaves to the user.  This is the central research question: can we make the
  selection *automatic* without loss of correctness?

**Reference**: https://github.com/QuantumBFS/Yao.jl

---

## 4. Stabilizer Simulators

### 4.1 Stim

**Origin**: Google Quantum AI / Craig Gidney.
**Approach**: Clifford circuit simulation via the stabilizer (Heisenberg-picture)
  tableau representation.  Efficiently simulates Clifford gates in O(n^2) time.
**Strengths**:
  - Exponential speedup for pure Clifford circuits.
  - Essential for fault-tolerant circuit simulation.
  - Fastest known stabilizer simulator.
**Weaknesses**:
  - Only works for Clifford gates (H, CNOT, S, Pauli, measurement).
  - Adding even one T gate breaks the stabilizer formalism.
  - No amplitude information (cannot compute expectation values).
**What AMQT improves**: The `representations/stabilizer/` module (planned) will
  integrate Stim-style tableau simulation as a representation, usable until a
  non-Clifford gate is encountered, at which point the dispatcher switches to
  MPS or dense.

**Reference**: https://github.com/quantumlib/Stim

---

## 5. Summary Comparison Table

| Simulator | Dense | Sparse | MPS/TN | DD | Stabilizer | Adaptive |
|---|---|---|---|---|---|---|
| Qiskit Aer | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| QuEST | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| qsim | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| CUDA-Q | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| ITensor | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| TeNPy | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Quimb | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| QMDD | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Yao.jl | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ (manual) |
| Stim | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| **AMQT** | ✓ | ✓ | ✓ | planned | planned | **✓** |

The distinctive claim of AMQT is the rightmost column: **automatic, runtime,
data-driven representation switching** behind a unified API.  No existing
simulator provides this as a first-class design goal.
