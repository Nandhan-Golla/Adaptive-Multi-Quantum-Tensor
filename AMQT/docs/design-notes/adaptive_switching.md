# Adaptive Representation Switching: Design Notes

## 1. Motivation

No single quantum state representation is optimal for all circuits.
Dense statevectors are exact but exponential in memory.
Sparse dicts are efficient early in a circuit but explode after entangling gates.
MPS is efficient for low-entanglement states but expensive when χ grows.

The core AMQT insight is: **the optimal representation changes during execution**.
A circuit that starts as a product state, builds entanglement through a GHZ
preparation, then partially disentangles via measurements, should ideally flow:

```
Sparse  →  MPS  →  (Sparse or Dense after measurement)
```

The dispatcher implements this as an automatic, transparent policy.

---

## 2. Metadata Signals Used for Switching

The `StateMetadata` dataclass tracks the signals that drive switching decisions:

### 2.1 Sparsity

```
sparsity = (number of near-zero amplitudes) / 2^n
```

- Range: [0, 1].  0 = fully dense; 1 = all zero (impossible for normalised state).
- Computed after every gate from the current representation.
- For dense rep: count amplitudes with |α| < 1e-10.
- For sparse rep: directly nnz / 2^n.
- For MPS: reconstruct statevector (expensive for large n; only done when metadata
  is explicitly requested).

**Cost**: For dense, O(2^n).  For MPS at large n we avoid recomputing SV by
estimating from metadata — this is a future optimisation (not yet implemented).

### 2.2 Entropy Estimate

We estimate the von-Neumann entropy of the bipartition (qubits 0..n//2-1 | n//2..n-1):

```
S ≈ -Σ_i λ_i log_2 λ_i
```

where λ_i are squared singular values of the (2^{n/2} × 2^{n/2}) reshape of sv.
This is normalised to [0, 1] by dividing by min(n//2, n - n//2).

- S_norm = 0: product state (no entanglement).
- S_norm = 1: maximally entangled state.

**Cost**: O(2^{n/2} · min(2^{n/2}, 2^{n/2})^2) for the SVD = O(2^{3n/4}) in
the worst case, which is cheaper than full gate simulation for large n but still
significant.  In practice we only call `get_metadata()` after each gate, and for
large n the user should disable auto_switch for performance benchmarking.

### 2.3 Bond Dimension (MPS only)

The max bond dimension χ currently in use.  When χ approaches χ_max, the
approximation error is growing rapidly — a signal that MPS is no longer efficient
and the circuit may need a different approach (e.g. full contraction via dense).

---

## 3. Switching Policy

The `RepresentationSwitcher` implements a simple threshold policy.  All thresholds
are class-level constants (easily overridable by subclassing or patching):

```
SPARSE_THRESHOLD = 0.95   # sparsity above which Dense -> Sparse
DENSE_THRESHOLD  = 0.70   # sparsity below which Sparse -> Dense
MPS_QUBIT_MIN    = 12     # minimum n to consider MPS
MPS_ENTROPY_MIN  = 0.50   # entropy above which Dense/Sparse -> MPS
MPS_EXIT_ENTROPY = 0.10   # entropy below which MPS exits (back to Dense/Sparse)
```

### Decision Table

| Current Rep | Condition | Action |
|---|---|---|
| Dense | sparsity > 0.95 | → Sparse |
| Sparse | sparsity < 0.70 | → Dense |
| Dense or Sparse | n ≥ 12 AND entropy > 0.50 | → MPS |
| MPS | entropy < 0.10 | → Sparse (if sparse) or Dense |
| Any | None of the above | Stay |

The `RepresentationSwitcher.switch_if_beneficial` method evaluates these rules in
order and returns a new representation if a switch is warranted, otherwise returns
the same object (identity).

---

## 4. Conversion Protocol

All switches pass through `to_dense()` as a safe universal intermediate:

```
Sparse → Dense → MPS
MPS    → Dense → Sparse
Dense  → Sparse  (direct, O(2^n) scan)
Dense  → MPS     (direct, O(n·2^n) SVD chain)
```

This is intentionally conservative.  Future optimisations could implement:
- Sparse → MPS directly (sweep-SVD on non-zero entries).
- MPS → Sparse directly (contract and threshold simultaneously).

---

## 5. Mathematical Justification of Thresholds

### 5.1 Dense → Sparse (sparsity > 0.95)

Memory ratio: sparse / dense = (24 · nnz) / (16 · 2^n)

For the switch to save memory:
```
24 · nnz < 16 · 2^n
nnz / 2^n < 2/3
density = 1 - sparsity < 2/3
sparsity > 1/3
```

So any sparsity above ~33% already saves memory.  We use 95% as a conservative
threshold to avoid thrashing (frequent switches) when the state is moderately sparse.
The gate-application speedup is another factor: sparse is faster when nnz ≪ 2^n,
which happens reliably only at very high sparsity.

### 5.2 Sparse → Dense (sparsity < 0.70)

After entangling gates, nnz can grow rapidly.  When it exceeds ~30% of 2^n,
the sparse overhead (dict lookups, Python objects per entry) starts to
exceed the cost of dense array operations.  We revert to dense at 70% density.

There is a hysteresis gap (70% threshold for exit, 95% for entry) to prevent
oscillation: a state at 75% sparsity would neither trigger Dense→Sparse nor
Sparse→Dense, keeping the current representation stable.

### 5.3 Dense/Sparse → MPS (n ≥ 12, entropy > 0.50)

MPS is only advantageous when:
1. The system is large enough that exponential memory is a bottleneck (n ≥ 12,
   where dense requires at least 64KB).
2. The state is entangled enough that there is entanglement *structure* to exploit
   (entropy > 0.50).

If the state is not yet entangled (entropy ≈ 0), MPS provides no benefit over
Sparse and incurs higher overhead from SVD in gate application.

The n ≥ 12 threshold is intentionally low to catch medium-scale circuits early.
For production use this should be tuned based on available memory and χ_max.

### 5.4 MPS → Dense/Sparse (entropy < 0.10)

If entanglement drops dramatically (e.g., after measurements or classical
conditioning), MPS loses its advantage.  A low-entanglement MPS with χ = 1 or 2
is essentially the same as a product state, which sparse or dense handles more
efficiently due to lower overhead per operation.

---

## 6. Switching Overhead

A switch costs:

| Transition | Cost |
|---|---|
| Dense → Sparse | O(2^n) — scan and threshold |
| Sparse → Dense | O(nnz + 2^n) — place into array |
| Dense → MPS | O(n · χ^3 · 2^n) — sequential SVD |
| MPS → Dense | O(n · χ^2 · 2^n) — tensor contraction |

For large n, switching to or from MPS is expensive.  The policy only does so
when the long-term gain (smaller memory, faster per-gate cost) outweighs the
one-time switching cost.  A future "switching cost estimator" could track gate
counts and projected future cost before deciding.

---

## 7. Future Policy Improvements

Several directions could replace or augment the current heuristic:

### 7.1 Learned Policy

Train a small neural network (or decision tree) to predict the optimal
representation given the metadata history.  Input: last k metadata snapshots.
Output: representation to use.  Training signal: measured wall-clock time.

### 7.2 Predictive Switching

Instead of reacting to current metadata, predict future metadata from the
upcoming circuit (if known ahead of time).  This requires a circuit IR.

### 7.3 Hysteresis Memory

Track how many consecutive gates have triggered the switch condition before
actually switching.  This prevents false positives from momentary sparsity spikes.

### 7.4 Cost-Aware Switching

Maintain an explicit cost model:
```
cost_dense_gate = C_dense * 2^n
cost_sparse_gate = C_sparse * nnz
cost_mps_gate = C_mps * chi^3
switch_cost_dense_to_mps = K * n * chi^3 * 2^n
```

Switch only when projected future cost savings exceed amortised switching cost:
```
if (expected_gates * (cost_current - cost_target)) > switch_cost:
    switch()
```

### 7.5 Block-Level Switching

Instead of switching the entire state, allow different *sub-blocks* (qubit
subsets) to use different representations.  This would require a more
sophisticated chunking architecture (see CLAUDE.md: Dynamic Chunking).

---

## 8. Design Invariants

The switching system must maintain:

1. **State equivalence**: The statevector before and after a switch must be
   identical (up to approximation error for MPS truncation).
2. **Monotone norm**: ||state|| = 1 is preserved by all representations.
3. **No hidden state**: The representation object contains all state; no
   global mutable state outside the Dispatcher.
4. **Reversibility**: Any sequence of switches must produce the same final
   state as running the same circuit without any switches (within the
   accumulated approximation error bound).
