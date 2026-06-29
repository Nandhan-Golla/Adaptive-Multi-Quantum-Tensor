# Mathematical Analysis of Quantum State Representations

## 1. State Space Definition

An n-qubit pure quantum state lives in the Hilbert space:

```
H = C^2 ⊗ C^2 ⊗ ... ⊗ C^2   (n times)
  = C^(2^n)
```

A state |ψ⟩ ∈ H can be written in the computational basis:

```
|ψ⟩ = Σ_{x ∈ {0,1}^n} α_x |x⟩
```

subject to the normalisation constraint:

```
Σ_{x} |α_x|^2 = 1
```

The 2^n complex amplitudes {α_x} constitute the *statevector*.  All three
representations below must store or implicitly encode these amplitudes.

---

## 2. Dense Statevector Representation

### Definition

Store all 2^n complex amplitudes explicitly as a rank-1 array:

```
sv ∈ C^(2^n),   ||sv||_2 = 1
```

Qubit ordering convention: qubit 0 is the most-significant (leftmost) bit of
the integer index.  Thus:

```
|q_0 q_1 ... q_{n-1}⟩  ↔  sv[q_0 * 2^{n-1} + q_1 * 2^{n-2} + ... + q_{n-1}]
```

### Gate Application

A single-qubit gate G on qubit k is implemented as:

```
I ⊗ ... ⊗ G ⊗ ... ⊗ I   (G at position k)
```

In tensor notation: reshape sv to shape [2]×n, contract index k with G,
reshape back.  This costs O(2^n) operations.

For a k-qubit gate U (shape 2^k × 2^k) on qubits (q_0, ..., q_{k-1}):
reshape to tensor, contract k indices, transpose back.  Cost: O(2^k · 2^n).

### Complexity

| Quantity | Value |
|---|---|
| Memory | 2^n × 16 bytes (complex128) |
| Single-qubit gate | O(2^n) FLOPs |
| Two-qubit gate | O(4 · 2^n) FLOPs |
| k-qubit gate | O(2^k · 2^n) FLOPs |

### When Optimal

- Small n (n ≤ 20 on typical hardware with 16 GB RAM).
- Highly entangled states where other representations inflate.
- When exact, reproducible results are required (zero approximation error).
- Clifford circuits on n ≤ ~25 qubits (GPU dense sim is fastest).

---

## 3. Sparse Representation

### Definition

Store only the non-zero amplitudes (|α_x| > ε) as a dictionary:

```
D = { x → α_x  :  |α_x| > ε }
```

Let nnz = |D| be the number of non-zero entries.

### Invariant

At all times the state is represented exactly (up to the threshold ε) as:

```
|ψ⟩ = Σ_{x ∈ D} α_x |x⟩
```

The threshold ε = 1e-10 is chosen so that dropped amplitudes contribute at
most ε^2 · nnz ≈ 0 to the norm squared.

### Gate Application

For a k-qubit gate U on target qubits T = (t_0, ..., t_{k-1}):

For each basis state x in D with amplitude α_x:
1. Extract the sub-index s = bits of x at positions T.
2. Apply U to s: new amplitudes = U[:, s] · α_x.
3. Accumulate contributions to a new dict.

Cost: O(2^k · nnz) per gate.

This is dramatically cheaper than dense when nnz ≪ 2^n, but degrades to
O(2^{k+n}) in the worst case (fully dense state).

### Complexity

| Quantity | Value |
|---|---|
| Memory | nnz × (8 + 16) bytes = nnz × 24 bytes |
| k-qubit gate | O(2^k · nnz) FLOPs |
| Best case (nnz=1) | O(2^k) FLOPs |
| Worst case (nnz=2^n) | O(2^{k+n}) FLOPs |

### Sparsity

Define sparsity ρ = 1 - nnz / 2^n.  The sparse representation is
advantageous when ρ > 0.95, i.e. more than 95% of amplitudes are zero.

### When Optimal

- Computational basis states and their small superpositions.
- Early circuit stages before entanglement builds up.
- Circuits with many classically controlled operations.
- Stabilizer circuits on a limited number of non-Clifford qubits (T-count limited states).

---

## 4. Matrix Product State (MPS) Representation

### Definition

An MPS with open boundary conditions represents |ψ⟩ as:

```
|ψ⟩ = Σ_{s_0,...,s_{n-1}} A^{s_0}[0] A^{s_1}[1] ··· A^{s_{n-1}}[n-1] |s_0...s_{n-1}⟩
```

where each tensor A[i] has shape (χ_{i-1}, 2, χ_i) with χ_0 = χ_n = 1.
The χ_i are the *bond dimensions* (virtual indices).

In left-canonical gauge: A[i]^† A[i] = I for all i < n-1.

### Bond Dimension and Entanglement

By the Schmidt decomposition across the bipartition [0..i | i+1..n-1]:

```
|ψ⟩ = Σ_α λ_α |α⟩_L ⊗ |α⟩_R
```

The Schmidt rank is at most χ_i.  For a general state, χ_i can be as large
as min(2^i, 2^{n-i}).  The MPS representation is exact only when χ_max ≥ max_i χ_i.

When we impose a maximum bond dimension χ_max and truncate smaller singular
values, we introduce an approximation error:

```
|| |ψ⟩ - |ψ_approx⟩ ||^2 ≤ Σ_{discarded} σ_j^2
```

### Gate Application

**Single-qubit gate G on site i:**

```
A'^{s'}[i] = Σ_s G[s', s] A^{s}[i]
```

Cost: O(χ^2) (local contraction, no SVD needed).

**Two-qubit gate U on adjacent sites (i, i+1):**

1. Contract: Θ_{α,s_i,s_{i+1},β} = Σ_γ A[i]_{α,s_i,γ} · A[i+1]_{γ,s_{i+1},β}
   Cost: O(χ^3)

2. Apply gate: Θ'_{α,s'_i,s'_{i+1},β} = Σ_{s_i,s_{i+1}} U[s'_i s'_{i+1}, s_i s_{i+1}] · Θ_{α,s_i,s_{i+1},β}
   Cost: O(4 · χ^2)

3. Reshape Θ' to (2χ, 2χ) and SVD: U S V^†.
   Truncate to χ_max singular values.
   Cost: O(χ^3) dominated by SVD.

4. A'[i] = U_trunc.reshape(χ_left, 2, χ_new)
   A'[i+1] = (diag(S) V†_trunc).reshape(χ_new, 2, χ_right)

**Non-adjacent two-qubit gate (q_0, q_1) with |q_0 - q_1| > 1:**

Route via SWAP gates to make qubits adjacent, apply gate, un-SWAP.
Introduces SWAP overhead of O(|q_0 - q_1| · χ^3) but is exact up to
truncation.

### Complexity

| Quantity | Value |
|---|---|
| Memory | O(n · χ^2 · 2) complex numbers = O(32 n χ^2) bytes |
| Single-qubit gate | O(χ^2) |
| Two-qubit gate (adjacent) | O(χ^3) |
| k-qubit gate (general) | O(2^k · χ^3) |
| Statevector reconstruction | O(n · χ^2 · 2^n) |

### Approximation Error

Truncating to χ_max singular values introduces cumulative error:

```
ε_total ≤ Σ_{gates} || σ_discarded ||_F
```

The AMQT implementation tracks `approximation_error` as this cumulative sum.

### When Optimal

- Low-entanglement circuits: product states, shallow circuits, 1D geometries.
- Simulating longer circuits on 20–100+ qubits when entanglement is bounded.
- DMRG-like algorithms where χ stays small throughout.
- Variational quantum eigensolvers on chain-topology Hamiltonians.

The MPS is sub-optimal (χ → exponential) for:
- Deeply entangled states (e.g. random quantum circuits beyond depth ~log n).
- Non-local gates on all-to-all topologies without structure.

---

## 5. Comparison Table

| Property | Dense | Sparse | MPS |
|---|---|---|---|
| Exact? | Yes | Yes (above threshold) | Approx (truncation) |
| Memory | O(2^n) | O(nnz) | O(n χ^2) |
| Gate cost | O(2^n) | O(2^k · nnz) | O(χ^3) |
| Best for | Small n, full entanglement | Many zero amplitudes | Low entanglement, large n |
| Worst case | Always O(2^n) | Entangled state: O(2^{k+n}) | χ = 2^{n/2}: as bad as dense |
| Approximation | None | Threshold pruning | SVD truncation |
| Error tracking | N/A | N/A (exact above threshold) | Cumulative σ² sum |

---

## 6. Representation Equivalence

All three representations encode the same mathematical object — an element
of the unit sphere in C^(2^n).  Conversion is always possible:

- Dense → Sparse: extract non-zero entries, O(2^n).
- Sparse → Dense: place entries into full array, O(nnz).
- Dense → MPS: sequential SVD decomposition, O(n · 2^n · χ) for χ = χ_max.
- MPS → Dense: tensor contraction chain, O(n · χ^2 · 2^n).
- Sparse → MPS: Sparse → Dense → MPS.
- MPS → Sparse: MPS → Dense → Sparse.

The AMQT dispatcher uses `to_dense()` as the universal intermediate.

---

## 7. Information-Theoretic Bounds

### Compression Lower Bound

For an arbitrary n-qubit state, any lossless representation requires at
least 2^{n+1} - 2 real numbers (up to global phase).  No representation
can do better in the worst case.  This is why:

1. Dense representation is optimal in the worst case.
2. MPS and sparse are optimal only for *structured* classes of states.
3. Any claimed exponential compression must specify its assumptions.

### Entanglement and MPS Bond Dimension

The von-Neumann entropy of the reduced state on site i:

```
S_i = -Tr(ρ_i log ρ_i)
```

is related to the Schmidt rank:

```
χ_i ≥ 2^{S_i / log 2}   (lower bound from entropy)
```

For area-law states (S_i = O(1)), χ_i = O(1) and MPS achieves exponential
compression.  For volume-law states (random circuits), S_i = Ω(min(i, n-i))
and χ_i must be exponential.
