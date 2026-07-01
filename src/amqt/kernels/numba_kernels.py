
from __future__ import annotations

import numpy as np

try:
    import numba

    @numba.njit(cache=True)
    def apply_gate_1q(sv, gate, qubit, n):
        stride = 1 << (n - 1 - qubit)
        dim = 1 << n
        g00 = gate[0, 0]
        g01 = gate[0, 1]
        g10 = gate[1, 0]
        g11 = gate[1, 1]
        i = 0
        while i < dim:
            for j in range(i, i + stride):
                a = sv[j]
                b = sv[j + stride]
                sv[j]          = g00 * a + g01 * b
                sv[j + stride] = g10 * a + g11 * b
            i += 2 * stride

    @numba.njit(cache=True)
    def apply_gate_2q(sv, gate, q0, q1, n):
        s0 = 1 << (n - 1 - q0)
        s1 = 1 << (n - 1 - q1)
        dim = 1 << n
        for i in range(dim):
            b0 = (i >> (n - 1 - q0)) & 1
            b1 = (i >> (n - 1 - q1)) & 1
            if b0 == 0 and b1 == 0:
                i00 = i
                i01 = i | s1
                i10 = i | s0
                i11 = i | s0 | s1
                v0 = sv[i00]
                v1 = sv[i01]
                v2 = sv[i10]
                v3 = sv[i11]
                sv[i00] = gate[0, 0]*v0 + gate[0, 1]*v1 + gate[0, 2]*v2 + gate[0, 3]*v3
                sv[i01] = gate[1, 0]*v0 + gate[1, 1]*v1 + gate[1, 2]*v2 + gate[1, 3]*v3
                sv[i10] = gate[2, 0]*v0 + gate[2, 1]*v1 + gate[2, 2]*v2 + gate[2, 3]*v3
                sv[i11] = gate[3, 0]*v0 + gate[3, 1]*v1 + gate[3, 2]*v2 + gate[3, 3]*v3

    NUMBA_AVAILABLE = True

except ImportError:
    NUMBA_AVAILABLE = False

    def apply_gate_1q(sv, gate, qubit, n):  # type: ignore[misc]
        raise RuntimeError("numba is not installed")

    def apply_gate_2q(sv, gate, q0, q1, n):  # type: ignore[misc]
        raise RuntimeError("numba is not installed")
