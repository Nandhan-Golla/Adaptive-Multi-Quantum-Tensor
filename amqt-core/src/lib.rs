use numpy::{Complex64, PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::prelude::*;

#[pyfunction]
fn apply_gate_1q(
    _py: Python<'_>,
    sv: &Bound<'_, PyArray1<Complex64>>,
    gate: PyReadonlyArray2<'_, Complex64>,
    qubit: usize,
    n: usize,
) -> PyResult<()> {
    let sv_data = unsafe { sv.as_slice_mut()? };
    let g = gate.as_slice()?;
    let g00 = g[0];
    let g01 = g[1];
    let g10 = g[2];
    let g11 = g[3];

    let stride = 1usize << (n - 1 - qubit);
    let dim = 1usize << n;

    if dim >= 16384 {

        let log_stride = stride.trailing_zeros() as usize;
        (0..(dim / 2)).into_par_iter().for_each(|flat| {
            let low = flat & (stride - 1);
            let high = flat >> log_stride;
            let i0 = (high << (log_stride + 1)) | low;
            let i1 = i0 | stride;
            unsafe {
                let a = *sv_data.get_unchecked(i0);
                let b = *sv_data.get_unchecked(i1);
                let p = sv_data.as_ptr().cast_mut();
                *p.add(i0) = g00 * a + g01 * b;
                *p.add(i1) = g10 * a + g11 * b;
            }
        });
    } else {
        let mut i = 0usize;
        while i < dim {
            for j in i..(i + stride) {
                let a = sv_data[j];
                let b = sv_data[j + stride];
                sv_data[j] = g00 * a + g01 * b;
                sv_data[j + stride] = g10 * a + g11 * b;
            }
            i += 2 * stride;
        }
    }
    Ok(())
}

#[pyfunction]
fn apply_gate_2q(
    _py: Python<'_>,
    sv: &Bound<'_, PyArray1<Complex64>>,
    gate: PyReadonlyArray2<'_, Complex64>,
    q0: usize,
    q1: usize,
    n: usize,
) -> PyResult<()> {
    let sv_data = unsafe { sv.as_slice_mut()? };
    let g = gate.as_slice()?;

    let s0 = 1usize << (n - 1 - q0);
    let s1 = 1usize << (n - 1 - q1);
    let dim = 1usize << n;
    let bit0 = n - 1 - q0;
    let bit1 = n - 1 - q1;

    for i in 0..dim {
        if ((i >> bit0) & 1) == 0 && ((i >> bit1) & 1) == 0 {
            let i00 = i;
            let i01 = i | s1;
            let i10 = i | s0;
            let i11 = i | s0 | s1;

            let v0 = sv_data[i00];
            let v1 = sv_data[i01];
            let v2 = sv_data[i10];
            let v3 = sv_data[i11];

            sv_data[i00] = g[0]*v0  + g[1]*v1  + g[2]*v2  + g[3]*v3;
            sv_data[i01] = g[4]*v0  + g[5]*v1  + g[6]*v2  + g[7]*v3;
            sv_data[i10] = g[8]*v0  + g[9]*v1  + g[10]*v2 + g[11]*v3;
            sv_data[i11] = g[12]*v0 + g[13]*v1 + g[14]*v2 + g[15]*v3;
        }
    }
    Ok(())
}


#[pymodule]
fn amqt_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_gate_1q, m)?)?;
    m.add_function(wrap_pyfunction!(apply_gate_2q, m)?)?;
    Ok(())
}
