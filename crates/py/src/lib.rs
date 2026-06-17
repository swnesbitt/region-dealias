//! PyO3 bindings: a thin numpy-array wrapper over `region_dealias_core`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use region_dealias_core::{sweep_folds, Params};

/// Per-gate fold counts for one sweep.
///
/// vel: float32 (nrays, ngates); mask: bool (nrays, ngates), True = excluded.
/// Returns int32 (nrays, ngates) fold counts; dealiased = vel + folds*2*nyquist.
#[pyfunction]
#[pyo3(signature = (
    vel, mask, nyquist, rays_wrap_around,
    interval_splits = 3, skip_between_rays = 100, skip_along_ray = 100,
    centered = true,
))]
#[allow(clippy::too_many_arguments)]
fn sweep_folds_py<'py>(
    py: Python<'py>,
    vel: PyReadonlyArray2<'py, f32>,
    mask: PyReadonlyArray2<'py, bool>,
    nyquist: f64,
    rays_wrap_around: bool,
    interval_splits: i64,
    skip_between_rays: i64,
    skip_along_ray: i64,
    centered: bool,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let v = vel.as_array();
    let m = mask.as_array();
    if v.shape() != m.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vel and mask must have the same shape",
        ));
    }
    let nrays = v.shape()[0];
    let ngates = v.shape()[1];

    // C-order contiguous slices (caller passes np.ascontiguousarray)
    let vslice = v
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("vel must be C-contiguous"))?;
    let mslice = m
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("mask must be C-contiguous"))?;

    let params = Params {
        interval_splits,
        skip_between_rays,
        skip_along_ray,
        centered,
    };

    let folds = py.allow_threads(|| {
        sweep_folds(vslice, nrays, ngates, mslice, nyquist, rays_wrap_around, params)
    });

    let arr = Array2::from_shape_vec((nrays, ngates), folds)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

#[pymodule]
fn _region_dealias(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sweep_folds_py, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
