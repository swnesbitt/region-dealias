//! Interval-split computation and region finding (`_find_sweep_interval_splits`
//! and `_find_regions` in Py-ART), faithful to the float dtypes pyart uses
//! (velocity float32 promoted to float64 in comparisons; limits float64).

use crate::label::label4;
use crate::util::linspace;

/// `_find_sweep_interval_splits`: velocity limits covering the Nyquist
/// co-interval (plus extra bins if data exceeds ±Nyquist). `mask` is the
/// excluded-gate filter (True = excluded), matching pyart's `gfilter`.
pub fn find_sweep_interval_splits(
    nyquist: f64,
    interval_splits: i64,
    vel: &[f32],
    mask: &[bool],
) -> Vec<f64> {
    let interval = (2.0 * nyquist) / (interval_splits as f64);
    let mut add_start: i64 = 0;
    let mut add_end: i64 = 0;

    let mut any = false;
    let mut maxv = f32::NEG_INFINITY;
    let mut minv = f32::INFINITY;
    for i in 0..vel.len() {
        if !mask[i] {
            any = true;
            let v = vel[i];
            if v > maxv {
                maxv = v;
            }
            if v < minv {
                minv = v;
            }
        }
    }
    if any {
        let max_vel = maxv as f64;
        let min_vel = minv as f64;
        if max_vel > nyquist || min_vel < -nyquist {
            add_start = ((max_vel - nyquist) / interval).ceil() as i64;
            add_end = (-(min_vel + nyquist) / interval).ceil() as i64;
        }
    }
    let start = -nyquist - (add_start as f64) * interval;
    let end = nyquist + (add_end as f64) * interval;
    let num = interval_splits + 1 + add_start + add_end;
    linspace(start, end, num.max(0) as usize)
}

/// `_find_regions`: for each consecutive pair of limits, label connected
/// regions of velocity within `[lmin, lmax)` (excluding filtered gates) and
/// accumulate into a single labeled field. Returns `(labels, nfeatures)`.
pub fn find_regions(
    vel: &[f32],
    mask: &[bool],
    limits: &[f64],
    nx: usize,
    ny: usize,
) -> (Vec<i32>, i32) {
    let n = nx * ny;
    let mut label = vec![0i32; n];
    let mut nfeatures: i32 = 0;
    let mut inp = vec![false; n];
    for w in limits.windows(2) {
        let (lmin, lmax) = (w[0], w[1]);
        for i in 0..n {
            inp[i] = if mask[i] {
                false
            } else {
                let v = vel[i] as f64; // float32 < float64 -> compare in float64
                lmin <= v && v < lmax
            };
        }
        let (limit_label, limit_nfeatures) = label4(&inp, nx, ny);
        if limit_nfeatures > 0 {
            for i in 0..n {
                if limit_label[i] != 0 {
                    label[i] = limit_label[i] + nfeatures;
                }
            }
            nfeatures += limit_nfeatures;
        }
    }
    (label, nfeatures)
}
