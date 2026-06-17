//! Numpy-faithful numeric helpers.

/// `np.round` / Python `round` semantics: round half to even (banker's),
/// then to integer. IEEE `round_ties_even` matches numpy's default rint.
#[inline]
pub fn rint_i64(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Replica of `numpy.linspace(start, stop, num, endpoint=True)`.
///
/// numpy computes `y = arange(num) * step + start` with `step = (stop-start)/(num-1)`
/// then forces `y[-1] = stop`. We reproduce that exactly (including the final
/// assignment, which avoids accumulated rounding at the endpoint).
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / ((num - 1) as f64);
    let mut v = Vec::with_capacity(num);
    for i in 0..num {
        v.push((i as f64) * step + start);
    }
    v[num - 1] = stop;
    v
}
