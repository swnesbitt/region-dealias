//! Region-based Doppler velocity dealiasing — a faithful Rust port of the core
//! (non-reference-anchored) path of Py-ART's `dealias_region_based`.
//!
//! The public entry point [`sweep_folds`] operates on a single sweep and
//! returns, for every gate, the integer number of Nyquist-interval folds to
//! add. Dealiased velocity is then `vel + folds * (2 * nyquist)`.

mod edges;
mod label;
mod network;
mod regions;
mod util;

use edges::edge_sum_and_count;
use network::network_reduce;
use regions::{find_regions, find_sweep_interval_splits};

/// Parameters mirroring pyart's defaults for the app's call path.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub interval_splits: i64,
    pub skip_between_rays: i64, // max_gap_x (rays / axis 0)
    pub skip_along_ray: i64,    // max_gap_y (gates / axis 1)
    pub centered: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            interval_splits: 3,
            skip_between_rays: 100,
            skip_along_ray: 100,
            centered: true,
        }
    }
}

/// Compute per-gate fold counts for one sweep.
///
/// * `vel` — velocity, C-order, shape `nrays × ngates` (axis 0 = rays/azimuth,
///   axis 1 = gates/range), float32 (as pyart uses).
/// * `mask` — excluded-gate filter, True = excluded (masked or invalid).
/// * `nyquist` — sweep Nyquist velocity.
/// * `rays_wrap_around` — true for PPI (axis 0 wraps).
///
/// Returns a `nrays × ngates` (C-order) array of fold counts (`i32`). Gates that
/// were not dealiased (masked, single-region sweeps, etc.) get 0.
pub fn sweep_folds(
    vel: &[f32],
    nrays: usize,
    ngates: usize,
    mask: &[bool],
    nyquist: f64,
    rays_wrap_around: bool,
    params: Params,
) -> Vec<i32> {
    let n = nrays * ngates;
    let mut folds = vec![0i32; n];
    if n == 0 {
        return folds;
    }

    let nyquist_interval = nyquist * 2.0;

    let limits =
        find_sweep_interval_splits(nyquist, params.interval_splits, vel, mask);
    if limits.len() < 2 {
        return folds;
    }

    let (labels, nfeatures) = find_regions(vel, mask, &limits, nrays, ngates);
    if nfeatures < 2 {
        return folds;
    }

    let mut region_sizes = vec![0i32; nfeatures as usize];
    for &l in &labels {
        if l != 0 {
            region_sizes[(l - 1) as usize] += 1;
        }
    }

    let edges = edge_sum_and_count(
        &labels,
        vel,
        nrays,
        ngates,
        rays_wrap_around,
        params.skip_between_rays,
        params.skip_along_ray,
    );
    if edges.i.is_empty() {
        return folds;
    }

    let unwrap =
        network_reduce(&edges, &region_sizes, nfeatures, nyquist_interval, params.centered);

    for i in 0..n {
        folds[i] = unwrap[labels[i] as usize];
    }
    folds
}
