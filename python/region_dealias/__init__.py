"""Fast region-based Doppler velocity dealiasing.

A Rust port of the core (non-reference-anchored) path of Py-ART's
``dealias_region_based``. Two layers:

* :func:`sweep_folds` — low-level, array in / fold-count out (one sweep).
* :func:`dealias_region_based` — drop-in mirror of pyart's signature operating
  on a pyart ``Radar`` object (returns the same field dictionary).
"""

from __future__ import annotations

import numpy as np

from ._region_dealias import sweep_folds_py as _sweep_folds  # type: ignore
from ._region_dealias import __version__  # noqa: F401

__all__ = ["sweep_folds", "dealias_region_based", "__version__"]

_DEFAULT_FILL = -9999.0


def sweep_folds(
    vel,
    mask,
    nyquist,
    rays_wrap_around,
    interval_splits=3,
    skip_between_rays=100,
    skip_along_ray=100,
    centered=True,
):
    """Per-gate integer fold counts for a single sweep.

    Dealiased velocity is ``vel + folds * (2 * nyquist)``.

    Parameters
    ----------
    vel : (nrays, ngates) array, float32-coercible
        Doppler velocity for one sweep (axis 0 = rays, axis 1 = gates).
    mask : (nrays, ngates) bool array
        True where the gate is excluded (masked or invalid).
    nyquist : float
    rays_wrap_around : bool
        True for PPI scans (azimuth axis wraps).
    """
    vel = np.ascontiguousarray(vel, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=bool)
    return _sweep_folds(
        vel,
        mask,
        float(nyquist),
        bool(rays_wrap_around),
        int(interval_splits),
        int(skip_between_rays),
        int(skip_along_ray),
        bool(centered),
    )


def dealias_region_based(
    radar,
    ref_vel_field=None,
    interval_splits=3,
    interval_limits=None,
    skip_between_rays=100,
    skip_along_ray=100,
    centered=True,
    nyquist_vel=None,
    check_nyquist_uniform=True,
    gatefilter=False,
    rays_wrap_around=None,
    keep_original=False,
    set_limits=True,
    vel_field=None,
    corr_vel_field=None,
    **kwargs,
):
    """Drop-in replacement for ``pyart.correct.dealias_region_based``.

    Supports the parameters used by the NEXRAD browser. ``ref_vel_field`` (the
    sounding-anchored branch) and explicit ``gatefilter``/``interval_limits``
    are **not** implemented here — if any are requested this raises
    ``NotImplementedError`` so callers can fall back to pyart.
    """
    if ref_vel_field is not None:
        raise NotImplementedError("ref_vel_field is not supported; use pyart")
    if interval_limits is not None:
        raise NotImplementedError("interval_limits is not supported; use pyart")
    if gatefilter not in (False, None):
        raise NotImplementedError("custom gatefilter is not supported; use pyart")

    if vel_field is None:
        vel_field = "velocity"
    if corr_vel_field is None:
        corr_vel_field = "corrected_velocity"

    vfield = radar.fields[vel_field]
    vdata = vfield["data"]
    und = np.ma.getdata(vdata).astype(np.float32, copy=True)
    gfilter = np.ma.getmaskarray(vdata) | ~np.isfinite(und)

    # per-sweep Nyquist
    if nyquist_vel is None:
        nyq = [
            float(radar.get_nyquist_vel(s, check_nyquist_uniform))
            for s in range(radar.nsweeps)
        ]
    elif np.isscalar(nyquist_vel):
        nyq = [float(nyquist_vel)] * radar.nsweeps
    else:
        nyq = [float(x) for x in nyquist_vel]

    if rays_wrap_around is None:
        rays_wrap_around = radar.scan_type == "ppi"

    out = und.copy()
    for s, sl in enumerate(radar.iter_slice()):
        svel = np.ascontiguousarray(und[sl], dtype=np.float32)
        sfilter = np.ascontiguousarray(gfilter[sl], dtype=bool)
        folds = _sweep_folds(
            svel,
            sfilter,
            nyq[s],
            bool(rays_wrap_around),
            int(interval_splits),
            int(skip_between_rays),
            int(skip_along_ray),
            bool(centered),
        )
        nyq_interval = 2.0 * nyq[s]
        # match numpy in-place float32 += float64 (compute in f64, store f32)
        corrected = svel.astype(np.float64) + folds.astype(np.float64) * nyq_interval
        out[sl] = corrected.astype(np.float32)

    fill_value = float(vfield.get("_FillValue", _DEFAULT_FILL))
    data = np.ma.array(out, mask=gfilter, fill_value=fill_value)
    if keep_original:
        data[gfilter] = und[gfilter]

    corr = {
        "data": data,
        "_FillValue": fill_value,
        "units": "meters_per_second",
        "standard_name": "corrected_radial_velocity",
        "long_name": "Corrected mean Doppler velocity",
    }
    if set_limits:
        _set_limits(data, nyq, corr)
    return corr


def _set_limits(data, nyquist_vel, dic):
    max_abs_vel = np.ma.max(np.ma.abs(data))
    if max_abs_vel is np.ma.masked:
        return
    max_nyq_vel = np.max(nyquist_vel)
    max_nyq_int = 2.0 * max_nyq_vel
    added = np.ceil((max_abs_vel - max_nyq_vel) / max_nyq_int)
    max_valid = max_nyq_vel + added * max_nyq_int
    dic["valid_min"] = float(-max_valid)
    dic["valid_max"] = float(max_valid)
