"""Parity against the *installed* Py-ART package (the wrapper + core, end to end).

Builds synthetic pyart Radar objects, runs both pyart's and our
``dealias_region_based``, and asserts the corrected velocity fields are exactly
equal at unmasked gates. Skipped automatically if pyart isn't installed.
"""

import numpy as np
import pytest

pyart = pytest.importorskip("pyart")

import region_dealias  # noqa: E402


def _make_radar(seed, nrays=180, ngates=120, nyquist=25.0):
    rng = np.random.default_rng(seed)
    radar = pyart.testing.make_empty_ppi_radar(ngates, nrays, 1)
    radar.range["data"] = np.arange(ngates, dtype="float32") * 250.0

    az = np.deg2rad(radar.azimuth["data"])[:, None]
    rg = np.linspace(0, np.pi, ngates)[None, :]
    true = np.zeros((nrays, ngates))
    for _ in range(4):
        true += rng.uniform(-1, 1) * np.sin(rng.integers(1, 5) * az + rng.uniform(0, 6)) \
            * np.cos(rng.integers(1, 5) * rg)
    true *= rng.uniform(1.5, 3.0) * nyquist
    obs = (((true + nyquist) % (2 * nyquist)) - nyquist).astype("float32")

    mask = rng.random((nrays, ngates)) < 0.2
    vel = np.ma.array(obs, mask=mask)
    radar.add_field("velocity", {"data": vel, "_FillValue": -9999.0})

    nyq = np.full(nrays, nyquist, dtype="float32")
    radar.instrument_parameters = {
        "nyquist_velocity": {"data": nyq},
    }
    return radar


@pytest.mark.parametrize("seed", range(40))
def test_wrapper_matches_pyart(seed):
    radar = _make_radar(seed)
    a = pyart.correct.dealias_region_based(
        radar, vel_field="velocity", keep_original=False
    )["data"]
    b = region_dealias.dealias_region_based(
        radar, vel_field="velocity", keep_original=False
    )["data"]

    ma = np.ma.getmaskarray(a)
    mb = np.ma.getmaskarray(b)
    assert np.array_equal(ma, mb)
    valid = ~ma
    assert np.array_equal(
        np.ma.getdata(a)[valid], np.ma.getdata(b)[valid]
    ), f"seed={seed}: {(np.ma.getdata(a)[valid] != np.ma.getdata(b)[valid]).sum()} gates differ"
