"""Bit-identical parity: Rust `region_dealias.sweep_folds` vs the pyart oracle.

Runs many randomized synthetic aliased sweeps plus structured edge cases and
asserts the integer fold fields are exactly equal.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from pyart_ref import ref_sweep_folds  # noqa: E402

import region_dealias  # noqa: E402


def _smooth_field(rng, nrays, ngates):
    """Low-frequency smooth velocity field (a few azimuth/range harmonics)."""
    az = np.linspace(0, 2 * np.pi, nrays, endpoint=False)[:, None]
    rg = np.linspace(0, np.pi, ngates)[None, :]
    f = np.zeros((nrays, ngates))
    for _ in range(4):
        ka = rng.integers(1, 5)
        kr = rng.integers(1, 5)
        pa = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(-1, 1)
        f += amp * np.sin(ka * az + pa) * np.cos(kr * rg)
    return f


def _make_case(seed):
    rng = np.random.default_rng(seed)
    nrays = int(rng.integers(16, 90))
    ngates = int(rng.integers(16, 80))
    nyq = float(rng.uniform(18.0, 35.0))
    wrap = bool(rng.integers(0, 2))

    scale = rng.uniform(1.5, 3.5) * nyq  # push well beyond Nyquist -> aliasing
    true = _smooth_field(rng, nrays, ngates) * scale
    # alias into [-nyq, nyq)
    obs = ((true + nyq) % (2 * nyq)) - nyq
    obs = obs.astype(np.float32)

    mask = rng.random((nrays, ngates)) < rng.uniform(0.0, 0.35)
    # carve a fully-masked band sometimes
    if rng.random() < 0.3:
        c = int(rng.integers(0, ngates))
        mask[:, c : min(ngates, c + int(rng.integers(1, 5)))] = True
    return obs, mask, nyq, wrap


@pytest.mark.parametrize("seed", range(200))
def test_parity_random(seed):
    obs, mask, nyq, wrap = _make_case(seed)
    ref = ref_sweep_folds(obs, mask, nyq, wrap)
    got = region_dealias.sweep_folds(obs, mask, nyq, wrap)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), (
        f"seed={seed} mismatch: {(got != ref).sum()} of {got.size} gates"
    )


def test_all_masked():
    obs = np.zeros((20, 20), np.float32)
    mask = np.ones((20, 20), bool)
    assert np.array_equal(
        region_dealias.sweep_folds(obs, mask, 25.0, True),
        ref_sweep_folds(obs, mask, 25.0, True),
    )


def test_single_region():
    obs = np.full((20, 20), 3.0, np.float32)
    mask = np.zeros((20, 20), bool)
    assert np.array_equal(
        region_dealias.sweep_folds(obs, mask, 25.0, True),
        ref_sweep_folds(obs, mask, 25.0, True),
    )


@pytest.mark.parametrize("wrap", [True, False])
def test_beyond_nyquist(wrap):
    rng = np.random.default_rng(7)
    obs = (rng.uniform(-60, 60, (50, 40))).astype(np.float32)  # exceeds nyquist
    mask = rng.random((50, 40)) < 0.1
    nyq = 25.0
    assert np.array_equal(
        region_dealias.sweep_folds(obs, mask, nyq, wrap),
        ref_sweep_folds(obs, mask, nyq, wrap),
    )
