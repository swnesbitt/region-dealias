#!/usr/bin/env python3
"""
Speed + parity benchmark for region-dealias on a real NEXRAD Level 2 volume:
the 29 June 2023 Midwest derecho, KILX (Lincoln, IL), 18:05:59Z.

It compares the Rust `region_dealias.sweep_folds` against the verbatim
pure-Python Py-ART reference algorithm shipped in this repo
(`tests/pyart_ref.py`, the same oracle the parity tests use) on the lowest
elevation sweep that contains Doppler velocity, and writes:

    benchmarks/results_derecho_2023.json   timing + parity numbers
    benchmarks/parity_derecho_2023.png     3-panel PPI (folded / dealiased / diff)

Run:
    pip install region-dealias metpy matplotlib numpy
    # get the volume (anonymous, public NSF Unidata archive bucket):
    curl -o /tmp/KILX20230629_180559_V06 \
      https://unidata-nexrad-level2.s3.amazonaws.com/2023/06/29/KILX/KILX20230629_180559_V06
    python benchmarks/benchmark_derecho.py /tmp/KILX20230629_180559_V06

The reported speedup is per-sweep wall time (best of N), region-dealias vs the
pure-Python Py-ART reference on the same sweep; absolute numbers vary by machine.
"""
import sys, os, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tests"))   # pyart_ref.py oracle

from pyart_ref import ref_sweep_folds      # noqa: E402
import region_dealias                       # noqa: E402
from metpy.io import Level2File             # noqa: E402
import matplotlib                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402

VOL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/KILX20230629_180559_V06"
N_RUST   = int(os.environ.get("N_RUST", "7"))    # best-of; fast
N_ORACLE = int(os.environ.get("N_ORACLE", "3"))  # best-of; slow

if not os.path.exists(VOL):
    sys.exit("volume not found: %s\nDownload it (see this file's docstring)." % VOL)

# ---- read the lowest sweep that has velocity --------------------------------
f = Level2File(VOL)
best = None
for i, swp in enumerate(f.sweeps):
    if b"VEL" not in swp[0][4]:
        continue
    el = float(np.mean([r[0].el_angle for r in swp]))
    if best is None or el < best[0]:
        best = (el, i, swp)
if best is None:
    sys.exit("no velocity sweeps in volume")
el, si, swp = best
nray = len(swp)
ng = max(r[4][b"VEL"][0].num_gates for r in swp)
vel = np.full((nray, ng), np.nan, np.float32)
az = np.zeros(nray)
for j, r in enumerate(swp):
    hdr, d = r[4][b"VEL"]
    d = np.asarray(d, dtype=np.float32)
    vel[j, :len(d)] = d
    az[j] = r[0].az_angle
nyq = float(np.median([r[3].nyq_vel for r in swp]))
h0 = swp[0][4][b"VEL"][0]
rng = h0.first_gate + np.arange(ng) * h0.gate_width
mask = ~np.isfinite(vel)
wrap = True
print("sweep %d  el=%.2f deg  shape=%s  nyq=%.2f m/s  valid=%d  max|v|=%.1f"
      % (si, el, vel.shape, nyq, (~mask).sum(), np.nanmax(np.abs(vel))), flush=True)

# ---- time both --------------------------------------------------------------
def timeit(fn, n):
    best_t = float("inf"); out = None
    for _ in range(n):
        t = time.perf_counter(); out = fn(); best_t = min(best_t, time.perf_counter() - t)
    return out, best_t

fr, tr = timeit(lambda: region_dealias.sweep_folds(vel, mask, nyq, wrap), N_RUST)
fp, tp = timeit(lambda: ref_sweep_folds(vel, mask, nyq, wrap), N_ORACLE)

identical = bool(np.array_equal(fr, fp))
ndiff = int((fr != fp).sum())
res = dict(volume=os.path.basename(VOL), sweep=int(si), elevation=round(el, 3),
           shape=[int(nray), int(ng)], nyquist=nyq,
           t_pyart_oracle_s=tp, t_rust_s=tr, speedup=tp / tr,
           folds_identical=identical, gates_differing=ndiff,
           valid_gates=int((~mask).sum()), max_abs_folds=int(np.abs(fr).max()))
out_json = os.path.join(HERE, "results_derecho_2023.json")
open(out_json, "w").write(json.dumps(res, indent=2))
print(json.dumps(res, indent=2), flush=True)

# ---- 3-panel PPI ------------------------------------------------------------
twoN = 2 * nyq
folded = np.where(mask, np.nan, vel)
dvr = np.where(mask, np.nan, vel + fr * twoN)
dvp = np.where(mask, np.nan, vel + fp * twoN)
th = np.deg2rad(az)[:, None]
X = rng[None, :] * np.sin(th); Y = rng[None, :] * np.cos(th)
vmax = float(np.nanmax(np.abs(dvr)))
fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
def ppi(a, dd, title, vl):
    m = a.pcolormesh(X, Y, np.ma.masked_invalid(dd), cmap="RdBu_r",
                     vmin=-vl, vmax=vl, shading="nearest")
    a.set_aspect("equal"); a.set_xlim(-150, 150); a.set_ylim(-150, 150)
    a.set_title(title, fontsize=11); a.set_xlabel("X (km)"); a.set_ylabel("Y (km)")
    fig.colorbar(m, ax=a, shrink=0.82, label="m/s")
ppi(ax[0], folded, "Folded velocity input (Nyq = %.0f m/s)" % nyq, nyq)
ppi(ax[1], dvr, "Dealiased - region-dealias (Rust)", vmax)
diff = dvr - dvp
md = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 0.0
ppi(ax[2], diff, "region-dealias minus Py-ART  (max|d| = %g m/s)" % md, 1.0)
fig.suptitle(
    "region-dealias vs Py-ART region-based dealiasing - KILX 2023-06-29 18:05:59Z (%.1f deg) derecho\n"
    "fold fields %s - %d of %d valid gates differ   .   per sweep: Py-ART %.0f ms vs region-dealias %.0f ms  (%.1fx faster)"
    % (el, "IDENTICAL" if identical else "DIFFER", ndiff, res["valid_gates"],
       tp * 1e3, tr * 1e3, res["speedup"]), fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out_png = os.path.join(HERE, "parity_derecho_2023.png")
fig.savefig(out_png, dpi=120)
print("wrote", out_json, "and", out_png, flush=True)
