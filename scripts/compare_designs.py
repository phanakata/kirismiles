"""Compare all simulated designs: stress-strain curves, local material strain,
and an elastic stretchability estimate.

For every results/configs/<name>.npz with a force history (F) and displacement
snapshots (U):
  nominal stress  sigma = F / (W h)            (MPa)
  applied strain  eps   = Delta / L
  local strain    eps_loc = max over triangles of
                  (max principal membrane strain) + (h/2) * |curvature|
      membrane strain: exact Green-Lagrange strain of each linear triangle of
      the P1 snapshot; curvature: angle between the normals of two neighbouring
      triangles divided by the distance between their centroids.
  stretchability  eps* = applied strain at which eps_loc first reaches the
                  yield strain of the material (default 2 %, PET)

The peak local strain sits at crack tips, where the true elastic field is
singular: eps_loc is the strain averaged over one tip element (0.25-0.4 mm,
i.e. 2-3 h, comparable to the tip radius of a laser cut). Use it to rank
designs, not as an absolute failure prediction.

Usage: python scripts/compare_designs.py [--eps-y 0.02]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kirigami.strain import crossing, local_strain, triangle_geometry

R = "results/configs"
INK, MUTED = "#0b0b0b", "#52514e"
RUNS = "results/runs"
# Fixed hue per highlighted design (colour follows the design, not its rank).
# Everything else is drawn as one grey "low-stretch" group.
COLORS = {"alternating": "#2a78d6", "alternating_long": "#6250d6", "rotating_squares": "#e34948",
          "roll": "#eda100", "yaw": "#1baf7a", "yaw_nostrips": "#008300", "pitch_chain": "#e87ba4"}
GREY = "#b9b8b2"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "figure.dpi": 150})


def analyse(path: str | os.PathLike[str], eps_y: float) -> dict[str, Any] | None:
    """Stress-strain and stretchability of one run; None if it has no force or snapshot
    history (or they do not match its load steps). eps_y: yield strain, > 0."""
    if not eps_y > 0:
        raise ValueError(f"the yield strain eps_y must be positive, not {eps_y}")
    with np.load(path) as d:
        if "F" not in d or "U" not in d or len(d["F"]) != len(d["Delta"]) or len(d["F"]) == 0:
            return None
        L, W, h = float(d["L"]), float(d["W"]), float(d["h"])
        x, cells, Delta, F, Us, F_unit = d["x"], d["cells"], d["Delta"], d["F"], d["U"], float(d["F_unit"])
    X, Ginv, orient, pairs, dist = triangle_geometry(x, cells)
    eps = Delta / L
    eloc = np.array([local_strain(X, U.astype(float), cells, Ginv, orient, pairs, dist, h).max()
                     for U in Us])
    sigma = F / (W * h)
    star, note = crossing(eps, eloc, eps_y)
    return dict(eps=eps, sigma=sigma, eloc=eloc, L=L, W=W, h=h,
                E_eff=F_unit * L / (W * h),                     # small-strain modulus, MPa
                stretch=star, note=note, eps_max=float(eps[-1]),
                sigma_star=float(np.interp(star, eps, sigma)),
                sigma_end=float(sigma[-1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps-y", type=float, default=0.02, help="yield strain of the material (> 0)")
    a = ap.parse_args()
    if not a.eps_y > 0:
        ap.error(f"--eps-y must be positive, not {a.eps_y}")
    res: dict[str, dict[str, Any]] = {}
    files = [(os.path.basename(f)[:-4], f) for f in sorted(glob.glob(f"{R}/*.npz"))]
    files += [(os.path.basename(os.path.dirname(f)), f)                 # CLI / api runs
              for f in sorted(glob.glob(f"{RUNS}/*/history.npz"))]
    for name, f in files:
        if name.startswith("mesh_"):
            continue
        r = analyse(f, a.eps_y)
        if r is not None:
            res[name] = r

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    grey_labelled = False
    for name, r in sorted(res.items(), key=lambda kv: kv[0] in COLORS):   # grey first, below
        lab: str | None
        if name in COLORS:
            c, lw, lab = COLORS[name], 1.8, name
        else:
            c, lw = GREY, 1.2
            lab = None if grey_labelled else "low-stretch: single/two cuts, pitch cells"
            grey_labelled = True
        a1.loglog(r["eps"][1:], r["sigma"][1:], color=c, lw=lw, label=lab)
        a2.loglog(r["eps"][1:], r["eloc"][1:], color=c, lw=lw, label=lab)
    e = np.geomspace(1e-4, 0.3, 10)
    a1.loglog(e, 3500 * e, color=MUTED, ls=":", lw=1.2, label="uncut PET, $E\\varepsilon$")
    a1.set(xlabel=r"applied strain $\Delta/L$", ylabel=r"nominal stress $F/(Wh)$ (MPa)",
           title="Stress–strain of all designs")
    a2.axhline(a.eps_y, color=INK, lw=1, ls="--")
    a2.text(1.2e-4, a.eps_y * 1.15, f"yield strain {a.eps_y:.0%}", fontsize=8, color=INK)
    a2.set(xlabel=r"applied strain $\Delta/L$", ylabel="peak local material strain",
           title="Local strain (membrane + bending, at crack tips)")
    a1.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_designs_stress_strain.png")

    table = []
    for name, r in sorted(res.items(), key=lambda kv: -kv[1]["stretch"]):
        table.append(dict(design=name, stretchability=round(r["stretch"], 4),
                          note=r["note"], simulated_to=round(r["eps_max"], 4),
                          stress_at_stretchability_MPa=round(r["sigma_star"], 4),
                          initial_modulus_MPa=round(r["E_eff"], 1)))
    with open(f"{R}/designs_summary.json", "w") as fh:
        json.dump(dict(yield_strain=a.eps_y, designs=table), fh, indent=2)
    for t in table:
        print(t)


if __name__ == "__main__":
    main()
