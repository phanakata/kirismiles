"""Stiffness of a stretched kirigami sheet from a finished simulation.

    python tutorials/fit_stiffness.py <run name> [--from 1] [--to 5]

Reads results/runs/<name>/history.npz and fits two straight lines to the force F (N)
against the applied stretch Delta (mm):

  before buckling   F = k0 * Delta, through the origin, over the steps with Delta < Delta_c
                    (with how far those steps stray from the line, and the solver's linear
                    in-plane stiffness for comparison)
  after buckling    F = k * Delta + F0, over the steps whose strain Delta/L is between
                    --from and --to percent (default: from 1 % to the last step)

and writes results/runs/<name>/stiffness.png with the force curve and both fits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kirigami.stiffness import fit_stiffness  # noqa: E402

RUNS = ROOT / "results" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="the run name, a folder in results/runs/")
    ap.add_argument("--from", dest="lo", type=float, default=1.0, help="fit from this strain, %% (default 1)")
    ap.add_argument("--to", dest="hi", type=float, default=None, help="fit up to this strain, %% (default: the end)")
    a = ap.parse_args()
    path = RUNS / a.name / "history.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found: is {a.name!r} a finished simulation?")
    d = np.load(path)
    L, Dc = float(d["L"]), float(d["Delta_c"])
    D, F = d["Delta"], d["F"]                        # mm, N, one value per load step

    try:
        fit = fit_stiffness(D, F, Dc, L, a.lo, a.hi)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    k0, k, F0 = fit["k_before_N_per_mm"], fit["k_after_N_per_mm"], fit["F0_N"]
    if k0 is None:
        raise SystemExit("no load step before buckling: add steps or start the load lower")
    if k is None:
        raise SystemExit(f"fewer than 2 steps between {fit['fit_from_pct']:g} % and "
                         f"{fit['fit_to_pct']:.3g} % strain: widen the range")
    n, m, hi = fit["n_before"], fit["n_after"], fit["fit_to_pct"]

    print(f"buckling at Delta_c = {Dc:.4f} mm (strain {100 * Dc / L:.3f} %)")
    print(f"stiffness before buckling  k0 = {k0:.4g} N/mm   ({n} step{'s' if n != 1 else ''} below Delta_c)")
    if fit["before_spread"] is None:
        print("   one step before buckling: linearity cannot be judged (rerun with \"load\": {\"pre_steps\": 5})")
    else:
        print(f"   the {n} pre-buckling steps lie within {100 * fit['before_spread']:.2g} % of the line F = k0 Delta")
    if "F_unit" in d.files:                          # the solver's linear in-plane stiffness
        print(f"   linear in-plane solve: {float(d['F_unit']):.4g} N/mm (k0 / that = {k0 / float(d['F_unit']):.4f})")
    print(f"stiffness after buckling   k  = {k:.4g} N/mm   (fit over {m} steps, "
          f"{a.lo:g} to {hi:.3g} % strain; F0 = {F0:.4g} N)")
    print(f"the sheet is {k0 / k:.0f} times softer once it has buckled")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, xmax in ((a1, D[-1]), (a2, 4 * Dc)):
        x = np.linspace(0, xmax, 200)
        ax.plot(D, F, "o", color="#1f1f1d", ms=4, label="simulation")
        ax.plot(x, k0 * x, color="#2a78d6", label=f"before buckling: {k0:.3g} N/mm")
        ax.plot(x, k * x + F0, color="#e34948", label=f"after buckling: {k:.3g} N/mm")
        ax.axvline(Dc, color="gray", ls="--", lw=1)
        ax.set(xlim=(0, xmax), ylim=(0, 1.15 * F[D <= xmax].max()),
               xlabel="applied stretch Δ (mm)", ylabel="force F (N)")
        ax.grid(alpha=0.3)
    a1.set_title("Whole run")
    a2.set_title("Zoom on the start (dashed: Δ_c)")
    a1.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = RUNS / a.name / "stiffness.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out.relative_to(RUNS.parent.parent).as_posix()}")


if __name__ == "__main__":
    main()
