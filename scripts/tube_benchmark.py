"""Uncut-tube benchmark for kirigami/tube.py, following Rafsanjani, Jin, Deng and
Bertoldi, PNAS 116, 8200 (2019), SI Sec. 4A and Fig. S13.

Material and sizes from that SI: polyester, E = 4.33 GPa, nu = 0.4, t = 76.2 um;
unit cell L = H = 12 mm, n cells around the tube, so R = n L / (2 pi).

  1. stress-free tube (c = 1) stretched to 15%: radius and axial force against
     the exact St Venant-Kirchhoff uniaxial solution;
  2. rolled flat sheet (c ramped 1 -> 0, the rolling step), then stretched: the
     hoop bending moment carried by one cell, M_z = m_ss H, against SI Eqs. S11
     and S13 (n = 4 and 8).

Run in the solver container:  ./run.sh python3 scripts/tube_benchmark.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kirigami.fvk import Material  # noqa: E402
from kirigami.tube import Tube  # noqa: E402

MAT = Material(E=4330.0, nu=0.4, h=0.0762)
L = H = 12.0
LZ = 3 * H
EPS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]
OUT = "results/tube/benchmark.json"


def radius(n):
    return n * L / (2 * np.pi)


def stretch(tb, c, label):
    rows = []
    for e in EPS:
        ok = tb.step(e * LZ, c)
        v = tb.mid_band(H / 2)
        v.update(eps=e, ok=bool(ok))
        rows.append(v)
        print(f"  {label} eps={e:.3f} R/R0={v['radius'] / tb.R:.5f} F={v['force']:.4f} "
              f"m_ss={v['m_ss']:.5f}", flush=True)
    return rows


def main():
    res = {"material": vars(MAT), "L": L, "H": H, "Lz": LZ}
    # 1. stress-free tube
    n = 4
    tb = Tube(radius(n), LZ, MAT, h=0.5)
    tb.setup()
    res["stress_free_n4"] = {"R": tb.R, "rows": stretch(tb, 1.0, "free n=4")}
    del tb
    # 2. rolled sheets
    for n, h in ((4, 0.5), (8, 0.6)):
        tb = Tube(radius(n), LZ, MAT, h=h)
        tb.setup()
        roll = []
        for c in (1.0, 0.75, 0.5, 0.25, 0.0):
            ok = tb.step(0.0, c)
            v = tb.mid_band(H / 2)
            v.update(c=c, ok=bool(ok))
            roll.append(v)
            print(f"  roll n={n} c={c:.2f} m_ss={v['m_ss']:.5f} R/R0={v['radius'] / tb.R:.5f}", flush=True)
        res[f"rolled_n{n}"] = {"R": tb.R, "roll": roll, "rows": stretch(tb, 0.0, f"rolled n={n}")}
        del tb
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
