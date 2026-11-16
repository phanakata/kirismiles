"""Single cut: compressive strain along the lips of the cut before buckling.

The scaling argument of Dias et al. (Soft Matter 2017) has an in-plane step,
Delta_perp ~ b Delta / W (their Eq. 2): the material along the cut is compressed,
over the length b, by a strain of order Delta / W. This script measures that
strain from the linear in-plane solve, for a unit applied displacement, along a
line just beside the cut (x = L/2 + 0.05 mm), and reports it in units of Delta/W:

  mean    the shortening of the lip from tip to tip, divided by b;
  middle  -eps_yy at the middle of the cut;
  comp    the mean of -eps_yy over the part of the lip that is in compression,
          and the length of that part as a fraction of b.

    ./run.sh python3 -u scripts/single_cut_lip_strain.py
Writes results/single_cut/lip_strain.csv
"""
import csv

import numpy as np
from dolfinx import geometry

from kirigami.fvk import Material, Sheet
from kirigami.geometry import single_cut

L, W, E, NU, H = 182.0, 100.0, 3500.0, 0.38, 0.127
OUT = "results/single_cut/lip_strain.csv"


def uy_along(sh, uh, xs, ys):
    pts = np.column_stack([xs, ys, np.zeros_like(xs)])
    tree = geometry.bb_tree(sh.msh, 2)
    coll = geometry.compute_colliding_cells(sh.msh, geometry.compute_collisions_points(tree, pts), pts)
    cells = [coll.links(i)[0] for i in range(len(pts))]
    return np.array([uh.eval(p, c)[1] for p, c in zip(pts, cells)])


def along(sh, f, xs, ys, comp=None):
    """Values of a Function at points (xs, ys); comp selects a vector component."""
    pts = np.column_stack([xs, ys, np.zeros_like(xs)])
    tree = geometry.bb_tree(sh.msh, 2)
    coll = geometry.compute_colliding_cells(sh.msh, geometry.compute_collisions_points(tree, pts), pts)
    cells = [coll.links(i)[0] for i in range(len(pts))]
    v = np.array([f.eval(p, c) for p, c in zip(pts, cells)])
    return v[:, comp] if comp is not None else v[:, 0]


def main():
    rows, prof = [], {}
    for bw in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        b = bw * W
        msh = single_cut(L, W, b, h_far=4.0, h_cut=0.35)
        sh = Sheet(msh, L, Material(E, NU, H))
        uh, N1, F1 = sh.solve_inplane(1.0)
        lams, modes = sh.linear_buckling(N1, nev=1)
        y = np.linspace(W / 2 - b / 2 + 0.2, W / 2 + b / 2 - 0.2, 801)
        xs = np.full_like(y, L / 2 + 0.05)
        uy = along(sh, uh, xs, y, comp=1)
        phi = along(sh, modes[0], xs, y)
        eyy = np.gradient(uy, y)
        unit = 1.0 / W                                   # Delta / W for Delta = 1
        mean = -(uy[-1] - uy[0]) / (y[-1] - y[0]) / unit
        mid = -eyy[len(y) // 2] / unit
        c = eyy < 0
        comp = -eyy[c].mean() / unit if c.any() else 0.0
        frac = c.mean()
        # compression weighted by the squared slope of the buckling mode along the lip, the
        # combination that sets the post-buckling amplitude in the 1D model of the cut edge
        dphi2 = np.gradient(phi, y) ** 2
        wmode = -np.trapezoid(eyy * dphi2, y) / np.trapezoid(dphi2, y) / unit
        s = (y - y[0]) / (y[-1] - y[0])                  # pinned sine mode over the cut, for comparison
        cs2 = np.cos(np.pi * s) ** 2
        wsine = -np.trapezoid(eyy * cs2, y) / np.trapezoid(cs2, y) / unit
        # 1D Galerkin estimate (delta = A phi in the edge model): (A/b)^2 = 2 Q (gamma_w - gamma_c)
        Q = np.trapezoid(np.gradient(phi, y) ** 2, y) / (b**2 * np.trapezoid(np.gradient(phi, y) ** 4, y))             * float(np.max(np.abs(phi))) ** 2
        rows.append([bw, b, F1, mean, mid, comp, frac, wmode, wsine, lams[0], Q, 2 * Q * wmode])
        prof[f"y_{bw}"], prof[f"eyy_{bw}"], prof[f"phi_{bw}"] = y, eyy, phi
        print(f"b/W={bw}: lip strain / (Delta/W): mean {mean:.3f}  middle {mid:.3f}  "
              f"compressed part {comp:.3f} over {frac:.2f} b;  mode-weighted {wmode:.3f}  "
              f"sine-weighted {wsine:.3f};  1D prefactor {2 * Q * wmode:.3f}   "
              f"(k = {F1:.1f} N/mm, Dc = {lams[0]*1e3:.3f} um)", flush=True)
        del msh, sh, uh, N1, modes
    with open(OUT, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["b_over_W", "b", "F_unit", "lip_mean", "lip_middle", "lip_comp", "comp_fraction",
                     "lip_mode_weighted", "lip_sine_weighted", "Delta_c", "Q", "prefactor_1d"])
        wr.writerows(rows)
    np.savez(OUT.replace(".csv", "_profiles.npz"), **prof)


if __name__ == "__main__":
    main()
