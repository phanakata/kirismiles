"""Fig. 2a/2c: in-plane state and linear buckling of a sheet with a single cut.

Outputs (results/single_cut/):
  convergence.csv        Delta_c vs mesh size / CIP penalty (b/w = 0.8)
  buckling_sweep.csv     Fc/(E h Delta_c) vs b/w for several thicknesses
  inplane_b80.npz        in-plane stress field (Fig. 2a) and first buckling modes (Fig. 2b)
"""
import csv
import sys
import time

import numpy as np
from dolfinx import fem, plot

from kirigami.fvk import Material, Sheet
from kirigami.geometry import single_cut

L, W = 182.0, 100.0
OUT = "results/single_cut"
E, NU = 3500.0, 0.38


def run(b, h, h_cut=0.35, h_far=4.0, alpha=10.0, nev=4):
    msh = single_cut(L, W, b, h_far=h_far, h_cut=h_cut)
    sh = Sheet(msh, L, Material(E, NU, h), alpha=alpha)
    u, N, F1 = sh.solve_inplane(1.0)        # unit applied displacement
    lams, modes = sh.linear_buckling(N, nev=nev)
    return sh, u, N, F1, lams, modes


def p1_field(sh, expr):
    """Interpolate a scalar UFL expression to P1 and return (x, cells, values)."""
    V1 = fem.functionspace(sh.msh, ("Lagrange", 1))
    f = fem.Function(V1)
    f.interpolate(fem.Expression(expr, V1.element.interpolation_points()))
    cells, _, x = plot.vtk_mesh(V1)
    return x[:, :2], cells.reshape(-1, 4)[:, 1:], f.x.array.copy()


def convergence():
    rows = []
    for h_cut, h_far, alpha in [(0.8, 6, 10), (0.5, 4, 10), (0.35, 4, 10), (0.25, 3, 10),
                                (0.18, 3, 10), (0.35, 4, 20), (0.35, 4, 40)]:
        t = time.time()
        sh, _, _, F1, lams, _ = run(80.0, 0.127, h_cut, h_far, alpha, nev=2)
        ndof = sh.W.dofmap.index_map.size_global
        rows.append((h_cut, h_far, alpha, ndof, F1, lams[0], F1 * lams[0]))
        print(f"h_cut={h_cut} alpha={alpha} ndof_w={ndof} F1={F1:.3f} Dc={lams[0]:.6f} "
              f"Fc={F1*lams[0]:.4f} ({time.time()-t:.1f}s)", flush=True)
    with open(f"{OUT}/convergence.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["h_cut", "h_far", "alpha", "ndof_w", "F_unit", "Delta_c", "F_c"])
        wr.writerows(rows)


def sweep():
    rows = []
    for h in (0.10, 0.12, 0.127, 0.14):
        for bw in np.round(np.arange(0.3, 0.81, 0.05), 3):
            b = bw * W
            sh, _, _, F1, lams, _ = run(b, h)
            Dc_th = h**2 / b
            Fc = F1 * lams[0]
            rows.append((h, b, bw, F1, lams[0], Fc, Fc / (E * h * Dc_th)))
            print(f"h={h} b/w={bw:.2f} Dc={lams[0]:.5f} Fc={Fc:.4f} N  Fc/EhDc={rows[-1][-1]:.3f}",
                  flush=True)
    with open(f"{OUT}/buckling_sweep.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["h", "b", "b_over_w", "F_unit", "Delta_c", "F_c", "Fc_over_EhDc"])
        wr.writerows(rows)


def fields_b80():
    import ufl
    h = 0.127
    sh, u, N, F1, lams, modes = run(80.0, h, nev=4)
    # Fig. 2a: w * (sigma_1 + sigma_2) / (E h) with Delta = h  (sigma = N / h)
    x, cells, s = p1_field(sh, W * ufl.tr(N * h) / h / (E * h))
    data = dict(x=x, cells=cells, trace_stress=s, Delta_c=np.array(lams), F_unit=F1)
    for i, m in enumerate(modes):
        _, _, data[f"mode{i}"] = p1_field(sh, m)
    _, _, data["ux"] = p1_field(sh, u[0])
    _, _, data["uy"] = p1_field(sh, u[1])
    np.savez(f"{OUT}/inplane_b80.npz", **data)
    print("Delta_c (b=80):", lams)


if __name__ == "__main__":
    tasks = sys.argv[1:] or ["fields", "convergence", "sweep"]
    if "fields" in tasks:
        fields_b80()
    if "convergence" in tasks:
        convergence()
    if "sweep" in tasks:
        sweep()
