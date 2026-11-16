"""Single cut: initial post-buckling (Koiter) coefficient from linear solves.

Near the threshold the FvK solution is w = A phi, u = Delta u_hat + A^2 u_2,
where phi is the first buckling mode (largest value 1, so A = delta_0) and u_2
is the in-plane displacement that relaxes the second-order membrane strain

    e_2 = sym grad u_2 + (1/2) grad phi (x) grad phi ,   u_2 = 0 at the clamps.

With K_b = (bending form)[phi, phi] and a_4 = min_u2 int C e_2 : e_2 dA, the
energy is Pi = (1/2) A^2 K_b (1 - Delta/Delta_c) + (1/2) A^4 a_4, so that

    delta_0^2 = c h^2 (Delta/Delta_c - 1),      c = K_b / (2 a_4 h^2),

and the prefactor of (delta_0/b)^2 = beta_1 (Delta - Delta_c)/W at the
threshold is beta_1 = c h^2 W / (b^2 Delta_c). In the FvK plate c does not
depend on h. Everything here is a linear solve, so the map over b/W and L/W
takes seconds per case.

    ./run.sh python3 -u scripts/single_cut_koiter.py
Writes results/single_cut/koiter.csv (rows already present are skipped).
"""
import csv
import gc
import os
import time

import numpy as np
import ufl
from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from kirigami.fvk import Material, Sheet, _membrane_stress
from kirigami.geometry import single_cut

OUT = "results/single_cut/koiter.csv"
E, NU, H, W = 3500.0, 0.38, 0.127, 100.0
BW = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
LW = [1.0, 1.82, 2.5, 3.5, 5.0]
COLS = ["b_over_W", "L_over_W", "F_unit", "Delta_c", "K_b", "K_b_from_Kg", "a_4", "c", "beta1_0",
        "extent_x_over_b", "extent_x_over_L", "seconds"]


def scalar(msh, form):
    return msh.comm.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM)


def koiter(sh, N1, Dc, phi):
    """K_b, K_b computed from the geometric stiffness (a check), and a_4."""
    msh, mat = sh.msh, sh.mat
    Kb = 2.0 * scalar(msh, sh.bending_energy(phi))
    Kb_g = -Dc * scalar(msh, ufl.inner(N1 * ufl.grad(phi), ufl.grad(phi)) * ufl.dx)
    g = 0.5 * ufl.outer(ufl.grad(phi), ufl.grad(phi))
    u, v = ufl.TrialFunction(sh.U), ufl.TestFunction(sh.U)
    eps = lambda z: ufl.sym(ufl.grad(z))
    a = ufl.inner(_membrane_stress(eps(u), mat), eps(v)) * ufl.dx
    rhs = -ufl.inner(_membrane_stress(g, mat), eps(v)) * ufl.dx
    u2 = LinearProblem(a, rhs, bcs=sh._u_bcs(sh.U, 0.0), petsc_options={
        "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}).solve()
    e2 = eps(u2) + g
    a4 = scalar(msh, ufl.inner(_membrane_stress(e2, mat), e2) * ufl.dx)
    return Kb, Kb_g, a4


def extent_x(sh, phi, L, n=2001):
    """Full width at half maximum of |phi| along the pull, on the centre line y = W/2."""
    xs = np.linspace(0.0, L, n)
    xs = xs[np.abs(xs - L / 2) > 0.05]                 # keep off the cut itself
    pts = np.column_stack([xs, np.full_like(xs, W / 2), np.zeros_like(xs)])
    tree = geometry.bb_tree(sh.msh, 2)
    coll = geometry.compute_colliding_cells(sh.msh, geometry.compute_collisions_points(tree, pts), pts)
    v = np.abs(np.array([phi.eval(p, coll.links(i)[0])[0] for i, p in enumerate(pts)]))
    above = xs[v >= 0.5 * v.max()]
    return above.max() - above.min()


def main():
    have = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            have = {(float(r["b_over_W"]), float(r["L_over_W"])) for r in csv.DictReader(f)}
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(COLS)
        for lw in LW:
            for bw in BW:
                if (bw, lw) in have:
                    continue
                t0 = time.time()
                L, b = round(lw * W, 3), round(bw * W, 3)
                msh = single_cut(L, W, b, h_far=4.0, h_cut=0.35)
                sh = Sheet(msh, L, Material(E, NU, H))
                _, N1, F1 = sh.solve_inplane(1.0)
                lams, modes = sh.linear_buckling(N1, nev=2)
                Dc, phi = lams[0], modes[0]
                Kb, Kb_g, a4 = koiter(sh, N1, Dc, phi)
                c = Kb / (2 * a4 * H**2)
                beta = c * H**2 * W / (b**2 * Dc)
                ext = extent_x(sh, phi, L)
                wr.writerow([bw, lw, F1, Dc, Kb, Kb_g, a4, c, beta, ext / b, ext / L,
                             round(time.time() - t0, 1)])
                f.flush()
                print(f"b/W={bw} L/W={lw}: Dc={Dc*1e3:.3f} um  K_b={Kb:.4e} ({Kb_g:.4e})  "
                      f"c={c:.3f}  beta1={beta:.3f}  extent={ext/b:.2f} b = {ext/L:.2f} L  "
                      f"({time.time()-t0:.0f}s)", flush=True)
                del msh, sh, N1, modes, phi
                gc.collect()


if __name__ == "__main__":
    main()
