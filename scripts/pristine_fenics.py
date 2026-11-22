"""Uncut sheet, FEniCS reference fields for the MuJoCo check (run in the FEniCSx container).

On a 120 x 60 mm uncut PET sheet (E 3500 MPa, h 0.127 mm, nu from the command line), clamped at x = 0 and x = L:
  tension     linear plane stress, Delta = 1 mm: displacement field and force F1 (N per mm)
  buckling    compression (Delta = -1 mm) and its first linear buckling load Delta_c and mode
  twist       the x = L clamp turned by theta about the sheet's axis: w = theta (y - W/2) there, dw/dx = 0 at both
              ends; linear Kirchhoff bending, its displacement field and energy
All fields are interpolated to P1 on the mesh vertices and saved with the mesh in results/mujoco_check/pristine_nu<nu>.npz.

    python3 scripts/pristine_fenics.py 0.38
"""
import sys
import time

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from kirigami import geometry as geo
from kirigami.api import _p1_mesh
from kirigami.fvk import Material, Sheet

L, W, H, E = 120.0, 60.0, 0.127, 3500.0
nu = float(sys.argv[1]) if len(sys.argv) > 1 else 0.38
theta = 1e-3                                         # rad, twist of the x = L clamp
t0 = time.time()
msh = geo.mesh_pattern(L, W, geo.empty_pattern(), h_cut=2.0, h_far=2.0, refine_dist=(1.0, 6.0))
sh = Sheet(msh, L, Material(E=E, nu=nu, h=H))
V1, x, cells = _p1_mesh(msh)
P1 = fem.functionspace(msh, ("Lagrange", 1))


def p1(f, space):
    g = fem.Function(space)
    g.interpolate(f)
    return g.x.array.copy()


# tension
u, N1, F1 = sh.solve_inplane(1.0)
U1v = fem.functionspace(msh, ("Lagrange", 1, (2,)))
u_t = p1(u, U1v).reshape(-1, 2)

# compression buckling
_, Nc, Fc = sh.solve_inplane(-1.0)
lams, modes = sh.linear_buckling(Nc, nev=3)
phi = p1(modes[0], P1)
# bending energy of the (max-1) mode, for the Rayleigh quotient in MuJoCo
Eb_phi = msh.comm.allreduce(fem.assemble_scalar(fem.form(sh.bending_energy(modes[0]))), op=MPI.SUM)

# twist: minimise the bending energy with w prescribed on both clamps
Wsp = sh.W
w, v = ufl.TrialFunction(Wsp), ufl.TestFunction(Wsp)
wf = fem.Function(Wsp)
Eb = sh.bending_energy(wf)
a = ufl.replace(ufl.derivative(ufl.derivative(Eb, wf, v), wf, w), {})
f = fem.Constant(msh, 0.0) * v * ufl.dx
dl = fem.locate_dofs_topological(Wsp, 1, sh.left_facets)
dr = fem.locate_dofs_topological(Wsp, 1, sh.right_facets)
g = fem.Function(Wsp)
g.interpolate(lambda X: theta * (X[1] - W / 2))
bcs = [fem.dirichletbc(fem.Constant(msh, 0.0), dl, Wsp), fem.dirichletbc(g, dr)]
wt = LinearProblem(a, f, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                 "pc_factor_mat_solver_type": "mumps"}).solve()
E_tw = msh.comm.allreduce(fem.assemble_scalar(fem.form(sh.bending_energy(wt))), op=MPI.SUM)
w_t = p1(wt, P1)

out = f"results/mujoco_check/pristine_nu{nu:g}.npz"
np.savez(out, x=x, cells=cells, L=L, W=W, h=H, E=E, nu=nu, theta=theta,
         F1=F1, u_tension=u_t, Delta_c=lams[0], phi=phi, Eb_phi=Eb_phi, Fc_unit=Fc, w_twist=w_t, E_twist=E_tw)
D = E * H**3 / (12 * (1 - nu**2))
print(f"nodes {len(x)}  tension F1 {F1:.4f} N/mm (E h W / L = {E * H * W / L:.4f})  "
      f"compression Delta_c {lams[0]:.5f} mm (clamped Euler 4 pi^2 D W / (L^2 F1) = {4 * np.pi**2 * D * W / (L**2 * F1):.5f})  "
      f"twist energy {E_tw:.4e} N mm at theta {theta}  ({time.time() - t0:.0f} s) -> {out}", flush=True)
