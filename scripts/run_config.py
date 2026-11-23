"""Run one cut configuration with the nonlinear Kirchhoff-Love shell model.

Usage: python3 scripts/run_config.py <name> [<name> ...]
Configurations are defined in CONFIGS below. Each run writes
results/configs/<name>.npz with the load history, the measured quantities and
P1 snapshots of the displacement at every step (for animations).
"""
import sys
import time

import numpy as np
from dolfinx import fem, geometry, plot
from mpi4py import MPI  # noqa: F401  (MPI is initialised before dolfinx objects are made)

from kirigami import geometry as geo
from kirigami.fvk import Material
from kirigami.shell import ShellSheet

OUT = "results/configs"
PET = dict(E=3500.0, nu=0.38, h=0.127)
PITCH_MESH = dict(h_far=3.0, h_cut=0.8, refine_dist=(1.0, 8.0))

# mesh: geometry function + kwargs; ell: length scale for the initial buckled
# guess; probes: named points where u is recorded; Dmax: final Delta (mm)
CONFIGS = {
    # Fig. 2 check against the FvK result
    "single_cut": dict(mesh=(geo.single_cut, dict(L=182.0, W=100.0, b=80.0)),
                       ell=80.0, Dmax=0.5, steps=20,
                       probes={"centre": (91.0 - 0.3, 50.0)}),
    # the same, to the full range of the FvK post-buckling runs (Delta/W = 0.1), to check
    # the plate model at large rotations
    "single_cut_long": dict(mesh=(geo.single_cut, dict(L=182.0, W=100.0, b=80.0)),
                            ell=80.0, Dmax=10.0, steps=40,
                            probes={"centre": (91.0 - 0.3, 50.0)}),
    # Fig. 3a-i / ii: two parallel cuts -> lift between them
    "two_cuts_ls10": dict(mesh=(geo.two_cuts, dict(L=182.0, W=100.0, b=40.0, ls=10.0)),
                          ell=40.0, Dmax=0.75, steps=25, probes={"centre": (91.0, 50.0)}),
    "two_cuts_ls30": dict(mesh=(geo.two_cuts, dict(L=182.0, W=100.0, b=40.0, ls=30.0)),
                          ell=40.0, Dmax=0.75, steps=25, probes={"centre": (91.0, 50.0)}),
    # Fig. 4a/b: pitch actuator, concave ((2Lc+Lpar)/w = 1.16) vs convex (0.80)
    "pitch_concave": dict(mesh=(geo.pitch_cell, dict(L=180.0, W=60.0, Lc=18.75, Lpar=32.0,
                                                     Lr=95.0, gap=4.0)),
                          ell=30.0, Dmax=14.0, steps=45, mesh_kw=PITCH_MESH,
                          probes={"rib_left": (90.0 - 44.0, 30.0), "rib_right": (90.0 + 44.0, 30.0)}),
    # mesh-convergence check of the concave cell: elements half the size at the cuts
    "pitch_concave_fine": dict(mesh=(geo.pitch_cell, dict(L=180.0, W=60.0, Lc=18.75, Lpar=32.0,
                                                          Lr=95.0, gap=4.0)),
                               ell=30.0, Dmax=14.0, steps=45,
                               mesh_kw=dict(h_far=2.0, h_cut=0.4, refine_dist=(1.0, 8.0)),
                               probes={"rib_left": (90.0 - 44.0, 30.0), "rib_right": (90.0 + 44.0, 30.0)}),
    "pitch_convex": dict(mesh=(geo.pitch_cell, dict(L=180.0, W=60.0, Lc=8.0, Lpar=32.0,
                                                    Lr=95.0, gap=4.0)),
                         ell=30.0, Dmax=14.0, steps=45, mesh_kw=PITCH_MESH,
                         probes={"rib_left": (90.0 - 44.0, 30.0), "rib_right": (90.0 + 44.0, 30.0)}),
    # Fig. 4c (right) / 5: roll actuator; panel probes above and below the
    # central horizontal slit
    "roll": dict(mesh=(geo.roll_cell, dict(L=150.0, W=60.0)),
                 mesh_kw=dict(h_far=3.0, h_cut=0.6, refine_dist=(1.0, 6.0)),
                 ell=30.0, Dmax=22.5, steps=45,
                 probes={"up_out": (90.0, 52.0), "up_in": (90.0, 34.0),
                         "lo_in": (90.0, 26.0), "lo_out": (90.0, 8.0)}),
    # Fig. 4c (left) / 1b / 5: yaw actuator (simplified); probes on the leaf's
    # major axis (tilted 20 deg) at +-0.7 a
    "yaw": dict(mesh=(geo.yaw_cell, dict(L=150.0, W=60.0)),
                mesh_kw=dict(h_far=3.0, h_cut=0.7, refine_dist=(1.0, 6.0)),
                ell=20.0, Dmax=15.0, steps=45,
                probes={"leaf_a": (75.0 - 9.21, 30.0 - 3.35), "leaf_b": (75.0 + 9.21, 30.0 + 3.35)}),
    # Repeating units. Alternating (staggered) cuts: 12 rows, d = 5 mm apart,
    # cuts b = 20 mm, ligaments g = 4 mm. Probes on two neighbouring facets
    # (strip segments between overlapping cuts of consecutive rows) near the
    # centre, to measure their tilt about y.
    "alternating": dict(mesh=(geo.alternating, dict(L=120.0, W=60.0, b=20.0, g=4.0, d=5.0,
                                                   n_rows=12)),
                        mesh_kw=dict(h_far=3.0, h_cut=0.7, refine_dist=(1.0, 4.0)),
                        ell=20.0, Dmax=24.0, steps=50,
                        probes={"facet1_a": (58.3, 24.0), "facet1_b": (61.7, 24.0),
                                "facet2_a": (63.3, 36.0), "facet2_b": (66.7, 36.0)}),
    # Three concave pitch cells in series along x (one column of Fig. 1a)
    "pitch_chain": dict(mesh=(geo.pitch_chain, dict(L=415.0, W=60.0, n=3, cell=125.0,
                                                   Lc=18.75, Lpar=32.0, Lr=95.0, gap=4.0)),
                        mesh_kw=dict(h_far=3.5, h_cut=1.0, refine_dist=(1.0, 6.0)),
                        ell=30.0, Dmax=30.0, steps=45,
                        probes={f"rib{i}_{s}": (82.5 + 125.0 * i + o, 30.0)
                                for i in range(3) for s, o in (("l", -44.0), ("r", 44.0))}),
    # simpler variant without frame strips (the leaf tumbles out of plane)
    "yaw_nostrips": dict(mesh=(geo.yaw_cell, dict(L=150.0, W=60.0, strip=0.0)),
                mesh_kw=dict(h_far=3.0, h_cut=0.7, refine_dist=(1.0, 6.0)),
                ell=20.0, Dmax=15.0, steps=45,
                probes={"leaf_a": (75.0 - 9.21, 30.0 - 3.35), "leaf_b": (75.0 + 9.21, 30.0 + 3.35)}),
}


def probe_fn(msh, pts):
    pts3 = np.array([[x, y, 0.0] for x, y in pts])
    tree = geometry.bb_tree(msh, 2)
    coll = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pts3), pts3)
    cells = [coll.links(i)[0] for i in range(len(pts))]
    return lambda f: np.array([f.eval(p, c) for p, c in zip(pts3, cells)])


def run(name):
    cfg = CONFIGS[name]
    t0 = time.time()
    gfun, gkw = cfg["mesh"]
    msh = gfun(**gkw, **cfg.get("mesh_kw", dict(h_far=4.0, h_cut=0.5)))
    L = gkw["L"]
    mat = Material(**PET)
    sh = ShellSheet(msh, L, mat)
    print(f"[{name}] cells={msh.topology.index_map(2).size_local} "
          f"dofs={sh.V3.dofmap.index_map.size_global * 3}", flush=True)

    # Linear buckling of the flat state (identical for FvK and KL)
    _, N1, F1 = sh.solve_inplane(1.0)
    lams, modes = sh.linear_buckling(N1, nev=3)
    Dc, mode = lams[0], modes[0]
    print(f"[{name}] Delta_c = {lams}  ({time.time()-t0:.0f}s)", flush=True)

    Vz, zmap = sh.V3.sub(2).collapse()
    mz = fem.Function(Vz); mz.interpolate(mode)
    # imperfection: transverse load in the shape of the mode, giving a linear
    # deflection of order 1e-3 h
    f_imp = 1e-3 * mat.h * mat.D / cfg["ell"] ** 4 * mode

    def seed(D, u, s):
        if D > Dc and np.max(np.abs(u.x.array[zmap])) < 0.1 * mat.h:
            u.x.array[zmap] = 0.5 * np.sqrt(cfg["ell"] * (D - Dc)) * mz.x.array

    V1 = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    u1 = fem.Function(V1)
    probe = probe_fn(msh, list(cfg["probes"].values()))
    cells, _, x = plot.vtk_mesh(V1)
    hist = dict(Delta=[], F=[], U=[], probes=[], uz_max=[])

    def record(D, u, s):
        # Saved after every step, so an interrupted run keeps its history
        u1.interpolate(u)
        hist["Delta"].append(D)
        hist["F"].append(s.force)
        hist["U"].append(u1.x.array.reshape(-1, 3).astype(np.float32))
        hist["probes"].append(probe(u))
        hist["uz_max"].append(np.max(np.abs(u.x.array[zmap])))
        np.savez(f"{OUT}/{name}.npz", L=L, W=gkw["W"], h=mat.h, Delta_c=Dc, F_unit=F1,
                 x=x[:, :2], cells=cells.reshape(-1, 4)[:, 1:],
                 probe_names=np.array(list(cfg["probes"])),
                 **{k: np.array(v) for k, v in hist.items()})

    Deltas = np.concatenate([[0.5 * Dc], np.geomspace(1.05 * Dc, cfg["Dmax"], cfg["steps"])])
    sh.solve(Deltas, f_imp=f_imp, observers=(record,), predictor=seed)
    res = hist["Delta"]
    print(f"[{name}] done: {len(res)} steps, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    for n in sys.argv[1:]:
        run(n)
