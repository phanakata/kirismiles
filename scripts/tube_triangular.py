"""Rolled kirigami tube with triangular cuts: Rafsanjani, Jin, Deng and Bertoldi,
PNAS 116, 8200 (2019), SI Fig. S17 (experiment and FE: l = 12 mm, delta = 0.75 mm,
n = 8 cells around).

Cut shape (Rafsanjani et al., Sci. Robot. 3, eaar7555 (2018), SI Sec. S1): cuts on
a triangular lattice a1 = l (cos 30, sin 30), a2 = l (cos 30, -sin 30); each
lattice point p carries one V-shaped cut p + a1 delta/l -> p + a1 -> p + a1 +
a2 (1 - delta/l). Unit cell L = 2 l cos 30 around the tube, H = l along it.
Here s runs around the tube and z along it; the tips point to +z.

Material (PNAS SI): polyester, E = 4.33 GPa, nu = 0.4, t = 76.2 um. The tube is
a rolled flat sheet: it is meshed as a tube, then its natural shape is taken
from curved to flat (the rolling step), then the top ring is pulled.

Nominal stress as in the SI: sigma = F / (n L t).

  ./run.sh python3 scripts/tube_triangular.py --rows 2 --mesh-only
  ./run.sh python3 scripts/tube_triangular.py --rows 2 --eps 0.10
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

E_PET, NU_PET, T_PET = 4330.0, 0.4, 0.0762


def triangular_cuts(n, rows, l=12.0, delta=0.75, margin=1.5, s0=None):
    """V-cuts of the triangular pattern on the unrolled tube. Returns
    (cuts, tips, C, Lz). s0 shifts the pattern around the tube (default L/4, which
    keeps the seam s = 0 away from cut tips and apexes)."""
    c30, s30 = np.cos(np.pi / 6), np.sin(np.pi / 6)
    a1, a2 = l * np.array([c30, s30]), l * np.array([c30, -s30])
    L, H = 2 * l * c30, l
    s0 = L / 4 if s0 is None else s0
    cuts, tips = [], []
    for k in range(2 * rows):                     # lattice rows at z = margin + k H/2
        for j in range(n):
            p = np.array([s0 + (k % 2) * L / 2 + j * L, margin + k * H / 2])
            P1, P3, P5 = p + a1 * delta / l, p + a1, p + a1 + a2 * (1 - delta / l)
            cuts.append([tuple(P1), tuple(P3), tuple(P5)])
            tips += [tuple(P1), tuple(P5)]
    return cuts, tips, n * L, rows * H + 2 * margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--delta", type=float, default=0.75)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--deps", type=float, default=0.0025)
    ap.add_argument("--h-far", type=float, default=1.5)
    ap.add_argument("--h-tip", type=float, default=0.2)
    ap.add_argument("--h-cut", type=float, default=0.5, help="element size along the cuts")
    ap.add_argument("--kerf", type=float, default=0.0,
                    help="cut width in mm (0: zero-width cracks; the PNAS SI models 0.1 mm voids)")
    ap.add_argument("--uncut", action="store_true",
                    help="same tube with no cuts (reference for the cut runs)")
    ap.add_argument("--periodic", action="store_true",
                    help="one period of rows along an infinitely long tube (kerf cuts only)")
    ap.add_argument("--imperfection", type=float, default=0.0,
                    help="max radial geometric imperfection in units of the thickness t "
                         "(kirigami.tube.radial_imperfection; the PNAS SI uses 0.1 t)")
    ap.add_argument("--imp-seed", type=int, default=0, help="seed of the imperfection field")
    ap.add_argument("--predictor", choices=("secant", "none"), default="secant",
                    help="Newton starting guess for each stretch step: 'secant' extrapolates "
                         "the last two converged states (the first step spreads the stretch "
                         "uniformly along the tube); 'none' starts from the last state with "
                         "the whole increment at the moved end (runs before 2026-11-17)")
    ap.add_argument("--mesh-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from kirigami.fvk import Material
    from kirigami.tube import PeriodicTube, Tube, slot, tube_cut_mesh

    if a.periodic:
        # period Lz = rows H; the lattice is shifted so z = 0 crosses the cut arms half
        # way along them, away from tips and apexes
        cuts, tips, C, _ = triangular_cuts(a.n, a.rows, delta=a.delta, margin=3.0)
        Lz = a.rows * 12.0
        if a.kerf <= 0 and not a.uncut:
            raise SystemExit("--periodic needs --kerf > 0 (or --uncut)")
    else:
        cuts, tips, C, Lz = triangular_cuts(a.n, a.rows, delta=a.delta)
    if a.uncut:
        cuts, tips = [], []
    t0 = time.time()
    if a.kerf > 0 or a.uncut:
        msh, ft, info = tube_cut_mesh(C, Lz, [], tips, h_far=a.h_far, h_tip=a.h_tip, h_cut=a.h_cut,
                                      holes=[slot(c, a.kerf) for c in cuts], periodic_z=a.periodic)
    else:
        msh, ft, info = tube_cut_mesh(C, Lz, cuts, tips, h_far=a.h_far, h_tip=a.h_tip, h_cut=a.h_cut)
    print("mesh", {k: v for k, v in info.items() if k not in ("ghost_cells", "master")},
          f"C={C:.3f} Lz={Lz:.2f}  {time.time() - t0:.1f}s", flush=True)
    mat = Material(E=E_PET, nu=NU_PET, h=T_PET)
    imp = {"amp": a.imperfection * T_PET, "seed": a.imp_seed} if a.imperfection > 0 else None
    if a.periodic:
        tb = PeriodicTube(info["R"], Lz, mat, (msh, ft), info, h=a.h_tip, imperfection=imp)
    else:
        tb = Tube(info["R"], Lz, mat, h=a.h_tip, mesh=(msh, ft), ends="clamped", imperfection=imp)
    print("dofs", tb.V.dofmap.index_map.size_global * 3,
          "imperfection", tb.imperfection, flush=True)
    import ufl
    from dolfinx import fem
    if a.periodic:
        # area of one period: the tube minus the slots (their periodic copies included)
        area_fe = fem.assemble_scalar(fem.form(tb.w * ufl.dx(domain=msh)))
        def poly_area(P):
            P = np.asarray(P)
            return 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))
        expect = C * Lz - sum(poly_area(slot(c, a.kerf)) for c in cuts)
        print(f"period area {area_fe:.3f} mm^2 (curved), expected {expect:.3f} (flat)"
              f"; ghost cells {len(info['ghost_cells'])}, tied nodes {len(info['master'])}", flush=True)
        if abs(area_fe - expect) > 1e-3 * expect:
            raise RuntimeError("periodic mesh does not cover one period")
        if a.mesh_only:
            return
    else:
        free = fem.assemble_scalar(fem.form(1.0 * ufl.ds(domain=msh)))
        expect = len(cuts) * (4 * (12.0 - a.delta) + 2 * a.kerf) + 2 * C
        print(f"free edge length {free:.3f} mm, expected {expect:.3f} (2 faces per cut + 2 rings)", flush=True)
        if abs(free - expect) > 1e-3 * expect:
            raise RuntimeError("cuts not fully opened")
        if a.mesh_only:
            return
    tb.setup()
    from kirigami import provenance
    prov = provenance.collect()            # in the container: no git, but the source hash
    tag = ("_periodic" if a.periodic else "") + (f"_imp{a.imperfection:g}" if a.imperfection > 0 else "")
    out = a.out or (f"results/tube/tri_n{a.n}_rows{a.rows}_uncut{tag}" if a.uncut
                    else f"results/tube/tri_n{a.n}_rows{a.rows}_d{a.delta}_k{a.kerf}{tag}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    area = C * T_PET
    X = tb.V.tabulate_dof_coordinates()
    cells = tb.V.dofmap.list
    if a.periodic:
        cells = cells[tb.real_cells]                  # the ghost band is not plotted
    snaps, rows_out = {}, []

    def record(tag):
        U = tb.u.x.array.reshape(-1, 3).copy()
        snaps[tag] = U

    # rolling: natural shape from curved (c = 1) to flat (c = 0); the bottom ring is
    # held, the top ring is free to move along the tube (zero axial force), as when
    # the rolled sheet is left free before the caps are glued on
    for c in np.linspace(1, 0, 6)[1:]:
        ok = tb.step(0.0, c, free_top=True)
        print(f"  roll c={c:.2f} ok={ok}", flush=True)
        if not ok:
            raise RuntimeError("rolling step failed")
    record("rolled")
    D0 = tb.top_uz()                              # length change from rolling
    Lz0 = Lz + D0                                 # strain is measured from the rolled length
    ok = tb.step(D0, 0.0)                         # now hold the top ring there
    def axial_force():
        return tb.force() if a.periodic else tb.mid_band(1.0)["force"]
    F0 = axial_force()
    print(f"  rolled: length change {D0:.4f} mm, axial force {F0:.2e} N", flush=True)
    # stretching, with step halving and, as a last resort, artificial damping
    # the step grows back (x2) only after two successes in a row at the current size:
    # near the pop-up a step that has just been halved would usually fail again when
    # doubled at once, and each failed attempt costs a full Newton loop
    # predictor (starting guess of Newton only; the converged state solves the same
    # equations to the same tolerance): linear extrapolation in eps of the last two
    # converged states, u_n + (u_n - u_{n-1}) (target - e_n) / (e_n - e_{n-1}); from the
    # rolled state, the stretch increment spread uniformly, u_z += dDelta z / Lz
    zfield = np.zeros_like(tb.u.x.array)
    zfield[2::3] = X[:, 2] / Lz
    hist = [(0.0, tb.u.x.array.copy())]

    def predict(target):
        if a.predictor == "none":
            return None
        e1, u1 = hist[-1]
        if len(hist) == 1:
            return u1 + (target - e1) * Lz0 * zfield
        e0, u0 = hist[-2]
        return u1 + (u1 - u0) * ((target - e1) / (e1 - e0))

    e, de, t1, streak = 0.0, a.deps, time.time(), 0
    while e < a.eps - 1e-12:
        target = min(e + de, a.eps)
        ok = tb.step(D0 + target * Lz0, 0.0, guess=predict(target))
        if not ok:
            streak = 0
            if de > a.deps / 16:
                de /= 2
                continue
            tb.eta.value = 1e-3 * mat.Y           # N/mm^3; used only where Newton fails
            ok = tb.step(D0 + target * Lz0, 0.0, guess=predict(target))
            tb.eta.value = 0.0
            if not ok:
                print(f"  failed at eps={target:.5f}", flush=True)
                break
            damped = True
        else:
            damped = False
        e = target
        hist = [hist[-1], (e, tb.u.x.array.copy())]
        streak += 1
        if streak >= 2 and de < a.deps:
            de, streak = min(a.deps, de * 2), 0
        Fz = axial_force()
        rows_out.append({"eps": e, "force": Fz, "sigma_kPa": Fz / area * 1e3, "damped": damped})
        print(f"  eps={e:.4f} F={Fz:.4f} N sigma={Fz / area * 1e3:.2f} kPa"
              f"{' (damped)' if damped else ''}  {time.time() - t1:.0f}s", flush=True)
        if abs(e / 0.005 - round(e / 0.005)) < 1e-6:      # a frame every 0.005 of strain
            record(f"eps{e:.4f}")
        with open(out + ".json", "w") as f:
            json.dump({"args": vars(a), "mesh": {k: v for k, v in info.items() if k not in ("ghost_cells", "master")}, "C": C, "Lz": Lz, "Lz_rolled": Lz0, "F_rolled": F0, "provenance": prov,
                       "rows": rows_out}, f, indent=1)
    record("final")
    np.savez_compressed(out + ".npz", X=X, cells=cells, **snaps)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
