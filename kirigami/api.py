"""Simulation entry points that take a JSON spec (kirigami/spec.py).

Run inside the FEniCSx container:
    python3 -m kirigami.api stiffness specs/a.json [b.json ...] # seconds: in-plane stiffness only
    python3 -m kirigami.api buckle   specs/alternating.json   # seconds: stiffness, Delta_c, F_c
    python3 -m kirigami.api simulate specs/alternating.json   # minutes: full post-buckling
    python3 -m kirigami.api export   specs/alternating.json   # exact cut paths (patterns with solids)
Each prints a JSON summary on the last line of stdout and writes its outputs
to results/runs/<name>/. scripts/kirigami_cli.py wraps these for use from the
host, e.g. by agents.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from kirigami import spec as S
from kirigami._types import Spec, Summary

if TYPE_CHECKING:                                  # the solver modules need FEniCSx
    from kirigami.fvk import Sheet
    from kirigami.shell import ShellSheet

RUNS = Path("results/runs")


class _Stop(Exception):
    """Raised by an observer to end the load ramp early on purpose."""


def _mesh_and_material(spec: Spec) -> tuple[Any, Any]:
    from kirigami import geometry as geo
    from kirigami.fvk import Material
    L, W = spec["sheet"]["L"], spec["sheet"]["W"]
    m = spec["mesh"]
    msh = geo.mesh_pattern(L, W, S.pattern(spec), h_cut=m["h_cut"], h_far=m["h_far"],
                           refine_dist=tuple(m["refine_dist"]), h_hinge=m.get("h_hinge"),
                           hinge_gap=m.get("hinge_gap", 1.0))
    return msh, Material(**spec["material"])


def _sheet(spec: Spec) -> Sheet:
    """The flat sheet for the linear analyses (plane stress, linear buckling)."""
    from kirigami.fvk import Sheet
    msh, mat = _mesh_and_material(spec)
    return Sheet(msh, spec["sheet"]["L"], mat)


def _shell(spec: Spec) -> ShellSheet:
    """The Kirchhoff-Love shell for the nonlinear post-buckling run."""
    from kirigami.shell import ShellSheet
    msh, mat = _mesh_and_material(spec)
    return ShellSheet(msh, spec["sheet"]["L"], mat, boundary=spec.get("boundary"))


def _p1_mesh(msh: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    from dolfinx import fem, plot
    V1 = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    cells, _, x = plot.vtk_mesh(V1)
    return V1, x[:, :2], cells.reshape(-1, 4)[:, 1:]


def buckle(spec: Spec, nev: int = 4) -> Summary:
    """Linear analysis: stiffness of the cut sheet and its first buckling loads."""
    if nev < 1:
        raise ValueError(f"nev (number of buckling modes) must be at least 1, not {nev}")
    from dolfinx import fem
    t0 = time.time()
    out = RUNS / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    sh = _sheet(spec)
    _, N1, F1 = sh.solve_inplane(1.0)
    lams, modes = sh.linear_buckling(N1, nev=nev)
    V1 = fem.functionspace(sh.msh, ("Lagrange", 1))
    m1 = fem.Function(V1)
    _, x, cells = _p1_mesh(sh.msh)
    shapes = []
    for m in modes:
        m1.interpolate(m)
        shapes.append(m1.x.array.copy())
    np.savez(out / "buckling.npz", x=x, cells=cells, Delta_c=np.array(lams), modes=np.array(shapes))
    L = spec["sheet"]["L"]
    res = dict(name=spec["name"], analysis="buckle",
               n_cells=int(sh.msh.topology.index_map(2).size_local),
               stiffness_N_per_mm=F1,
               Delta_c_mm=lams[0] if lams else None,
               strain_c=lams[0] / L if lams else None,
               F_c_N=F1 * lams[0] if lams else None,
               Delta_c_higher_modes_mm=lams[1:],
               seconds=round(time.time() - t0, 1),
               files=[str(out / "buckling.npz")])
    (out / "buckling.json").write_text(json.dumps(res, indent=2))
    return res


def stiffness(spec: Spec) -> Summary:
    """Linear in-plane stiffness F/Delta of the flat cut sheet (small strain, no buckling).
    The cheapest response the engine computes: one plane-stress solve."""
    t0 = time.time()
    out = RUNS / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    sh = _sheet(spec)
    _, _, F1 = sh.solve_inplane(1.0)
    res = dict(name=spec["name"], analysis="stiffness",
               n_cells=int(sh.msh.topology.index_map(2).size_local),
               stiffness_N_per_mm=F1, seconds=round(time.time() - t0, 1))
    (out / "stiffness.json").write_text(json.dumps(res, indent=2))
    return res


def stiffness_batch(paths: Iterable[str | os.PathLike[str]]) -> Summary:
    """stiffness() of several specs in one process, so the solver starts once.
    A spec that fails is reported (ok=False) without stopping the others."""
    results: list[dict[str, Any]] = []
    for path in paths:
        try:
            spec = S.load(path)
            errs = S.validate(spec)
            results.append(dict(name=spec.get("name"), ok=False, errors=errs) if errs
                           else dict(ok=True, **stiffness(spec)))
        except Exception as exc:
            results.append(dict(spec=str(path), ok=False, errors=[f"{type(exc).__name__}: {exc}"]))
    return dict(analysis="stiffness_batch", results=results)


def _pair_angles(probes: np.ndarray, names: dict[str, tuple[float, float]]) -> dict[str, dict[str, list[float]]]:
    """For every probe pair 'X_a'/'X_b', the out-of-plane tilt and the in-plane
    turn of the segment a->b (degrees) at each step."""
    ref = {n: np.asarray(p, float) for n, p in names.items()}
    out = {}
    for n in ref:
        if not n.endswith("_a") or n[:-2] + "_b" not in ref:
            continue
        b = n[:-2] + "_b"
        ia, ib = list(names).index(n), list(names).index(b)
        v0 = np.append(ref[b] - ref[n], 0.0)
        v = v0 + probes[:, ib, :] - probes[:, ia, :]
        tilt = np.degrees(np.arcsin(v[:, 2] / np.linalg.norm(v, axis=1)))
        turn = np.degrees(np.arctan2(v[:, 1], v[:, 0]) - np.arctan2(v0[1], v0[0]))
        out[n[:-2]] = dict(tilt_deg=tilt.tolist(), turn_deg=turn.tolist())
    return out


def simulate(spec: Spec) -> Summary:
    """Full nonlinear (Kirchhoff-Love shell) post-buckling under increasing Delta.

    Optional spec keys:
      "imperfection_sign": +1 / -1   which way the seeded buckling mode points
      "load": {"stop_local_strain": 0.02}   end the ramp once the peak local
          material strain (kirigami/strain.py) passes this value; the step that
          crosses it is kept, so the crossing can be interpolated.
    """
    from dolfinx import fem, geometry
    from kirigami import strain as ST
    t0 = time.time()
    out = RUNS / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    sh = _shell(spec)
    msh, mat, L = sh.msh, sh.mat, spec["sheet"]["L"]
    bnd = sh.boundary
    # The linear stiffness and buckling estimate (which seeds the out-of-plane
    # branch) always use the clamped condition; for other boundary types they
    # are only a guide. A purely in-plane model cannot buckle.
    _, N1, F1 = sh.solve_inplane(1.0)
    lams, modes = ([], []) if bnd["in_plane"] else sh.linear_buckling(N1, nev=2)
    if lams:
        Dc, mode = lams[0], modes[0]
    else:                                  # no compression anywhere: the sheet stays flat
        Dc, mode = np.inf, fem.Function(sh.W)
    sign = float(spec.get("imperfection_sign", 1.0))
    ell = spec.get("ell", min(spec["sheet"]["W"], 30.0))   # length scale for the seed amplitude

    Vz, zmap = sh.V3.sub(2).collapse()
    mz = fem.Function(Vz); mz.interpolate(mode)
    mz.x.array[:] *= sign
    f_imp = sign * 1e-3 * mat.h * mat.D / ell ** 4 * mode    # tiny load in the mode's shape

    def seed(D, u, s):                                       # start on the buckled branch
        if D > Dc and np.max(np.abs(u.x.array[zmap])) < 0.1 * mat.h:
            u.x.array[zmap] = 0.5 * np.sqrt(ell * (D - Dc)) * mz.x.array

    names = spec["probes"]
    pts = np.array([[x, y, 0.0] for x, y in names.values()]).reshape(-1, 3)
    if len(pts):
        tree = geometry.bb_tree(msh, 2)
        coll = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pts), pts)
        pcells = [coll.links(i)[0] for i in range(len(pts))]
    V1, x, cells = _p1_mesh(msh)
    u1 = fem.Function(V1)
    hist: dict[str, list[Any]] = dict(Delta=[], F=[], U=[], probes=[], uz_max=[], eloc=[])
    tg = ST.triangle_geometry(x, cells)
    stop_at = spec["load"].get("stop_local_strain")

    def record(D, u, s):
        u1.interpolate(u)
        hist["Delta"].append(D)
        hist["F"].append(s.force)
        hist["U"].append(u1.x.array.reshape(-1, 3).astype(np.float32))
        hist["probes"].append([u.eval(p, c) for p, c in zip(pts, pcells)] if len(pts) else [])
        hist["uz_max"].append(float(np.max(np.abs(u.x.array[zmap]))))
        X, Ginv, orient, pairs, dist = tg
        hist["eloc"].append(float(ST.local_strain(X, hist["U"][-1].astype(float), cells, Ginv,
                                                  orient, pairs, dist, mat.h).max()))
        np.savez(out / "history.npz", L=L, W=spec["sheet"]["W"], h=mat.h, Delta_c=Dc, F_unit=F1,
                 x=x, cells=cells, probe_names=np.array(list(names)),
                 **{k: np.array(v) for k, v in hist.items()})
        if stop_at is not None and hist["eloc"][-1] >= stop_at:
            raise _Stop(f"local strain {hist['eloc'][-1]:.4f} >= {stop_at} at Delta={D:.4f}")

    n, Dmax = spec["load"]["steps"], spec["load"]["Dmax"]
    if spec["load"].get("spacing", "geometric") == "linear":   # equal steps from the start
        Deltas = np.linspace(Dmax / n, Dmax, n)
    elif np.isfinite(Dc) and Dc < Dmax:
        Deltas = np.concatenate([[0.5 * Dc], np.geomspace(1.05 * Dc, Dmax, n)])
    else:                                  # flat sheets can yield below 1 %: start small
        Deltas = np.geomspace(1e-3 * Dmax, Dmax, n)
    status = "completed"
    try:
        sh.solve(Deltas, f_imp=f_imp, observers=(record,), predictor=seed)
    except RuntimeError as e:                                # keep what converged
        status = f"stopped early: {e}"
    except _Stop as e:
        status = f"stopped at target: {e}"
    probes = np.array(hist["probes"])
    summary = dict(name=spec["name"], analysis="simulate", status=status, boundary=bnd,
                   Delta_c_mm=Dc, stiffness_N_per_mm=F1, steps=len(hist["Delta"]),
                   Delta_final_mm=hist["Delta"][-1] if hist["Delta"] else None,
                   strain_final=hist["Delta"][-1] / L if hist["Delta"] else None,
                   uz_max_final_mm=hist["uz_max"][-1] if hist["uz_max"] else None,
                   eloc_final=hist["eloc"][-1] if hist["eloc"] else None,
                   force_N=hist["F"] or None,
                   pair_angles_final={k: dict(tilt_deg=v["tilt_deg"][-1], turn_deg=v["turn_deg"][-1])
                                      for k, v in _pair_angles(probes, names).items()}
                   if len(probes) else {},
                   seconds=round(time.time() - t0, 1),
                   files=[str(out / "history.npz")])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def export_exact(spec: Spec) -> Summary:
    """Cut paths via gmsh booleans (needed for patterns with solids)."""
    from kirigami import export as X
    out = RUNS / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    L, W = spec["sheet"]["L"], spec["sheet"]["W"]
    paths = X.cut_paths(L, W, S.pattern(spec), exact=True)
    X.write_svg(paths, L, W, out / "cut.svg")
    X.write_dxf(paths, out / "cut.dxf")
    return dict(name=spec["name"], analysis="export", n_paths=len(paths),
                files=[str(out / "cut.svg"), str(out / "cut.dxf")])


MESH_FORMATS = ("msh", "inp", "bdf", "vtk")      # gmsh, Abaqus, Nastran, VTK; plus XDMF


def export_mesh(spec: Spec) -> Summary:
    """The finite element mesh of a design, exactly as the solver uses it, for other codes:
    gmsh (.msh), Abaqus (.inp), Nastran (.bdf), VTK (.vtk) and XDMF (.xdmf). Linear
    triangles in mm; the nodes along every cut are already duplicated, so its two lips are
    free edges. Physical groups in the .msh: 1 the sheet, 10 the cut lines, 11 cut mouths."""
    from dolfinx import io
    from mpi4py import MPI
    from kirigami import geometry as geo
    out = RUNS / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    L, W = spec["sheet"]["L"], spec["sheet"]["W"]
    m = spec["mesh"]
    files = [out / f"mesh.{ext}" for ext in MESH_FORMATS]
    msh = geo.mesh_pattern(L, W, S.pattern(spec), h_cut=m["h_cut"], h_far=m["h_far"],
                           refine_dist=tuple(m["refine_dist"]), h_hinge=m.get("h_hinge"),
                           hinge_gap=m.get("hinge_gap", 1.0), msh_file=[str(f) for f in files])
    with io.XDMFFile(MPI.COMM_WORLD, str(out / "mesh.xdmf"), "w") as f:
        f.write_mesh(msh)
    files += [out / "mesh.xdmf", out / "mesh.h5"]
    tdim = msh.topology.dim
    return dict(name=spec["name"], analysis="mesh",
                nodes=int(msh.geometry.x.shape[0]),
                triangles=int(msh.topology.index_map(tdim).size_local),
                element="linear triangle (the solver uses quadratic fields on it)",
                files=[str(f) for f in files if f.exists()])


def export_stl(spec: Spec, thickness: float, kerf: float) -> Summary:
    """A solid model of the cut sheet for 3D printing: thickness > 0 and kerf >= 0, in mm."""
    if not thickness > 0:
        raise ValueError(f"thickness must be positive, not {thickness}")
    if not kerf >= 0:
        raise ValueError(f"kerf must not be negative, not {kerf}")
    from kirigami import export as X
    out = RUNS / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"sheet_t{thickness:g}.stl"
    X.write_stl(spec["sheet"]["L"], spec["sheet"]["W"], S.pattern(spec), thickness, str(f), kerf)
    return dict(name=spec["name"], analysis="stl", thickness_mm=thickness, kerf_mm=kerf,
                files=[str(f)])


COMMANDS = ("stiffness", "buckle", "simulate", "export", "mesh", "stl")

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in COMMANDS or (sys.argv[1] == "stl" and len(sys.argv) != 5):
        raise SystemExit(f"usage: python3 -m kirigami.api {{{'|'.join(COMMANDS)}}} spec.json [...]\n"
                         "       python3 -m kirigami.api stl spec.json <thickness_mm> <kerf_mm>")
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "stiffness":                       # one or more specs: stiffness spec1.json spec2.json ...
        print(json.dumps(stiffness_batch(sys.argv[2:])))
        sys.exit(0)
    spec = S.load(path)
    errs = S.validate(spec)
    if errs:
        print(json.dumps(dict(name=spec.get("name"), ok=False, errors=errs)))
        sys.exit(2)
    if cmd == "buckle":
        res = buckle(spec)
    elif cmd == "simulate":
        res = simulate(spec)
    elif cmd == "export":
        res = export_exact(spec)
    elif cmd == "mesh":
        res = export_mesh(spec)
    elif cmd == "stl":
        res = export_stl(spec, float(sys.argv[3]), float(sys.argv[4]))
    else:
        raise SystemExit(f"unknown command {cmd!r}")
    print(json.dumps(res))
