"""Build the KiriSMILES database (kirigami/kdb.py) from the stored results.

Sources (read only, nothing in them is changed):
  results/runs/*/spec.json + summary.json    flat-sheet runs of the platform
  results/tube/tri_*.json                    triangular-cut tube runs
  results/tube/rafsanjani_s17a_digitised.json  experiment and FE of Rafsanjani et al.

Output (rebuilt from scratch each time; IDs are content hashes, so they are stable):
  results/kdb/{materials,designs,specimens,results}/<id>.json
  results/kdb/index.sqlite     one table per record kind, key columns + the JSON
  results/kdb/report.json      counts, conversion problems, fields missing per source

  python scripts/build_kdb.py [--out results/kdb] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kirigami import grammar, kirismiles as K, spec as S                    # noqa: E402
from kirigami.argtypes import count                                         # noqa: E402
from kirigami.kdb import (Cylinder, Design, FlatSheet, Generator, Loading, Material,  # noqa: E402
                          Method, Provenance, Quantity, Reference, Result, Specimen, data_ref, hid)

ROOT = Path(__file__).resolve().parent.parent

# what the arrays of each raw-data file mean (kirigami/api.py, scripts/tube_triangular.py)
FLAT_ARRAYS = {
    "L": "sheet length (mm)", "W": "sheet width (mm)", "h": "thickness (mm)",
    "Delta_c": "critical end displacement of linear buckling (mm)",
    "F_unit": "in-plane force for a unit end displacement (N)",
    "x": "reference node coordinates (mm)", "cells": "triangles, node indices",
    "probe_names": "names of the probe points", "Delta": "end displacement per step (mm)",
    "F": "force on the pulled end per step (N)", "U": "displacement of every node per step (mm)",
    "probes": "displacement at the probe points per step (mm)",
    "uz_max": "largest deflection per step (mm)",
    "x0": "reference node coordinates (mm)", "modes": "buckling modes",
}
TUBE_ARRAYS = {
    "X": "reference node coordinates of the tube (mm)",
    "cells": "P2 triangles, node indices (first three are the vertices)",
    "rolled": "displacement after rolling (mm)",
    "eps*": "displacement at the applied strain in the name (mm)",
    "final": "displacement at the last step (mm)",
}
FILE_DESCRIPTIONS = {"history": "load steps: force, displacements, sheet and mesh",
                     "summary": "summary of the run", "spec": "design spec of the run",
                     "provenance": "code version and solver image",
                     "deformation": "animation of the deformation",
                     "stress_strain": "stress-strain plot", "buckling": "buckling modes"}
RUNS, TUBE = ROOT / "results" / "runs", ROOT / "results" / "tube"

KNOWN_MATERIALS = {      # (E MPa, nu) -> name, reference
    (3500.0, 0.38): ("PET, platform default", None),
    (4330.0, 0.4): ("polyester (PET)", Reference(citation_key="rafsanjani2019popups",
                                                 doi="10.1073/pnas.1817763116", role="constants",
                                                 figure="SI")),
    (2.0, 0.49): ("elastomer, nearly incompressible", None),
}
FAMILY = {"alternating": "parallel slits (staggered)", "rotating_squares": "rotating squares",
          "yaw_cell": "yaw actuator cell"}


LIG: dict[str, float] = {}     # geometry key -> narrowest ligament (mm)


class DB:
    """The records being built, by kind and ID, and the problems met on the way."""

    def __init__(self) -> None:
        self.recs: dict[str, dict[str, Any]] = {k: {} for k in ("material", "design", "specimen", "result")}
        self.problems: collections.Counter[str] = collections.Counter()
        self.examples: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
        self.missing: collections.defaultdict[str, collections.Counter[str]] = \
            collections.defaultdict(collections.Counter)

    def add(self, kind: str, rec: Any) -> str:
        """Add a record (Material, Design, Specimen or Result) under its ID; returns the ID."""
        key = {"material": "material_id", "design": "design_id",
               "specimen": "specimen_id", "result": "result_id"}[kind]
        rid = getattr(rec, key)
        old = self.recs[kind].get(rid)
        if old is not None and old.model_dump() != rec.model_dump() and kind != "design":
            self.problem("id collision with different content: " + kind, rid)
        self.recs[kind].setdefault(rid, rec)
        return rid

    def problem(self, what: str, where: object) -> None:
        self.problems[what] += 1
        if len(self.examples[what]) < 5:
            self.examples[what].append(str(where))

    def material(self, E: float, nu: float, model: str = "svk") -> str:
        name, ref = KNOWN_MATERIALS.get((float(E), float(nu)), (None, None))
        m = Material(material_id=Material.make_id(E, nu, model), name=name, model=model,
                     E=Quantity(value=float(E), unit="MPa"), nu=float(nu),
                     references=[ref] if ref else [])
        return self.add("material", m)


# ------------------------------------------------------------- flat-sheet runs
KERF_RE = r"k\d+(?:\.\d+)?"


def split_kerf(canon: str, L: float, W: float) -> tuple[str, float | None]:
    """(design string, kerf). A kerf shared by every cut of a free-form string is
    fabrication: it is taken out of the string and returned. Mixed widths are
    geometry and stay. The split is kept only if string + kerf cut the same sheet."""
    import re
    ks = set(re.findall(KERF_RE, canon))
    if not ks or canon.startswith("P"):
        return canon, None
    ff = K.parse(canon)
    if not isinstance(ff, K.FreeForm):
        return canon, None
    prims = K._flatten(ff.items)
    widths = {getattr(p, "kerf", "n/a") for p in prims if hasattr(p, "kerf")}
    if len(widths) != 1 or None in widths:
        return canon, None
    kerf = float(widths.pop())
    bare = K.canonical(re.sub(KERF_RE, "", canon))
    back = f"({bare})k{K._n(kerf)}"
    if K.geometry_key(back, L, W) != K.geometry_key(canon, L, W):
        return canon, None
    return bare, float(kerf)


def min_ligament(pat: dict[str, Any], tol: float = 1e-6) -> float:
    """Narrowest ligament between two different pieces of a pattern (cuts and the
    outlines of holes and slots), ignoring the free edges. Pieces that touch or
    overlap (slots where cuts meet) count as one piece. inf for a single piece."""
    import numpy as np
    seg = K._segments(pat)
    if len(seg) < 2:
        return float("inf")
    own = np.unique([str(o) for o, _, _ in seg], return_inverse=True)[1]
    n = own.max() + 1
    if n < 2:
        return float("inf")
    P = np.array([p for _, p, _ in seg], float)
    Q = np.array([q for _, _, q in seg], float)
    D = np.full((n, n), np.inf)
    i, j = np.triu_indices(len(seg), 1)
    m = own[i] != own[j]
    i, j = i[m], j[m]
    for a in range(0, len(i), 200000):
        sl = slice(a, a + 200000)
        d = grammar._seg_dist(P[i[sl]], Q[i[sl]], P[j[sl]], Q[j[sl]])
        np.minimum.at(D, (own[i[sl]], own[j[sl]]), d)
    D = np.minimum(D, D.T)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a_, b_ in zip(*np.nonzero(D < tol)):
        parent[find(a_)] = find(b_)
    g = np.array([find(x) for x in range(n)])
    other = g[:, None] != g[None, :]
    return float(D[other].min()) if other.any() else float("inf")


def design_of(spec: dict[str, Any], L: float, W: float) -> tuple[Any, str, str | None, float | None]:
    """(Design, geometry_key, note). Tries to give every pattern a canonical string and
    checks that the string cuts the same geometry as the original generator."""
    p = spec["pattern"]
    gen = p.get("generator")
    gkey = K.geometry_key(S.pattern(spec), L, W)
    code = None
    if gen in ("kirismiles", "lattice"):
        code = p["params"]["code"]
    elif gen in ("alternating", "rotating_squares"):
        try:
            code = grammar.encode(grammar.from_legacy(gen, p["params"], W))
        except (KeyError, TypeError, ValueError):          # parameters outside the grammar
            code = None
    note = None
    kerf = None
    canon: str | None = None
    if code:
        canon = K.canonical(code)
        if K.geometry_key(canon, L, W) != gkey:
            note = "string does not reproduce the generator's geometry"
            canon = None
        else:
            canon, kerf = split_kerf(canon, L, W)
    else:
        canon = None
    if canon:
        d = Design(design_id=Design.make_id(kirismiles=canon), kirismiles=canon,
                   family=FAMILY.get(gen), cuts_as_slots=kerf is not None)
    elif gen and gen not in ("kirismiles", "lattice") and "params" in p:
        g = Generator(name=gen, params=p["params"])
        d = Design(design_id=Design.make_id(generator=g.model_dump()), generator=g,
                   scale_free=False, family=FAMILY.get(gen))
    else:
        d = Design(design_id=Design.make_id(geometry_key=gkey), scale_free=False,
                   generator=Generator(name="explicit geometry", params={"geometry_key": gkey}))
    return d, gkey, note, kerf


def _read_json(f: Path) -> Any:
    """A JSON file's content, or None if it is missing, unreadable or not JSON."""
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def flat_run(db: DB, d: Path) -> None:
    """The design, specimen and result records of one stored flat-sheet run."""
    src = d.relative_to(ROOT).as_posix()
    spec = _read_json(d / "spec.json")
    if not isinstance(spec, dict):
        db.problem("unreadable spec.json", src)
        return
    summ_f = d / "summary.json"
    summ = _read_json(summ_f)
    if summ is None:
        db.problem("no summary.json (unfinished or queued run)", src)
        return
    L, W = float(spec["sheet"]["L"]), float(spec["sheet"]["W"])
    mat = spec.get("material", {})
    for k in ("E", "nu", "h"):
        if k not in mat:
            db.missing["flat run"][f"material.{k} (default used)"] += 1
    E, nu, h = float(mat.get("E", 3500.0)), float(mat.get("nu", 0.38)), float(mat.get("h", 0.127))
    mid = db.material(E, nu, mat.get("model", "svk"))
    try:
        des, gkey, note, kerf = design_of(spec, L, W)
    except (KeyError, TypeError, ValueError) as e:      # an old or hand-edited spec
        db.problem(f"pattern could not be built: {type(e).__name__}", src)
        return
    if note:
        db.problem(note, src)
    db.add("design", des)
    geom = FlatSheet(L=L, W=W)
    if gkey not in LIG:
        LIG[gkey] = min_ligament(S.pattern(spec))
    lig = LIG[gkey]
    dimless = {"L_over_W": L / W, "h_over_W": h / W}
    if lig != float("inf") and lig > 0:
        dimless.update(min_ligament_over_W=lig / W, h_over_min_ligament=h / lig)
        if kerf:
            dimless["kerf_over_min_ligament"] = kerf / lig
    sp = Specimen(specimen_id=Specimen.make_id(des.design_id, "mm", geom.model_dump(), h, mid, kerf),
                  design_id=des.design_id, unit="mm", geometry=geom, thickness=h, material_id=mid,
                  kerf=kerf, geometry_key=gkey, dimensionless=dimless)
    db.add("specimen", sp)

    b = spec.get("boundary") or {}
    if not spec.get("boundary"):
        db.missing["flat run"]["boundary (default clamped assumed)"] += 1
    btype = b.get("type", "clamped")
    load = spec.get("load", {})
    loading = Loading(type="biaxial" if btype == "biaxial" else "uniaxial", boundary=btype,
                      in_plane=bool(b.get("in_plane", False)), max_displacement=load.get("Dmax"),
                      max_strain=(load["Dmax"] / L) if load.get("Dmax") else None,
                      steps=load.get("steps"), spacing=load.get("spacing", "geometric"),
                      stop_local_strain=load.get("stop_local_strain"),
                      imperfection_sign=spec.get("imperfection_sign"))
    method = Method(engine="kirigami", model="Kirchhoff-Love shell (C0 interior penalty)",
                    membrane_law=mat.get("model", "svk"), analysis=summ.get("analysis"),
                    mesh=spec.get("mesh"), seconds=summ.get("seconds"))
    pv = d / "provenance.json"
    p = _read_json(pv) if pv.exists() else None
    if isinstance(p, dict):
        method.code_version, method.image = p.get("code_version"), p.get("image")
        method.source_hash, method.image_id = p.get("source_hash"), p.get("image_id")
    else:
        db.missing["flat run"]["method.code_version (run predates provenance.json)"] += 1

    props = {}
    k = summ.get("stiffness_N_per_mm")
    if k is not None:
        props["stiffness"] = Quantity(value=k, unit="N/mm")
        if loading.type == "uniaxial":
            props["E_eff_over_E"] = Quantity(value=k * L / (E * h * W), unit="1")
    if summ.get("Delta_c_mm") is not None:
        props["Delta_c"] = Quantity(value=summ["Delta_c_mm"], unit="mm")
        props["eps_c"] = Quantity(value=summ["Delta_c_mm"] / L, unit="1")
    if summ.get("strain_final") is not None:
        props["strain_final"] = Quantity(value=summ["strain_final"], unit="1")
    if summ.get("uz_max_final_mm") is not None:
        props["uz_max_final"] = Quantity(value=summ["uz_max_final_mm"], unit="mm")
        props["uz_max_final_over_W"] = Quantity(value=summ["uz_max_final_mm"] / W, unit="1")
    if summ.get("eloc_final") is not None:
        props["local_strain_final"] = Quantity(value=summ["eloc_final"], unit="1")
    props = {n: q for n, q in props.items() if q.value is not None and math.isfinite(q.value)}  # inf: no buckling
    data = {Path(f).stem: data_ref(f, ROOT, description=FILE_DESCRIPTIONS.get(Path(f).stem),
                                   array_descriptions=FLAT_ARRAYS)
            for f in summ.get("files", []) if (ROOT / f).exists()}
    if not data:
        db.missing["flat run"]["raw data file"] += 1
    status = str(summ.get("status", ""))
    res = Result(result_id=hid("kr", {"src": src}), specimen_id=sp.specimen_id, kind="simulation",
                 status=status or None,
                 completed=status.startswith(("completed", "stopped at target")) if status else None,
                 loading=loading, method=method, data=data, properties=props,
                 provenance=Provenance(source=src, created=time.strftime(
                     "%Y-%m-%d", time.localtime(summ_f.stat().st_mtime))))
    db.add("result", res)


# ------------------------------------------------------------------- tube runs
def tri_design(db: DB) -> str:
    g = Generator(name="triangular V-cuts (scripts/tube_triangular.py)",
                  params={"l": 12.0, "delta_over_l": 0.75 / 12.0, "lattice": "triangular",
                          "cut": "V, arms along the two 30-degree lattice directions"})
    # l is a length: written in units of l itself (l = 12 mm in every specimen here)
    d = Design(design_id=Design.make_id(generator=g.model_dump()), generator=g, scale_free=False,
               family="triangular (kirigami shells)",
               references=[Reference(citation_key="rafsanjani2019popups",
                                     doi="10.1073/pnas.1817763116", role="origin", figure="S17"),
                           Reference(doi="10.1126/scirobotics.aar7555", role="origin",
                                     figure="SI Sec. S1 (cut shape)")])
    return db.add("design", d)


def tube_runs(db: DB) -> None:
    did = tri_design(db)
    mid = db.material(4330.0, 0.4)
    for f in sorted(TUBE.glob("tri_n*.json")):
        src = f.relative_to(ROOT).as_posix()
        dd = _read_json(f)
        if not isinstance(dd, dict) or "args" not in dd:
            db.problem("unreadable tube run", src)
            continue
        a = dd["args"]
        if a.get("uncut"):
            g = Generator(name="uncut", params={})
            dz = Design(design_id=Design.make_id(generator=g.model_dump()), generator=g,
                        family="no cuts (reference)")
            this = db.add("design", dz)
        else:
            this = did
        geom = Cylinder(circumference=dd["C"], length=dd["Lz"], n_around=a["n"], rows=a["rows"])
        kerf = a.get("kerf") or None
        sp = Specimen(specimen_id=Specimen.make_id(this, "mm", geom.model_dump(), 0.0762, mid, kerf),
                      design_id=this, unit="mm", geometry=geom, thickness=0.0762, material_id=mid,
                      kerf=kerf, dimensionless=dict(
                          {"t_over_l": 0.0762 / 12.0} if a.get("uncut") else
                          {"t_over_l": 0.0762 / 12.0, "h_over_min_ligament": 0.0762 / 0.75,
                           **({"kerf_over_min_ligament": kerf / 0.75} if kerf else {})}))
        db.add("specimen", sp)
        rows = dd["rows"]
        clean = [r for r in rows if not r.get("damped")]
        e = [r["eps"] for r in clean if r["eps"] <= 0.01 + 1e-9]
        s = [r["sigma_kPa"] for r in clean if r["eps"] <= 0.01 + 1e-9]
        props = {"eps_final": Quantity(value=rows[-1]["eps"] if rows else None, unit="1")}
        if e:
            slope = sum(x * y for x, y in zip(e, s)) / sum(x * x for x in e) / 1e3
            props["E_eff"] = Quantity(value=slope, unit="MPa")
            props["E_eff_over_E"] = Quantity(value=slope / 4330.0, unit="1")
        if clean:
            props["sigma_max_undamped"] = Quantity(value=max(r["sigma_kPa"] for r in clean), unit="kPa")
        periodic = bool(a.get("periodic"))
        res = Result(
            result_id=hid("kr", {"src": src}), specimen_id=sp.specimen_id, kind="simulation",
            status=f"reached eps={rows[-1]['eps']:.4f}" if rows else "no steps",
            completed=bool(rows) and rows[-1]["eps"] >= a["eps"] - 1e-9,
            loading=Loading(boundary="periodic" if periodic else "clamped",
                            max_strain=a["eps"], steps=len(rows), rolled_first=True,
                            damped_steps=sum(1 for r in rows if r.get("damped"))),
            method=Method(engine="kirigami", model="Kirchhoff-Love shell on a closed tube "
                          "(kirigami/tube.py)", analysis="stretch after rolling",
                          mesh={"h_far": a["h_far"], "h_tip": a["h_tip"]},
                          image="dolfinx/dolfinx:v0.9.0"),
            data={k: data_ref(v, ROOT, description=d, array_descriptions=TUBE_ARRAYS)
                  for k, v, d in (("curve", src, "stress-strain rows, settings and provenance"),
                                  ("fields", src[:-5] + ".npz", "displacement snapshots every "
                                                                "0.005 of strain"))
                  if (ROOT / v).exists()},
            properties=props,
            provenance=Provenance(source=src, created=time.strftime(
                "%Y-%m-%d", time.localtime(f.stat().st_mtime)),
                notes="nominal stress sigma = F / (C t), as in the PNAS SI; damped steps "
                      "excluded from derived properties"))
        p = dd.get("provenance")
        if p:
            res.method.code_version = p.get("code_version")
            res.method.source_hash = p.get("source_hash")
        else:
            db.missing["tube run"]["method.code_version (run predates provenance)"] += 1
        db.add("result", res)

    ref_f = TUBE / "rafsanjani_s17a_digitised.json"
    if ref_f.exists():
        ref = json.loads(ref_f.read_text())
        src = ref_f.relative_to(ROOT).as_posix()
        geom = Cylinder(circumference=8 * 2 * 12.0 * 0.8660254037844386, length=None,
                        n_around=8, rows=None)
        sp = Specimen(specimen_id=Specimen.make_id(did, "mm", geom.model_dump(), 0.0762, mid, None),
                      design_id=did, unit="mm", geometry=geom, thickness=0.0762, material_id=mid,
                      dimensionless={"t_over_l": 0.0762 / 12.0})
        db.add("specimen", sp)
        db.missing["digitised"]["specimen length / rows (not in our records)"] += 1
        db.missing["digitised"]["kerf (laser cut width not reported)"] += 1
        cite = Reference(citation_key="rafsanjani2019popups", doi="10.1073/pnas.1817763116",
                         figure="S17A", role="data")
        for curve, kind, meth in (("experiment", "experiment", Method(engine="tensile test")),
                                  ("unit_cell_FEM", "digitised",
                                   Method(engine="Abaqus", model="rolled unit cell, periodic BCs"))):
            eps_c, sig = zip(*ref["eps_sigma_kPa"][curve])
            m = [(x, y) for x, y in zip(eps_c, sig) if x <= 0.01 + 1e-9]
            props = {"sigma_max": Quantity(value=max(sig), unit="kPa")}
            if m:
                slope = sum(x * y for x, y in m) / sum(x * x for x, _ in m) / 1e3
                props["E_eff"] = Quantity(value=slope, unit="MPa")
                props["E_eff_over_E"] = Quantity(value=slope / 4330.0, unit="1")
            db.add("result", Result(
                result_id=hid("kr", {"src": src, "curve": curve}), specimen_id=sp.specimen_id,
                kind=kind, loading=Loading(boundary="periodic" if curve != "experiment" else "clamped",
                                          max_strain=max(eps_c)),
                method=meth, data={"curve": data_ref(src, ROOT, key=curve,
                                                     description="digitised strain-stress curve")},
                properties=props,
                provenance=Provenance(source=src, approximate=True, references=[cite],
                                      notes=ref.get("source"))))


# -------------------------------------------------------------------- output
def write(db: DB, out: str | os.PathLike[str]) -> dict[str, Any]:
    """Write the records, the SQLite index and report.json into `out`, replacing an earlier
    build there. A folder that is not an earlier build (no report.json and index.sqlite) is
    never deleted."""
    out = Path(out)
    if out.exists():
        if not ((out / "report.json").is_file() and (out / "index.sqlite").is_file()):
            raise SystemExit(f"{out} exists and is not a database build (no report.json and "
                             "index.sqlite): choose another --out")
        shutil.rmtree(out)
    for kind, recs in db.recs.items():
        (out / f"{kind}s").mkdir(parents=True)
        for rid, r in recs.items():
            (out / f"{kind}s" / f"{rid}.json").write_text(r.model_dump_json(indent=1, exclude_none=True))
    con = sqlite3.connect(out / "index.sqlite")
    con.executescript("""
      CREATE TABLE materials (material_id TEXT PRIMARY KEY, name TEXT, E REAL, nu REAL, model TEXT, json TEXT);
      CREATE TABLE designs   (design_id TEXT PRIMARY KEY, kirismiles TEXT, generator TEXT,
                              family TEXT, scale_free INTEGER, json TEXT);
      CREATE TABLE specimens (specimen_id TEXT PRIMARY KEY, design_id TEXT, unit TEXT, geometry TEXT,
                              L REAL, W REAL, thickness REAL, material_id TEXT, geometry_key TEXT, json TEXT);
      CREATE TABLE results   (result_id TEXT PRIMARY KEY, specimen_id TEXT, kind TEXT, boundary TEXT,
                              completed INTEGER, E_eff_over_E REAL, source TEXT, json TEXT);
    """)
    for r in db.recs["material"].values():
        con.execute("INSERT INTO materials VALUES (?,?,?,?,?,?)",
                    (r.material_id, r.name, r.E.value, r.nu, r.model, r.model_dump_json()))
    for r in db.recs["design"].values():
        con.execute("INSERT INTO designs VALUES (?,?,?,?,?,?)",
                    (r.design_id, r.kirismiles, r.generator.name if r.generator else None, r.family,
                     int(r.scale_free), r.model_dump_json()))
    for r in db.recs["specimen"].values():
        g = r.geometry
        con.execute("INSERT INTO specimens VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (r.specimen_id, r.design_id, r.unit, g.type, getattr(g, "L", None),
                     getattr(g, "W", None), r.thickness, r.material_id, r.geometry_key,
                     r.model_dump_json()))
    for r in db.recs["result"].values():
        q = r.properties.get("E_eff_over_E")
        con.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?)",
                    (r.result_id, r.specimen_id, r.kind, r.loading.boundary,
                     None if r.completed is None else int(r.completed), q.value if q else None,
                     r.provenance.source, r.model_dump_json()))
    for t, c in (("specimens", "design_id"), ("results", "specimen_id"), ("designs", "kirismiles")):
        con.execute(f"CREATE INDEX idx_{t}_{c} ON {t}({c})")
    con.commit()
    con.close()
    report = {"counts": {k: len(v) for k, v in db.recs.items()},
              "designs_with_string": sum(1 for d in db.recs["design"].values() if d.kirismiles),
              "problems": dict(db.problems), "problem_examples": dict(db.examples),
              "missing_fields": {k: dict(v) for k, v in db.missing.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/kdb", help="replaced if it holds an earlier build")
    ap.add_argument("--limit", type=count, default=None, help="read only the first this many runs")
    a = ap.parse_args()
    db = DB()
    dirs = sorted(d for d in RUNS.glob("*") if d.is_dir())[: a.limit]
    for i, d in enumerate(dirs):
        if not (d / "spec.json").exists():
            db.problem("no spec.json", d.relative_to(ROOT).as_posix())
            continue
        flat_run(db, d)
    tube_runs(db)
    rep = write(db, ROOT / a.out)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
