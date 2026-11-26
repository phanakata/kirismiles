"""JSON design specs: the exchange format between people, agents and the solver.

A spec describes one kirigami sheet completely:

{
  "name": "alt_b20_g4_d5",
  "sheet":    {"L": 120, "W": 60},                       # mm; x is the pulled direction
  "material": {"E": 3500, "nu": 0.38, "h": 0.127},       # MPa, -, mm  (optional; PET default)
  "pattern":  {"generator": "alternating",               # either a generator ...
               "params": {"b": 20, "g": 4, "d": 5, "n_rows": 12}},
  # ... or explicit geometry:  "pattern": {"cuts": [[[x, y], [x, y]], ...],
  #                                        "holes": [[[x, y], ...], ...], "solids": [...]}
  "mesh":     {"h_cut": 0.7, "h_far": 3.0, "refine_dist": [1.0, 4.0]},   # optional
  "load":     {"Dmax": 24, "steps": 50},                 # optional; final Delta in mm
  "boundary": {"type": "clamped", "in_plane": false},    # optional; see BOUNDARY_TYPES
  "probes":   {"facet_a": [58.3, 24], "facet_b": [61.7, 24]}             # optional
}

Optional extras: "material": {"model": "neo_hookean"} (membrane law, default "svk");
"mesh": {"h_hinge": 0.1, "hinge_gap": 0.8} (fine elements only at the narrow necks,
so h_cut can be coarse); "load": {"spacing": "linear"} (equal steps instead of the
default geometric ones, which spend half the steps at very small Delta);
"load": {"pre_steps": 5} (geometric spacing only: 5 equal steps up to 0.95 Delta_c
before the post-buckling ramp, instead of the default single step at 0.5 Delta_c, so the
stiffness before buckling is fitted to several points).

Generators are the pattern functions of kirigami/geometry.py; L and W are passed
from "sheet", everything else from "params". Only numpy is needed here, so specs
can be built and checked without FEniCSx.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from kirigami import geometry as geo
from kirigami._types import Pattern, Polyline, Spec
from kirigami import grammar
from kirigami import kirismiles

GENERATORS: dict[str, Callable[..., Pattern]] = {
    "single_cut": geo.single_cut_pattern,
    "two_cuts": geo.two_cuts_pattern,
    "pitch_cell": geo.pitch_pattern,
    "pitch_chain": geo.pitch_chain_pattern,
    "roll_cell": geo.roll_pattern,
    "yaw_cell": geo.yaw_pattern,
    "alternating": geo.alternating_pattern,
    "rotating_squares": geo.rotating_squares_pattern,
    "lattice": grammar.lattice_pattern,          # any pattern string (kirigami/grammar.py)
    "kirismiles": kirismiles.spec_pattern,       # any KiriSMILES string (kirigami/kirismiles.py)
}

DEFAULTS: dict[str, dict[str, Any]] = {
    "material": {"E": 3500.0, "nu": 0.38, "h": 0.127},
    "mesh": {"h_cut": 0.6, "h_far": 3.0, "refine_dist": [1.0, 6.0]},
    "load": {"Dmax_over_L": 0.1, "steps": 40},
    "boundary": {"type": "clamped", "in_plane": False},
}

# Membrane laws of the shell model (spec["material"]["model"], default "svk"); see
# kirigami/shell.py _membrane. "neo_hookean" is for hinges squeezed far in compression.
MATERIAL_MODELS = ("svk", "neo_hookean")

# spec["name"] names the run folder results/runs/<name>: plain letters, digits, _ and -,
# so a name can never point outside that folder ("../x", "C:/x", "a/b").
RUN_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")

# How the sheet is held and loaded (spec["boundary"]["type"]). Delta is the
# imposed stretch along x; Delta / L is the nominal strain.
BOUNDARY_TYPES = {
    "clamped": "x = 0 fixed, x = L moved by (Delta, 0, 0), no rotation at either end "
               "(grips of a tensile tester; the paper's setup)",
    "roller": "x = 0 and x = L move only along x (by 0 and Delta), free to slide in y "
              "and to rotate; one point pinned in y. Uniaxial stress, as in Cho et al. 2014 Fig. S5",
    "biaxial": "all four edges move outward along their normals, by Delta/2 on x = 0, L "
               "and Delta*W/(2L) on y = 0, W, free to slide along the edge: equal-biaxial strain",
}
# spec["boundary"]["in_plane"] = True additionally holds every point at z = 0 and
# drops the bending energy: a pure in-plane (membrane, plane-stress) model, the
# analogue of the plane-strain 2D calculations of rotating-unit kirigami.


def load(path_or_dict: str | os.PathLike[str] | Mapping[str, Any]) -> Spec:
    """Read a spec from a JSON file (or accept a dict) and fill in defaults."""
    spec = (dict(path_or_dict) if isinstance(path_or_dict, Mapping)
            else json.loads(Path(path_or_dict).read_text()))
    for key in DEFAULTS:
        if not isinstance(spec.get(key, {}), Mapping):
            raise ValueError(f"spec[{key!r}] must be an object, not {type(spec[key]).__name__}")
    for key, default in DEFAULTS.items():
        spec[key] = {**default, **spec.get(key, {})}
    spec.setdefault("name", "design")
    spec.setdefault("probes", {})
    if "Dmax" not in spec["load"]:
        spec["load"]["Dmax"] = spec["load"]["Dmax_over_L"] * spec["sheet"]["L"]
    return spec


def pattern(spec: Spec) -> Pattern:
    """The cuts / holes / solids of a spec, generated if needed."""
    p = spec["pattern"]
    L, W = spec["sheet"]["L"], spec["sheet"]["W"]
    if "generator" in p:
        if p["generator"] not in GENERATORS:
            raise ValueError(f"unknown generator {p['generator']!r}; "
                             f"available: {sorted(GENERATORS)}")
        return GENERATORS[p["generator"]](L, W, **p.get("params", {}))
    return {k: [cast(Polyline, [tuple(pt) for pt in g]) for g in p.get(k, [])]   # shapes checked in validate()
            for k in ("cuts", "holes", "solids")}


def _num(v: Any) -> float | None:
    """v as a finite float, or None if it is missing or not a number (bools are not numbers)."""
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def check_name(name: object) -> str:
    """The run name if it is a plain folder name (RUN_NAME); ValueError otherwise."""
    if not (isinstance(name, str) and RUN_NAME.fullmatch(name)):
        raise ValueError(f"name must be 1-64 letters, digits, _ or - (it names the folder "
                         f"results/runs/<name>), not {name!r}")
    return name


def run_folder(root: str | os.PathLike[str], spec: Spec) -> Path:
    """root/<spec name>, created if needed; ValueError for a name that is not a plain folder
    name, so nothing is ever written outside root."""
    d = Path(root) / check_name(spec.get("name", "design"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def validate(spec: Spec) -> list[str]:
    """Cheap checks before meshing. Returns a list of human-readable problems
    (empty if the spec looks fine); it never raises for a malformed spec. Sections
    other than sheet and pattern fall back to DEFAULTS, as in load()."""
    if not isinstance(spec, Mapping):
        return [f"a spec must be a JSON object, not {type(spec).__name__}"]
    errs: list[str] = []
    try:
        check_name(spec.get("name", "design"))
    except ValueError as e:
        errs.append(str(e))
    sec: dict[str, Mapping[str, Any]] = {}
    for key in ("sheet", "pattern", *DEFAULTS):
        v = spec.get(key, DEFAULTS.get(key))
        if v is None:
            return errs + [f"{key} is required"]
        if not isinstance(v, Mapping):
            return errs + [f"{key} must be an object, not {type(v).__name__}"]
        sec[key] = {**DEFAULTS.get(key, {}), **v}
    L, W = _num(sec["sheet"].get("L")), _num(sec["sheet"].get("W"))
    if L is None or W is None:
        return ["sheet.L and sheet.W are required numbers (mm)"]
    if L <= 0 or W <= 0:
        errs.append("sheet dimensions must be positive")
    m = sec["material"]
    E, h, nu = _num(m.get("E")), _num(m.get("h")), _num(m.get("nu"))
    if E is None or h is None or nu is None:
        errs.append("material.E, material.h and material.nu must be numbers")
    elif not (E > 0 and h > 0 and -1 < nu < 0.5):
        errs.append("material needs E > 0, h > 0 and -1 < nu < 0.5")
    if m.get("model", "svk") not in MATERIAL_MODELS:
        errs.append(f"material.model must be one of {list(MATERIAL_MODELS)}")
    mesh = sec["mesh"]
    hc = _num(mesh.get("h_cut"))
    if hc is None or hc <= 0 or not ((_num(mesh.get("h_far")) or 0) > 0):
        errs.append("mesh.h_cut and mesh.h_far must be positive numbers (mm)")
        hc = None
    if mesh.get("h_hinge") is not None and not ((_num(mesh["h_hinge"]) or 0) > 0
                                                and (_num(mesh.get("hinge_gap", 1.0)) or 0) > 0):
        errs.append("mesh.h_hinge and mesh.hinge_gap must be positive")
    load_ = sec["load"]
    if load_.get("spacing", "geometric") not in ("geometric", "linear"):
        errs.append("load.spacing must be 'geometric' or 'linear'")
    steps = load_.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        errs.append("load.steps must be a whole number of at least 1")
    pre = load_.get("pre_steps", 1)
    if isinstance(pre, bool) or not isinstance(pre, (int, np.integer)) or not 1 <= pre <= 50:
        errs.append(f"load.pre_steps must be a whole number from 1 to 50, not {pre!r}")
    b = sec["boundary"]
    if b.get("type") not in BOUNDARY_TYPES:
        errs.append(f"boundary.type must be one of {sorted(BOUNDARY_TYPES)}")
    if not isinstance(b.get("in_plane"), bool):
        errs.append("boundary.in_plane must be true or false")
    spec = {**spec, **sec}
    try:
        pat = pattern(spec)
    except Exception as e:                                  # bad generator params etc.
        return errs + [f"pattern could not be built: {e}"]
    tol = 1e-6
    for kind in ("cuts", "holes", "solids"):
        for i, g in enumerate(pat[kind]):
            a = np.asarray(g, float)
            if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2:
                errs.append(f"{kind}[{i}] must be a list of [x, y] points")
                continue
            if kind == "cuts" and ((a[:, 0] < -tol) | (a[:, 0] > L + tol) |
                                   (a[:, 1] < -tol) | (a[:, 1] > W + tol)).any():
                errs.append(f"cuts[{i}] leaves the sheet; clip it to [0, L] x [0, W]")
            if kind == "cuts" and ((a[:, 0] < tol) | (a[:, 0] > L - tol)).any():
                errs.append(f"cuts[{i}] touches a clamped edge (x = 0 or x = L)")
    if not any(pat[k] for k in ("cuts", "holes")):
        errs.append("pattern has no cuts or holes")
    if hc is not None:
        for i, c in enumerate(pat["cuts"]):
            seg = np.diff(np.asarray(c, float), axis=0)
            if (np.hypot(seg[:, 0], seg[:, 1]) < 2 * hc).any():
                errs.append(f"cuts[{i}] has a segment shorter than 2*h_cut; refine the mesh "
                            "or lengthen it")
    errs += _crossing_cuts(pat["cuts"])
    return errs


def _crossing_cuts(cuts: list[Polyline]) -> list[str]:
    """Zero-width cuts must not cross or touch each other: the crack model cannot
    split junctions. Returns one message per offending pair."""
    segs = [(i, np.asarray(p, float), np.asarray(q, float))
            for i, c in enumerate(cuts) for p, q in zip(c[:-1], c[1:])]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    bad = set()
    for k, (i, p1, q1) in enumerate(segs):
        for j, p2, q2 in segs[k + 1:]:
            if i == j or (i, j) in bad:
                continue
            d1, d2 = cross(p2, q2, p1), cross(p2, q2, q1)
            d3, d4 = cross(p1, q1, p2), cross(p1, q1, q2)
            if d1 * d2 <= 0 and d3 * d4 <= 0 and not (d1 == d2 == 0):   # proper or touching
                bad.add((i, j))
    return [f"cuts[{i}] and cuts[{j}] cross or touch; use slots (holes) where cuts meet"
            for i, j in sorted(bad)]


def save(spec: Spec, path: str | os.PathLike[str]) -> None:
    Path(path).write_text(json.dumps(spec, indent=2))
