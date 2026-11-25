"""Meshing of rectangular sheets with cuts, slots and windows using gmsh.

The sheet occupies [0, L] x [0, W]. Loading is along x: the edge x=0 is clamped
and the edge x=L is displaced. Each cut is a polyline given as a list of (x, y)
points. Cuts are embedded in the surface and then split with gmsh's "Crack"
plugin so that the two faces of each cut are independent (free) edges.
Patterns whose slits meet (T-junctions) are modelled with narrow slots
(holes) instead, since the Crack plugin cannot split junctions.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from kirigami._types import Pattern, Point, Polyline

# gmsh / dolfinx are imported inside the functions that need them, so the
# pattern functions below also work in a plain Python without FEniCSx.
if TYPE_CHECKING:
    from dolfinx.mesh import Mesh

Shape = Sequence[Sequence[float]] | np.ndarray      # a polyline or polygon: [(x, y), ...] or an (n, 2) array


def _require(**checks: tuple[float, str]) -> None:
    """Raise ValueError naming every argument that is out of range: checks are name=(value, rule),
    rule '>0', '>=0' or '>=1'."""
    tests: dict[str, Callable[[float], bool]] = {">0": lambda v: v > 0, ">=0": lambda v: v >= 0,
                                                 ">=1": lambda v: v >= 1}
    bad = [f"{name} = {v} (needs {rule[:-1]} {rule[-1]})"
           for name, (v, rule) in checks.items() if not tests[rule](v)]
    if bad:
        raise ValueError("out of range: " + ", ".join(bad))


def build_geometry(L: float, W: float, cuts: Iterable[Shape] = (), holes: Iterable[Shape] = (),
                   solids: Iterable[Shape] = ()) -> None:
    """Create the sheet with its holes, solids and embedded cut lines in the
    current gmsh model (OpenCASCADE kernel) and synchronise it. The material is
    the sheet minus the holes plus the solids; the parts of cuts that lie outside
    the material (inside a hole) are dropped, so a cut crossing a hole edge ends
    on that edge."""
    import gmsh
    occ = gmsh.model.occ
    cuts, holes, solids = list(cuts), list(holes), list(solids)

    def polygon(pts: Shape) -> int:
        tags = [occ.addPoint(x, y, 0) for x, y in pts]
        lines = [occ.addLine(a, b) for a, b in zip(tags, tags[1:] + tags[:1])]
        return occ.addPlaneSurface([occ.addCurveLoop(lines)])

    sheet = [(2, occ.addRectangle(0, 0, 0, L, W))]
    if holes:
        sheet, _ = occ.cut(sheet, [(2, polygon(h)) for h in holes])
    if solids:
        sheet, _ = occ.fuse(sheet, [(2, polygon(s)) for s in solids])
    # Cuts that touch or lie inside a hole or solid outline are trimmed to the material;
    # the others are added untouched (trimming renumbers the lines, which moves mesh nodes).
    outlines = [np.vstack([np.asarray(p, float), np.asarray(p[:1], float)])
                for p in [*holes, *solids]]
    lines: list[int] = []
    trim: list[int] = []
    for poly in cuts:
        pts = [occ.addPoint(x, y, 0) for x, y in poly]
        new = [occ.addLine(a, b) for a, b in zip(pts[:-1], pts[1:])]
        P = np.asarray(poly, float)
        (trim if any(_touch(P, Q) for Q in outlines) else lines).extend(new)
    if trim:
        kept, _ = occ.intersect([(1, l) for l in trim], sheet, removeObject=True, removeTool=False)
        lines += [t for d, t in kept if d == 1]
    if lines:
        occ.fragment(sheet, [(1, l) for l in lines])
    occ.synchronize()


def _sample(poly: Shape, closed: bool, step: float) -> np.ndarray:
    """Points along a polyline (or closed polygon) no more than `step` apart."""
    P = np.asarray(poly, float)
    if closed:
        P = np.vstack([P, P[:1]])
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.vstack(out)


def _touch(P: np.ndarray, Q: np.ndarray, tol: float = 1e-9) -> bool:
    """True if two polylines (closed ones repeat their first point) intersect or
    touch, or if a closed one contains the other."""
    a, b = P[:-1, None, :], P[1:, None, :]
    c, d = Q[None, :-1, :], Q[None, 1:, :]
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]   # noqa: E731
    d1, d2 = cross(b - a, c - a), cross(b - a, d - a)
    d3, d4 = cross(d - c, a - c), cross(d - c, b - c)
    if np.any((d1 * d2 <= tol) & (d3 * d4 <= tol) &
              (np.maximum(a[..., 0], b[..., 0]) >= np.minimum(c[..., 0], d[..., 0]) - tol) &
              (np.maximum(c[..., 0], d[..., 0]) >= np.minimum(a[..., 0], b[..., 0]) - tol) &
              (np.maximum(a[..., 1], b[..., 1]) >= np.minimum(c[..., 1], d[..., 1]) - tol) &
              (np.maximum(c[..., 1], d[..., 1]) >= np.minimum(a[..., 1], b[..., 1]) - tol)):
        return True

    def inside(p: np.ndarray, R: np.ndarray) -> bool:     # point in closed polygon (even-odd)
        x, y = p
        xi, yi, xj, yj = R[:-1, 0], R[:-1, 1], R[1:, 0], R[1:, 1]
        hit = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj == yi, 1, yj - yi) + xi)
        return bool(np.count_nonzero(hit) % 2 == 1)
    closed = lambda R: len(R) > 3 and np.allclose(R[0], R[-1])          # noqa: E731
    return bool((closed(Q) and inside(P[0], Q)) or (closed(P) and inside(Q[0], P)))


def hinge_points(L: float, W: float, cuts: Sequence[Shape] = (), holes: Sequence[Shape] = (),
                 gap: float = 1.0) -> np.ndarray:
    """Centres of the narrow necks of material ("hinges"): midpoints of the places
    where two different cuts or holes, or a cut or hole and the sheet edge, come
    closer than `gap` without touching. Features that touch or overlap (crossing
    slots, cracks running into an edge) form no neck and are skipped. Returns an
    (n, 2) array, with points closer than gap / 2 merged."""
    _require(L=(L, ">0"), W=(W, ">0"), gap=(gap, ">0"))
    step = gap / 4
    polys = ([np.asarray(c, float) for c in cuts] +
             [np.vstack([np.asarray(h, float), np.asarray(h, float)[:1]]) for h in holes])
    feats = [_sample(c, False, step) for c in cuts] + [_sample(h, True, step) for h in holes]
    feats.append(_sample([(0, 0), (L, 0), (L, W), (0, W)], True, step))      # sheet edge
    boxes = np.array([[f[:, 0].min(), f[:, 1].min(), f[:, 0].max(), f[:, 1].max()] for f in feats])
    mids: list[np.ndarray] = []
    for i in range(len(feats)):
        near = np.flatnonzero((boxes[i + 1:, 0] < boxes[i, 2] + gap) & (boxes[i + 1:, 2] > boxes[i, 0] - gap) &
                              (boxes[i + 1:, 1] < boxes[i, 3] + gap) & (boxes[i + 1:, 3] > boxes[i, 1] - gap))
        for j in near + i + 1:
            if j == len(feats) - 1:              # against the sheet edge: only edge-parallel distance
                A = feats[i]
                d = np.minimum.reduce([A[:, 0], L - A[:, 0], A[:, 1], W - A[:, 1]])
                if d.min() < 1e-3:               # the feature reaches the edge: no neck
                    continue
                for p in A[d < gap]:
                    q = p.copy(); k = np.argmin([p[0], L - p[0], p[1], W - p[1]])
                    q[k // 2] = (0.0, L, 0.0, W)[k]
                    mids.append((p + q) / 2)
                continue
            A, B = feats[i], feats[j]
            D = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
            if D.min() >= gap:
                continue
            if _touch(polys[i], polys[j]):       # touching or overlapping: no neck
                continue
            ia, ib = np.nonzero(D < gap)
            mids += list((A[ia] + B[ib]) / 2)
    if not mids:
        return np.zeros((0, 2))
    M = np.array(mids)
    keep: list[np.ndarray] = []
    for p in M[np.lexsort((M[:, 1], M[:, 0]))]:  # greedy merge
        if not keep or np.min(np.hypot(*(np.array(keep) - p).T)) >= gap / 2:
            keep.append(p)
    return np.array(keep)


def sheet_with_cuts(L: float, W: float, cuts: Sequence[Shape] = (), holes: Sequence[Shape] = (),
                    solids: Sequence[Shape] = (), h_far: float = 3.0, h_cut: float = 0.4,
                    h_tip: float | None = None, refine_dist: tuple[float, float] = (2.0, 25.0),
                    h_hinge: float | None = None, hinge_gap: float = 1.0, comm: Any = None,
                    msh_file: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None = None) -> Mesh:
    """Return a dolfinx mesh of an L x W sheet containing the given cuts.

    cuts       zero-width slits, each a polyline [(x, y), ...]
    holes      regions removed from the sheet, each a closed polygon
    solids     regions added back after the holes are removed (e.g. a disc
               and arms inside a window), each a closed polygon
    h_far      element size far from the cuts
    h_cut      element size along the cuts and hole edges
    h_tip      element size at cut end points (defaults to h_cut / 2)
    refine_dist  (d_min, d_max): size grows from h_cut to h_far between these
                 distances from the cuts
    h_hinge    optional element size within hinge_gap / 2 of each hinge (hinge_points
               with hinge_gap), growing to h_far over a further hinge_gap. With it, h_cut can be much
               coarser: pieces that move almost rigidly need few elements.
    comm       MPI communicator (default COMM_WORLD); msh_file: file(s) to write the mesh to.
    Raises ValueError for a size out of range, before gmsh starts.
    """
    _require(L=(L, ">0"), W=(W, ">0"), h_far=(h_far, ">0"), h_cut=(h_cut, ">0"),
             hinge_gap=(hinge_gap, ">0"), **({} if h_tip is None else {"h_tip": (h_tip, ">0")}),
             **({} if h_hinge is None else {"h_hinge": (h_hinge, ">0")}))
    if not 0 <= refine_dist[0] < refine_dist[1]:
        raise ValueError(f"refine_dist = {tuple(refine_dist)} needs 0 <= d_min < d_max")
    import gmsh
    from mpi4py import MPI
    from dolfinx.io import gmshio
    comm = MPI.COMM_WORLD if comm is None else comm
    h_tip = h_cut / 2 if h_tip is None else h_tip
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sheet")

    if comm.rank == 0:
        build_geometry(L, W, cuts, holes, solids)

        # Curves bounding a surface are real edges (outer edges, hole edges);
        # every other curve is embedded in a surface, i.e. part of a cut.
        tol = 1e-4  # OCC pads bounding boxes slightly
        surfs = [s for _, s in gmsh.model.getEntities(2)]
        bnd = {abs(c) for _, c in gmsh.model.getBoundary([(2, s) for s in surfs],
                                                          combined=False, oriented=False)}
        cut_curves = [c for _, c in gmsh.model.getEntities(1) if c not in bnd]

        def on_outer(c: int) -> bool:
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, c)
            return ((abs(x0) < tol and abs(x1) < tol) or (abs(x0 - L) < tol and abs(x1 - L) < tol)
                    or (abs(y0) < tol and abs(y1) < tol) or (abs(y0 - W) < tol and abs(y1 - W) < tol))
        hole_curves = [c for c in bnd if not on_outer(c)]

        # Cut end points: interior ones are crack tips (refined, nodes kept
        # shared); points where a cut meets a free edge (the sheet edge or a hole
        # edge it runs into) are crack mouths, split by the Crack plugin so that
        # the mouth can open.
        free = [c for c in bnd if len(gmsh.model.getAdjacencies(1, c)[0]) == 1]  # not a seam
        bnd_pts = {abs(p) for _, p in gmsh.model.getBoundary([(1, c) for c in free],
                                                              combined=False, oriented=False)}
        cut_pts = {abs(p) for _, p in gmsh.model.getBoundary([(1, c) for c in cut_curves],
                                                              combined=False, oriented=False)}
        ends = [(x, y) for poly in cuts for (x, y) in (poly[0], poly[-1])]
        tip_pts: list[int] = []
        mouth_pts: list[int] = []
        for _, p in gmsh.model.getEntities(0):
            xp, yp, _ = gmsh.model.getValue(0, p, [])
            if p in bnd_pts and p in cut_pts:
                mouth_pts.append(p)
            elif p not in bnd_pts and any(np.hypot(xp - x, yp - y) < tol for x, y in ends):
                tip_pts.append(p)

        gmsh.model.addPhysicalGroup(2, surfs, 1)
        cut_tag = gmsh.model.addPhysicalGroup(1, cut_curves, 10) if cut_curves else None
        mouth_tag = gmsh.model.addPhysicalGroup(0, mouth_pts, 11) if mouth_pts else None

        # Mesh size fields: refine along cuts / hole edges, more at the tips
        f = gmsh.model.mesh.field
        fields: list[int] = []
        if cut_curves or hole_curves:
            d_cut = f.add("Distance"); f.setNumbers(d_cut, "CurvesList", cut_curves + hole_curves)
            f.setNumber(d_cut, "Sampling", 400)
            t_cut = f.add("Threshold"); f.setNumber(t_cut, "InField", d_cut)
            f.setNumber(t_cut, "SizeMin", h_cut); f.setNumber(t_cut, "SizeMax", h_far)
            f.setNumber(t_cut, "DistMin", refine_dist[0]); f.setNumber(t_cut, "DistMax", refine_dist[1])
            fields.append(t_cut)
        if tip_pts:
            d_tip = f.add("Distance"); f.setNumbers(d_tip, "PointsList", tip_pts)
            t_tip = f.add("Threshold"); f.setNumber(t_tip, "InField", d_tip)
            f.setNumber(t_tip, "SizeMin", h_tip); f.setNumber(t_tip, "SizeMax", h_far)
            f.setNumber(t_tip, "DistMin", 0.5); f.setNumber(t_tip, "DistMax", 10.0)
            fields.append(t_tip)
        if h_hinge is not None:
            for xh, yh in hinge_points(L, W, cuts, holes, hinge_gap):
                b = f.add("Ball")
                f.setNumber(b, "XCenter", xh); f.setNumber(b, "YCenter", yh); f.setNumber(b, "ZCenter", 0)
                f.setNumber(b, "Radius", hinge_gap / 2); f.setNumber(b, "Thickness", hinge_gap)
                f.setNumber(b, "VIn", h_hinge); f.setNumber(b, "VOut", h_far)
                fields.append(b)
        if fields:
            fmin = f.add("Min"); f.setNumbers(fmin, "FieldsList", fields)
            f.setAsBackgroundMesh(fmin)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMax", h_far)
        for opt in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
                    "Mesh.MeshSizeFromCurvature"):
            gmsh.option.setNumber(opt, 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

        # Duplicate nodes along the cuts (interior end points stay shared -> crack tips)
        if cut_tag is not None:
            gmsh.plugin.setNumber("Crack", "Dimension", 1)
            gmsh.plugin.setNumber("Crack", "PhysicalGroup", cut_tag)
            if mouth_tag is not None:
                gmsh.plugin.setNumber("Crack", "OpenBoundaryPhysicalGroup", mouth_tag)
            gmsh.plugin.run("Crack")
        gmsh.model.mesh.renumberNodes()
        gmsh.model.mesh.renumberElements()
        if msh_file:              # one file or several; gmsh picks the format by extension
            files = [os.fspath(p) for p in ([msh_file] if isinstance(msh_file, (str, os.PathLike))
                                            else msh_file)]
            for name in [n for n in files if n.endswith(".msh")]:
                gmsh.write(name)                      # with the cut lines and mouths as groups
            others = [n for n in files if not n.endswith(".msh")]
            if others:            # other codes get the triangles only: no line or point elements
                gmsh.model.removePhysicalGroups([(d, t) for d, t in gmsh.model.getPhysicalGroups()
                                                 if d < 2])
                for name in others:
                    gmsh.write(name)

    mesh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return mesh


# ---------------------------------------------------------------- patterns
# A *pattern* is plain data: dict(cuts=[polyline, ...], holes=[polygon, ...],
# solids=[polygon, ...]) in sheet coordinates (mm). Pattern functions build
# them; translate()/tile() combine them into repeating arrays; mesh_pattern()
# (or the *_cell wrappers) turns one into a dolfinx mesh. Patterns can be
# saved as JSON (see kirigami/spec.py) and exported for laser cutting
# (kirigami/export.py).

def empty_pattern() -> Pattern:
    return dict(cuts=[], holes=[], solids=[])


def merge(*patterns: Mapping[str, Iterable[Shape]]) -> Pattern:
    """Concatenate several patterns."""
    out = empty_pattern()
    for pat in patterns:
        for k in out:
            out[k] += [[(p[0], p[1]) for p in g] for g in pat.get(k, [])]
    return out


def translate(pattern: Mapping[str, Iterable[Shape]], dx: float, dy: float) -> Pattern:
    """Shift every cut, hole and solid of a pattern by (dx, dy)."""
    return {k: [[(x + dx, y + dy) for x, y in g] for g in pattern.get(k, [])]
            for k in ("cuts", "holes", "solids")}


def tile(unit: Pattern, nx: int, ny: int, px: float, py: float, origin: Point = (0.0, 0.0),
         stagger: float = 0.0) -> Pattern:
    """Repeat a unit pattern nx times along x (period px) and ny times along y
    (period py). Every odd column is shifted by `stagger` along y, which gives
    alternating (brick-like) arrays."""
    out = empty_pattern()
    for i in range(nx):
        for j in range(ny):
            dy = origin[1] + j * py + (stagger if i % 2 else 0.0)
            out = merge(out, translate(unit, origin[0] + i * px, dy))
    return out


def clip_to_sheet(pattern: Pattern, L: float, W: float, min_len: float = 1.0) -> Pattern:
    """Clip straight two-point cuts to the sheet [0, L] x [0, W] and drop
    pieces shorter than min_len. A clipped cut that reaches an edge becomes an
    edge crack. Polyline cuts, holes and solids are kept unchanged."""
    cuts: list[Polyline] = []
    for c in pattern["cuts"]:
        if len(c) != 2:
            cuts.append(c)
            continue
        (x0, y0), (x1, y1) = c
        t0, t1 = 0.0, 1.0
        for p, q, lo, hi in ((x0, x1 - x0, 0.0, L), (y0, y1 - y0, 0.0, W)):
            if abs(q) < 1e-12:
                if p < lo or p > hi:
                    t0, t1 = 1.0, 0.0
                continue
            ta, tb = sorted(((lo - p) / q, (hi - p) / q))
            t0, t1 = max(t0, ta), min(t1, tb)
        if t1 - t0 <= 0:
            continue
        a = (x0 + t0 * (x1 - x0), y0 + t0 * (y1 - y0))
        b = (x0 + t1 * (x1 - x0), y0 + t1 * (y1 - y0))
        if np.hypot(b[0] - a[0], b[1] - a[1]) >= min_len:
            cuts.append([a, b])
    return dict(pattern, cuts=cuts)


def mesh_pattern(L: float, W: float, pattern: Mapping[str, Sequence[Shape]], **kw: Any) -> Mesh:
    """Mesh an L x W sheet containing a pattern (see sheet_with_cuts for kw)."""
    return sheet_with_cuts(L, W, cuts=pattern.get("cuts", ()), holes=pattern.get("holes", ()),
                           solids=pattern.get("solids", ()), **kw)


# ------------------------------------------------------ fundamental patterns

def single_cut_pattern(L: float, W: float, b: float, x_cut: float | None = None) -> Pattern:
    """Single straight cut of length b, parallel to the clamped edges (along y),
    centred in the sheet (Fig. 2)."""
    _require(b=(b, ">0"))
    x_cut = L / 2 if x_cut is None else x_cut
    return dict(cuts=[[(x_cut, W / 2 - b / 2), (x_cut, W / 2 + b / 2)]], holes=[], solids=[])


def two_cuts_pattern(L: float, W: float, b: float, ls: float) -> Pattern:
    """Two parallel cuts of length b (along y), a distance ls apart (Fig. 3a)."""
    _require(b=(b, ">0"), ls=(ls, ">0"))
    return dict(cuts=[[(L / 2 + s, W / 2 - b / 2), (L / 2 + s, W / 2 + b / 2)]
                      for s in (-ls / 2, ls / 2)], holes=[], solids=[])


def pitch_pattern(L: float, W: float, Lc: float, Lpar: float, Lr: float, gap: float,
                  xm: float | None = None) -> Pattern:
    """Pitch actuator unit cell (Fig. 4a), centred at x = xm (default L/2).

    An inner ribbon of length Lr (along x) and width Lpar (along y) is cut free
    on all sides except for a gap of width `gap` in the middle of its two long
    sides. Two edge cracks of length Lc run from y=0 and y=W along x=xm,
    through the gaps. (2 Lc + Lpar)/W < 1: convex cell (lift);
    > 1: concave cell (rotation about y).
    """
    _require(Lc=(Lc, ">=0"), Lpar=(Lpar, ">0"), Lr=(Lr, ">0"), gap=(gap, ">0"))
    xm = L / 2 if xm is None else xm
    ym = W / 2
    x1, x2 = xm - Lr / 2, xm + Lr / 2
    yb, yt = ym - Lpar / 2, ym + Lpar / 2
    g = gap / 2
    cuts: list[Polyline] = [[(xm - g, yt), (x1, yt), (x1, yb), (xm - g, yb)],
            [(xm + g, yt), (x2, yt), (x2, yb), (xm + g, yb)],
            [(xm, 0.0), (xm, Lc)],
            [(xm, W), (xm, W - Lc)]]
    return dict(cuts=cuts, holes=[], solids=[])


def _slot(p: Sequence[float], q: Sequence[float], t: float) -> Polyline:
    """Rectangle of width t around the segment p-q (a slit with kerf t)."""
    _require(t=(t, ">0"))
    a, b = np.asarray(p, float), np.asarray(q, float)
    if np.allclose(a, b):
        raise ValueError(f"a slot needs two different end points, got {tuple(p)} twice")
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]]) * t / 2
    return [_pt(a - n), _pt(b - n), _pt(b + n), _pt(a + n)]


def _pt(v: np.ndarray) -> Point:
    return (float(v[0]), float(v[1]))


def roll_pattern(L: float = 150.0, W: float = 60.0, kerf: float = 1.0) -> Pattern:
    """Roll actuator (Fig. 4c, right), traced from the paper's schematic and
    scaled to W = 60 mm. Mirror-symmetric about x = L/2. On each side:
      - edge cracks from y=0 and y=W at |x - L/2| = 49 mm, leaving a 13.5 mm gap
      - a long internal slit at |x - L/2| = 41.6 mm (y from 2.9 to W - 3.5)
      - short edge notches (7.5 mm) at |x - L/2| = 34.4 mm
      - a horizontal slit along y = W/2 from the long slit (T-junction) to an
        arrow-shaped end that leaves a 2.5 mm neck at the centre.
    All slits are slots of width kerf; they overlap at the T-junctions.
    """
    xc, yc = L / 2, W / 2
    k = kerf
    holes: list[Polyline] = []
    for s in (-1, 1):
        X = lambda d, s=s: xc + s * d     # noqa: E731
        holes += [_slot((X(49.0), -1.0), (X(49.0), yc - 6.75), k),          # edge cracks
                  _slot((X(49.0), W + 1.0), (X(49.0), yc + 6.75), k),
                  _slot((X(41.6), 2.9), (X(41.6), W - 3.5), k),            # long slit
                  _slot((X(34.4), -1.0), (X(34.4), 7.5), k),               # edge notches
                  _slot((X(34.4), W + 1.0), (X(34.4), W - 7.5), k),
                  _slot((X(41.6), yc), (X(5.0), yc), k),                   # horizontal slit
                  [(X(1.25), yc - 2.9), (X(5.5), yc), (X(1.25), yc + 2.9)]]  # arrow tip
    return dict(cuts=[], holes=holes, solids=[])


def _thick_curve(pts: Shape, width: float) -> Polyline:
    """Polygon of constant width around a sampled 2D curve."""
    P = np.asarray(pts, float)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]]) * width / 2
    return [_pt(p) for p in np.vstack([P + n, (P - n)[::-1]])]


def _bezier(ctrl: Shape, n: int = 40) -> np.ndarray:
    """n points along the cubic Bezier curve with the four control points ctrl."""
    P = np.asarray(ctrl, float)
    s = np.linspace(0, 1, n)[:, None]
    return ((1 - s) ** 3 * P[0] + 3 * (1 - s) ** 2 * s * P[1]
            + 3 * (1 - s) * s ** 2 * P[2] + s ** 3 * P[3])


def yaw_pattern(L: float = 150.0, W: float = 60.0, window: tuple[float, float] = (45.0, 105.0),
                a: float = 14.0, b: float = 8.0, tilt: float = 20.0, arm_width: float = 3.0,
                strip: float = 4.0, crack: float = 0.5) -> Pattern:
    """Yaw actuator after Fig. 4c (left) / Fig. 1b. Point-symmetric about the
    sheet centre.

    A window x0 < x < x1 is cut out, leaving two thin frame strips of width
    `strip` along y=0 and y=W that join the clamped pads. Inside the window an
    elliptical leaf (semi-axes a, b, tilted by `tilt` degrees) hangs from two
    S-shaped arms anchored on the top strip near the left end and on the
    bottom strip near the right end. Edge cracks beside the window (from y=W
    at x0 and from y=0 at x1, reaching `crack` * W) make those corners
    compliant, as in the paper's schematic. Stretching the strips pulls the
    arm anchors apart, loading the leaf by a couple that turns it about z.

    strip=0 gives a simpler variant without strips or edge cracks: the pads
    are joined only through the arms and the leaf.
    """
    _require(a=(a, ">0"), b=(b, ">0"), arm_width=(arm_width, ">0"), strip=(strip, ">=0"),
             crack=(crack, ">=0"))
    if not window[0] < window[1]:
        raise ValueError(f"window = {tuple(window)} needs x0 < x1")
    xc, yc = L / 2, W / 2
    x0, x1 = window
    th = np.radians(tilt)
    s = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    leaf = np.column_stack([a * np.cos(s), b * np.sin(s)]) @ np.array(
        [[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]) + [xc, yc]
    mirror = lambda P: np.column_stack([2 * xc - P[:, 0], 2 * yc - P[:, 1]])     # noqa: E731
    cuts: list[Polyline] = []
    if strip > 0:
        # arm hanging from the top strip, curving down into the leaf's upper flank
        arm = _bezier([(x0 + 6.0, W - strip / 2), (x0 + 6.0, yc + 2.0),
                       (xc - 16.0, yc + 14.0), (xc - 2.0, yc + 4.0)])
        holes: list[Polyline] = [[(x0, strip), (x1, strip), (x1, W - strip), (x0, W - strip)]]
        cuts = [[(x0 - 1.5, W), (x0 - 1.5, W - crack * W)],
                [(x1 + 1.5, 0.0), (x1 + 1.5, crack * W)]]
    else:
        # arm from the left pad (top) to the upper flank of the leaf
        arm = _bezier([(x0 - 2.0, W - 8.0), (x0 + 14.0, W - 2.0),
                       (xc + 4.0, W - 6.0), (xc - 2.0, yc + 4.0)])
        holes = [[(x0, -1.0), (x1, -1.0), (x1, W + 1.0), (x0, W + 1.0)]]
    solids = [[_pt(p) for p in leaf], _thick_curve(arm, arm_width),
              _thick_curve(mirror(arm), arm_width)]
    return dict(cuts=cuts, holes=holes, solids=solids)


# -------------------------------------------------------- repeating patterns

def alternating_pattern(L: float, W: float, b: float, g: float, d: float, n_rows: int) -> Pattern:
    """Classic alternating (staggered) kirigami: n_rows rows of straight cuts
    parallel to the clamped edges (along y), d apart along x. Within a row,
    cuts of length b are separated by ligaments of length g (period b + g).
    Consecutive rows are shifted by half a period, so each cut faces a
    ligament of the next row. Rows are centred in the sheet; cuts that cross
    y = 0 or y = W are clipped and become edge cracks."""
    _require(b=(b, ">0"), g=(g, ">0"), d=(d, ">0"), n_rows=(n_rows, ">=1"))
    p = b + g
    unit: Pattern = dict(cuts=[[(0.0, -b / 2), (0.0, b / 2)]], holes=[], solids=[])
    m = int(np.ceil(W / p)) + 2                       # enough cuts to cover the width
    x0 = L / 2 - (n_rows - 1) * d / 2
    pat = tile(unit, n_rows, 2 * m + 1, d, p, origin=(x0, W / 2 - m * p), stagger=p / 2)
    return clip_to_sheet(pat, L, W, min_len=g)


def rotating_squares_pattern(L: float, W: float, a: float, t: float, nx: int) -> Pattern:
    """Rotating-squares kirigami: a band of nx x (W/a) square panels of side a,
    centred along x and spanning the full width, joined at their corners by
    hinges of width ~t/2. Slits of length 2a - t are centred on the lattice
    vertices, horizontal (along x) at vertices with i + j even and vertical at
    i + j odd, so each slit ends t/2 short of the middle of a perpendicular
    one. Stretching rotates the panels in-plane (an auxetic mechanism)."""
    _require(a=(a, ">0"), t=(t, ">0"), nx=(nx, ">=1"))
    if not t < 2 * a:
        raise ValueError(f"the hinge t = {t} must be shorter than 2a = {2 * a}")
    ny = int(round(W / a))
    x0 = L / 2 - nx * a / 2
    half = a - t / 2
    cuts: list[Polyline] = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = x0 + i * a, j * a
            if (i + j) % 2 == 0:
                if 0 < i < nx and 0 < j < ny:       # not on the pads or along the free edges
                    cuts.append([(x - half, y), (x + half, y)])
            else:
                cuts.append([(x, y - half), (x, y + half)])
    return clip_to_sheet(dict(cuts=cuts, holes=[], solids=[]), L, W, min_len=t)


def pitch_chain_pattern(L: float, W: float, n: int, cell: float, Lc: float, Lpar: float, Lr: float,
                        gap: float) -> Pattern:
    """n pitch unit cells in series along x (period `cell`), centred in the
    sheet: the repeating arrangement of Fig. 1a, one column."""
    _require(n=(n, ">=1"), cell=(cell, ">0"))
    x0 = L / 2 - (n - 1) * cell / 2
    return merge(*[pitch_pattern(L, W, Lc, Lpar, Lr, gap, xm=x0 + i * cell) for i in range(n)])


# ------------------------------------------- mesh wrappers (used by drivers)

def single_cut(L: float, W: float, b: float, x_cut: float | None = None, **kw: Any) -> Mesh:
    return mesh_pattern(L, W, single_cut_pattern(L, W, b, x_cut), **kw)


def two_cuts(L: float, W: float, b: float, ls: float, **kw: Any) -> Mesh:
    return mesh_pattern(L, W, two_cuts_pattern(L, W, b, ls), **kw)


def pitch_cell(L: float, W: float, Lc: float, Lpar: float, Lr: float, gap: float, **kw: Any) -> Mesh:
    return mesh_pattern(L, W, pitch_pattern(L, W, Lc, Lpar, Lr, gap), **kw)


def roll_cell(L: float = 150.0, W: float = 60.0, kerf: float = 1.0, **kw: Any) -> Mesh:
    return mesh_pattern(L, W, roll_pattern(L, W, kerf), **kw)


def yaw_cell(L: float = 150.0, W: float = 60.0, window: tuple[float, float] = (45.0, 105.0),
             a: float = 14.0, b: float = 8.0, tilt: float = 20.0, arm_width: float = 3.0,
             strip: float = 4.0, crack: float = 0.5, **kw: Any) -> Mesh:
    return mesh_pattern(L, W, yaw_pattern(L, W, window, a, b, tilt, arm_width, strip, crack), **kw)


def alternating(L: float, W: float, b: float, g: float, d: float, n_rows: int, **kw: Any) -> Mesh:
    return mesh_pattern(L, W, alternating_pattern(L, W, b, g, d, n_rows), **kw)


def rotating_squares(L: float, W: float, a: float, t: float, nx: int, **kw: Any) -> Mesh:
    return mesh_pattern(L, W, rotating_squares_pattern(L, W, a, t, nx), **kw)


def pitch_chain(L: float, W: float, n: int, cell: float, Lc: float, Lpar: float, Lr: float, gap: float,
                **kw: Any) -> Mesh:
    return mesh_pattern(L, W, pitch_chain_pattern(L, W, n, cell, Lc, Lpar, Lr, gap), **kw)
