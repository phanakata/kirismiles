"""Import of laser-cut files (SVG, DXF) as KiriSMILES strings.

A cut file is a set of curves in a plane. The importer turns it into a design in four steps:

1. read     every curve becomes a polyline in mm with y pointing up. Straight entities are
            kept exactly; curves (arcs, circles, ellipses, Bezier segments, polyline bulges)
            are replaced by polylines whose distance from the true curve is at most `tol` mm.
2. chain    loose segments that meet end to end (a DXF of LINEs, for instance) are joined
            into polylines; a polyline that ends where it starts is closed.
3. classify the sheet outline is the closed axis-aligned rectangle that contains everything
            (or the sheet is given as L x W). Open polylines are cuts. Closed polylines
            inside the sheet are holes or solids by nesting: a closed curve inside an even
            number of others is a hole, inside an odd number a solid added back (an island).
4. write    the pieces are written as a free-form string, C(...) for cuts, X(...) for holes
            and +(...) for solids, with x from the sheet centre and y from the centre line,
            rounded to `digits` decimals, and put in canonical form.

Only the outline of what is cut is needed; layers, colours and line widths are ignored.
No dependencies beyond numpy.
"""
from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from kirigami import kirismiles as ks
from kirigami._types import Pattern, Polyline

PathLike = str | os.PathLike[str]


def _positive(**values: float) -> None:
    """Raise ValueError for every value that is not > 0."""
    bad = [f"{k} = {v}" for k, v in values.items() if not v > 0]
    if bad:
        raise ValueError("must be positive: " + ", ".join(bad))

UNITS_MM = {"mm": 1.0, "cm": 10.0, "in": 25.4, "pt": 25.4 / 72, "pc": 25.4 / 6,
            "px": 25.4 / 96, "": 25.4 / 96}
DXF_UNITS_MM = {0: 1.0, 1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0, 8: 25.4e-6, 9: 0.0254,
                10: 914.4, 13: 1e-3, 14: 100.0}


# ----------------------------------------------------------------- curves

def _arc_steps(r: float, sweep: float, tol: float) -> int:
    """Number of chords so that each lies within tol of an arc of radius r and angle sweep."""
    if r <= tol:
        return max(int(math.ceil(abs(sweep) / (math.pi / 2))), 1)
    return max(int(math.ceil(abs(sweep) / (2 * math.acos(1 - tol / r)))), 1)


def _arc(cx: float, cy: float, rx: float, ry: float, phi: float, t0: float, dt: float,
         tol: float) -> np.ndarray:
    """Points of an elliptical arc (centre, radii, rotation phi, start angle t0, sweep dt)."""
    n = _arc_steps(max(rx, ry), dt, tol)
    t = t0 + dt * np.linspace(0, 1, n + 1)
    c, s = math.cos(phi), math.sin(phi)
    x, y = rx * np.cos(t), ry * np.sin(t)
    return np.column_stack([cx + c * x - s * y, cy + s * x + c * y])


def _bezier(ctrl: Sequence[Any], tol: float) -> np.ndarray:
    """Points of a quadratic or cubic Bezier segment, chords within about tol of the curve."""
    P = np.asarray(ctrl, float)
    span = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    n = max(int(math.ceil(math.sqrt(span / max(tol, 1e-9)))), 2)
    t = np.linspace(0, 1, n + 1)[:, None]
    if len(P) == 3:
        return (1 - t) ** 2 * P[0] + 2 * (1 - t) * t * P[1] + t ** 2 * P[2]
    return ((1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2]
            + t ** 3 * P[3])


def _svg_arc(p0: np.ndarray, rx: float, ry: float, phi_deg: float, large: int, sweep: int, p1: np.ndarray,
             tol: float) -> np.ndarray:
    """SVG elliptical arc from p0 to p1 (endpoint parametrisation, SVG 1.1 appendix F.6)."""
    if rx == 0 or ry == 0 or np.allclose(p0, p1):
        return np.array([p0, p1])
    phi = math.radians(phi_deg)
    c, s = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1, y1 = c * dx + s * dy, -s * dx + c * dy
    rx, ry = abs(rx), abs(ry)
    lam = x1 ** 2 / rx ** 2 + y1 ** 2 / ry ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1 ** 2 - ry ** 2 * x1 ** 2
    k = math.sqrt(max(num, 0) / (rx ** 2 * y1 ** 2 + ry ** 2 * x1 ** 2))
    if large == sweep:
        k = -k
    cx1, cy1 = k * rx * y1 / ry, -k * ry * x1 / rx
    cx = c * cx1 - s * cy1 + (p0[0] + p1[0]) / 2
    cy = s * cx1 + c * cy1 + (p0[1] + p1[1]) / 2
    ang = lambda ux, uy, vx, vy: math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)  # noqa: E731
    t0 = ang(1, 0, (x1 - cx1) / rx, (y1 - cy1) / ry)
    dt = ang((x1 - cx1) / rx, (y1 - cy1) / ry, (-x1 - cx1) / rx, (-y1 - cy1) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    pts = _arc(cx, cy, rx, ry, phi, t0, dt, tol)
    pts[0], pts[-1] = p0, p1
    return pts


# -------------------------------------------------------------------- SVG

def _length_mm(v: str | None) -> float | None:
    m = re.fullmatch(r"\s*([-+0-9.eE]+)\s*([a-z%]*)\s*", v or "")
    if not m or m.group(2) == "%":
        return None
    return float(m.group(1)) * UNITS_MM.get(m.group(2), 25.4 / 96)


def _transform(text: str | None) -> np.ndarray:
    """3 x 3 matrix of an SVG transform attribute."""
    M = np.eye(3)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", text or ""):
        a = [float(v) for v in re.split(r"[\s,]+", args.strip()) if v]
        if name == "matrix":
            T = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif name == "translate":
            T = np.array([[1, 0, a[0]], [0, 1, a[1] if len(a) > 1 else 0], [0, 0, 1]])
        elif name == "scale":
            T = np.diag([a[0], a[1] if len(a) > 1 else a[0], 1])
        elif name == "rotate":
            r = math.radians(a[0])
            T = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
            if len(a) == 3:
                T = (np.array([[1, 0, a[1]], [0, 1, a[2]], [0, 0, 1]]) @ T
                     @ np.array([[1, 0, -a[1]], [0, 1, -a[2]], [0, 0, 1]]))
        elif name == "skewX":
            T = np.array([[1, math.tan(math.radians(a[0])), 0], [0, 1, 0], [0, 0, 1]])
        elif name == "skewY":
            T = np.array([[1, 0, 0], [math.tan(math.radians(a[0])), 1, 0], [0, 0, 1]])
        else:
            continue
        M = M @ T
    return M


def _path(d: str, tol: float) -> list[np.ndarray]:
    """Polylines of an SVG path's d attribute (user units)."""
    toks = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", d)
    out: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    i = 0
    p = start = np.zeros(2)
    cmd: str | None = None
    last_ctrl = np.zeros(2)
    prev: str | None = None

    def nums(k: int) -> list[float]:
        nonlocal i
        v = [float(t) for t in toks[i:i + k]]
        i += k
        return v

    while i < len(toks):
        if re.fullmatch(r"[A-Za-z]", toks[i]):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                if cur:
                    cur.append(start.copy())
                    out.append(np.array(cur))
                cur, p, prev = [], start.copy(), "Z"
                continue
        if cmd is None:
            raise ValueError(f"an SVG path must start with a command, not {toks[i]!r}")
        rel = cmd.islower()
        base = p if rel else np.zeros(2)
        C = cmd.upper()
        if C == "M":
            if cur:
                out.append(np.array(cur))
            x, y = nums(2)
            p = start = base + [x, y]
            cur = [p.copy()]
            cmd = "l" if rel else "L"                   # further pairs are line-tos
        elif C in "LHV":
            if C == "L":
                x, y = nums(2)
                q = base + [x, y]
            elif C == "H":
                (x,) = nums(1)
                q = np.array([p[0] + x if rel else x, p[1]])
            else:
                (y,) = nums(1)
                q = np.array([p[0], p[1] + y if rel else y])
            cur.append(q)
            p = q
        elif C in "CS":
            if C == "C":
                v = nums(6)
                c1, c2, q = base + v[0:2], base + v[2:4], base + v[4:6]
            else:
                v = nums(4)
                c1 = 2 * p - last_ctrl if prev and prev in "CS" else p.copy()
                c2, q = base + v[0:2], base + v[2:4]
            cur += list(_bezier([p, c1, c2, q], tol)[1:])
            last_ctrl, p = c2, q
        elif C in "QT":
            if C == "Q":
                v = nums(4)
                c1, q = base + v[0:2], base + v[2:4]
            else:
                v = nums(2)
                c1 = 2 * p - last_ctrl if prev and prev in "QT" else p.copy()
                q = base + v[0:2]
            cur += list(_bezier([p, c1, q], tol)[1:])
            last_ctrl, p = c1, q
        elif C == "A":
            rx, ry, phi, large, sweep, x, y = nums(7)
            q = base + [x, y]
            cur += list(_svg_arc(p, rx, ry, phi, int(large), int(sweep), q, tol)[1:])
            p = q
        prev = C
    if cur and len(cur) > 1:
        out.append(np.array(cur))
    return [o for o in out if len(o) > 1]


def read_svg(path: PathLike, tol: float = 0.01) -> list[np.ndarray]:
    """Polylines (mm, y up) of the shapes in an SVG file; curves within tol mm (> 0)."""
    _positive(tol=tol)
    root = ET.parse(path).getroot()
    ns = lambda tag: tag.split("}")[-1]                 # noqa: E731
    vb = [float(v) for v in re.split(r"[\s,]+", root.get("viewBox", "").strip()) if v]
    w_mm, h_mm = _length_mm(root.get("width")), _length_mm(root.get("height"))
    if len(vb) == 4 and w_mm:
        sx = w_mm / vb[2]
        sy = (h_mm / vb[3]) if h_mm else sx
        S = np.array([[sx, 0, -vb[0] * sx], [0, sy, -vb[1] * sy], [0, 0, 1]])
    else:
        S = np.diag([25.4 / 96, 25.4 / 96, 1.0])          # user units are CSS px
    scale = math.sqrt(abs(np.linalg.det(S[:2, :2])))
    # page height in mm: y is flipped about it, so (0, 0) is the lower-left corner of the page
    page_h = h_mm if h_mm else (vb[3] * S[1, 1] if len(vb) == 4 else 0.0)
    out: list[np.ndarray] = []

    def walk(el: ET.Element, M: np.ndarray) -> None:
        M = M @ _transform(el.get("transform"))
        tag = ns(el.tag)
        t = tol / max(scale * math.sqrt(abs(np.linalg.det(M[:2, :2]))), 1e-12)   # in user units
        def f(k: str, d: float = 0.0) -> float:          # a numeric attribute, d if absent
            v = el.get(k)
            return d if v is None else float(v)
        polys: list[np.ndarray] = []
        if tag == "line":
            polys = [np.array([[f("x1"), f("y1")], [f("x2"), f("y2")]])]
        elif tag in ("polyline", "polygon"):
            v = [float(a) for a in re.split(r"[\s,]+", el.get("points", "").strip()) if a]
            P = np.array(v[:len(v) // 2 * 2]).reshape(-1, 2)
            if tag == "polygon" and len(P):
                P = np.vstack([P, P[:1]])
            polys = [P]
        elif tag == "rect":
            x, y, w, h = f("x"), f("y"), f("width"), f("height")
            polys = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]])]
        elif tag in ("circle", "ellipse"):
            rx = f("r") if tag == "circle" else f("rx")
            ry = f("r") if tag == "circle" else f("ry")
            P = _arc(f("cx"), f("cy"), rx, ry, 0.0, 0.0, 2 * math.pi, t)
            polys = [P]
        elif tag == "path":
            polys = _path(el.get("d", ""), t)
        for P in polys:
            if len(P) > 1:
                Q = (S @ M @ np.column_stack([P, np.ones(len(P))]).T).T[:, :2]
                out.append(np.column_stack([Q[:, 0], page_h - Q[:, 1]]))   # SVG y points down
        for ch in el:
            if ns(ch.tag) not in ("defs", "clipPath", "mask", "symbol", "metadata", "title"):
                walk(ch, M)

    walk(root, np.eye(3))
    return out


# -------------------------------------------------------------------- DXF

def _dxf_pairs(path: PathLike) -> list[tuple[int, str]]:
    """(group code, value) pairs of an ASCII DXF file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh]
    return [(int(lines[k]), lines[k + 1]) for k in range(0, len(lines) - 1, 2)
            if lines[k].lstrip("-").isdigit()]


def _bulge_arc(v0: Sequence[float], v1: Sequence[float], b: float, tol: float) -> np.ndarray:
    """Arc between two polyline vertices with DXF bulge b = tan(sweep / 4)."""
    p0, p1 = np.asarray(v0, float), np.asarray(v1, float)
    if abs(b) < 1e-12:
        return np.array([p0, p1])
    sweep = 4 * math.atan(b)                            # signed, counterclockwise > 0
    chord = np.linalg.norm(p1 - p0)
    r = chord / (2 * math.sin(abs(sweep) / 2))
    mid = (p0 + p1) / 2
    nrm = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / chord    # left of the chord
    c = mid + nrm * (chord / 2) / math.tan(sweep / 2)
    t0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
    pts = _arc(c[0], c[1], r, r, 0.0, t0, sweep, tol)
    pts[0], pts[-1] = p0, p1
    return pts


def read_dxf(path: PathLike, tol: float = 0.01) -> list[np.ndarray]:
    """Polylines (mm) of LINE, LWPOLYLINE, POLYLINE, CIRCLE, ARC and ELLIPSE entities of an
    ASCII DXF file, curves within tol mm (> 0). SPLINEs are approximated by their fit or
    control points."""
    _positive(tol=tol)
    pairs = _dxf_pairs(path)
    unit = 1.0
    for k, (c, v) in enumerate(pairs):
        if c == 9 and v == "$INSUNITS" and k + 1 < len(pairs):
            unit = DXF_UNITS_MM.get(int(pairs[k + 1][1]), 1.0)
    try:
        s = next(k for k, (c, v) in enumerate(pairs) if c == 2 and v == "ENTITIES")
    except StopIteration:
        return []
    ents: list[tuple[str, list[tuple[int, str]]]] = []
    cur: tuple[str, list[tuple[int, str]]] | None = None
    for c, v in pairs[s + 1:]:
        if c == 0:
            if cur:
                ents.append(cur)
            if v in ("ENDSEC", "EOF"):
                break
            cur = (v, [])
        elif cur is not None:
            cur[1].append((c, v))
    out: list[np.ndarray] = []
    poly: dict[str, Any] | None = None
    for typ, g in ents:
        d: dict[int, list[str]] = {}
        for c, v in g:
            d.setdefault(c, []).append(v)
        f = lambda c, k=0, dflt=0.0: float(d[c][k]) if c in d and len(d[c]) > k else dflt  # noqa: E731
        if typ == "LINE":
            out.append(np.array([[f(10), f(20)], [f(11), f(21)]]))
        elif typ == "LWPOLYLINE":
            pts: list[list[float | None]] = []
            bulges: list[float] = []
            closed = int(f(70)) & 1
            for c, v in g:
                if c == 10:
                    pts.append([float(v), None])
                    bulges.append(0.0)
                elif c == 20 and pts:
                    pts[-1][1] = float(v)
                elif c == 42 and bulges:
                    bulges[-1] = float(v)
            out.append(_with_bulges(pts, bulges, closed, tol))
        elif typ == "POLYLINE":
            poly = dict(pts=[], bulges=[], closed=int(f(70)) & 1)
        elif typ == "VERTEX" and poly is not None:
            poly["pts"].append([f(10), f(20)])
            poly["bulges"].append(f(42))
        elif typ == "SEQEND" and poly is not None:
            out.append(_with_bulges(poly["pts"], poly["bulges"], poly["closed"], tol))
            poly = None
        elif typ == "CIRCLE":
            out.append(_arc(f(10), f(20), f(40), f(40), 0.0, 0.0, 2 * math.pi, tol))
        elif typ == "ARC":
            a0, a1 = math.radians(f(50)), math.radians(f(51))
            out.append(_arc(f(10), f(20), f(40), f(40), 0.0, a0, (a1 - a0) % (2 * math.pi) or
                            2 * math.pi, tol))
        elif typ == "ELLIPSE":
            mx, my, ratio = f(11), f(21), f(40, dflt=1.0)
            rx = math.hypot(mx, my)
            t0, t1 = f(41), f(42, dflt=2 * math.pi)
            out.append(_arc(f(10), f(20), rx, rx * ratio, math.atan2(my, mx), t0,
                            (t1 - t0) % (2 * math.pi) or 2 * math.pi, tol))
        elif typ == "SPLINE":
            key = (11, 21) if 11 in d else (10, 20)
            ctrl = np.column_stack([[float(v) for v in d.get(key[0], [])],
                                    [float(v) for v in d.get(key[1], [])]])
            if len(ctrl) > 1:
                out.append(ctrl)
    return [P * unit for P in out if len(P) > 1]


def _with_bulges(vertices: Sequence[Sequence[float | None]], bulges: list[float], closed: int,
                 tol: float) -> np.ndarray:
    """A DXF polyline's points, with its bulged segments replaced by arcs."""
    pts: list[Any] = [p for p in vertices if p[1] is not None]
    if not pts:
        return np.zeros((0, 2))
    if closed:
        pts, bulges = pts + [pts[0]], bulges + [0.0]
    seg = [np.array([pts[0]], float)]
    for k in range(len(pts) - 1):
        seg.append(_bulge_arc(pts[k], pts[k + 1], bulges[k], tol)[1:])
    return np.vstack(seg)


# ---------------------------------------------------------- to a pattern

def chain(polys: Iterable[np.ndarray], tol: float = 1e-4) -> list[np.ndarray]:
    """Join polylines that meet end to end into longer ones (only where exactly two ends
    meet, so that T-junctions stay separate pieces). tol: how close two ends must be (> 0)."""
    _positive(tol=tol)
    polys = [np.asarray(p, float) for p in polys if len(p) > 1]
    key = lambda q: (round(q[0] / tol), round(q[1] / tol))    # noqa: E731
    changed = True
    while changed:
        changed = False
        ends: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for i, p in enumerate(polys):
            if np.linalg.norm(p[0] - p[-1]) <= tol and len(p) > 2:
                continue                                 # already closed
            for j, q in ((0, p[0]), (-1, p[-1])):
                ends.setdefault(key(q), []).append((i, j))
        for _, hits in ends.items():
            if len(hits) != 2 or hits[0][0] == hits[1][0]:
                if len(hits) == 2 and hits[0][0] == hits[1][0]:      # both ends of one: close
                    i = hits[0][0]
                    polys[i] = np.vstack([polys[i][:-1], polys[i][:1]])
                    changed = True
                    break
                continue
            (i, a), (j, b) = hits
            P = polys[i] if a == -1 else polys[i][::-1]
            Q = polys[j] if b == 0 else polys[j][::-1]
            polys[i] = np.vstack([P, Q[1:]])
            del polys[j]
            changed = True
            break
    return polys


def _closed(p: np.ndarray, tol: float = 1e-4) -> bool:
    return bool(len(p) > 3 and np.linalg.norm(p[0] - p[-1]) <= tol)


def _inside(pt: Sequence[float], poly: np.ndarray) -> bool:
    """Point-in-polygon (even-odd rule). The point is nudged by about 1e-7 mm so that the
    test ray never passes exactly through a vertex (curves drawn from the same centre share
    vertex heights)."""
    x, y = pt[0] + 1.37e-7, pt[1] + 2.91e-7
    P = np.asarray(poly)
    xi, yi, xj, yj = P[:-1, 0], P[:-1, 1], P[1:, 0], P[1:, 1]
    cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
    return bool(np.count_nonzero(cross) % 2)


def _is_outline(p: np.ndarray, tol: float = 1e-4) -> bool:
    """An axis-aligned rectangle (after dropping collinear points)."""
    q = ks._straighten([tuple(np.round(v, 9)) for v in p[:-1]], True)
    if len(q) != 4:
        return False
    xs, ys = sorted({round(v[0], 6) for v in q}), sorted({round(v[1], 6) for v in q})
    return len(xs) == 2 and len(ys) == 2


def to_pattern(polys: Iterable[np.ndarray], L: float | None = None, W: float | None = None,
               origin: Sequence[float] | None = None,
               chain_tol: float = 1e-4) -> tuple[Pattern, float, float, dict[str, Any]]:
    """Sort polylines into the cuts, holes and solids of an L x W sheet.

    The sheet is the largest closed axis-aligned rectangle covering nearly the whole
    drawing (pieces may cross its edge). Without one, give L and W, and origin, the
    lower-left corner of the sheet in the drawing (default: the drawing's origin, which for
    an SVG is the lower-left corner of the page). Returns (pattern, L, W, report)."""
    if (L is None) != (W is None):
        raise ValueError("give both L and W, or neither (the sheet is then the outline in the file)")
    if L is not None and W is not None:
        _positive(L=L, W=W)
    curves = chain(polys, chain_tol)
    closed = [p for p in curves if _closed(p, chain_tol)]
    report: dict[str, Any] = {"curves": len(curves)}
    outline: np.ndarray | None = None
    if L is None or W is None:
        rects = sorted([p for p in closed if _is_outline(p, chain_tol)],
                       key=lambda p: -np.ptp(p[:, 0]) * np.ptp(p[:, 1]))
        if rects:
            # the largest rectangle is the sheet if it covers nearly the whole drawing; pieces
            # may cross its edge (slots that open an edge start outside the sheet)
            r = rects[0]
            allp = np.vstack(curves)
            if np.ptp(r[:, 0]) * np.ptp(r[:, 1]) >= 0.9 * np.ptp(allp[:, 0]) * np.ptp(allp[:, 1]):
                outline = r
        if outline is None:
            raise ValueError("no sheet outline (a closed rectangle around everything) was "
                             "found; give the sheet size L and W")
        o = outline.min(0)
        L, W = (outline.max(0) - o).tolist()
    else:
        o = np.asarray((0.0, 0.0) if origin is None else origin, float)   # default: the drawing's origin
    assert L is not None and W is not None
    pat: Pattern = {"cuts": [], "holes": [], "solids": []}
    closed_ids = [i for i, p in enumerate(curves) if _closed(p, chain_tol) and p is not outline]
    box = {i: (curves[i].min(0), curves[i].max(0)) for i in closed_ids}
    for i, p in enumerate(curves):
        if p is outline:
            continue
        q = p - o
        if i not in closed_ids:
            pat["cuts"].append([(float(v[0]), float(v[1])) for v in q])
            continue
        # nesting depth: the closed curves that contain this one entirely (every vertex);
        # curves that merely overlap it, such as crossing slots, do not count
        lo, hi = box[i]
        depth = sum(1 for j in closed_ids if j != i
                    and np.all(box[j][0] <= lo) and np.all(box[j][1] >= hi)
                    and all(_inside(v, curves[j]) for v in p[:-1]))
        pat["solids" if depth % 2 else "holes"].append([(float(v[0]), float(v[1])) for v in q[:-1]])
    report.update(L=L, W=W, cuts=len(pat["cuts"]), holes=len(pat["holes"]),
                  solids=len(pat["solids"]), outline_found=outline is not None)
    return pat, float(L), float(W), report


def to_string(pat: Mapping[str, Sequence[Polyline]], L: float, W: float, digits: int = 3) -> str:
    """Free-form KiriSMILES string of a pattern (x from the sheet centre, y from the centre
    line), coordinates rounded to `digits` decimals (0 to 6), in canonical form."""
    if not 0 <= digits <= 6:
        raise ValueError(f"digits must be between 0 and 6, not {digits}")
    fmt = lambda v: ks._n(round(float(v), digits) + 0.0)      # noqa: E731

    def pts(poly: Polyline, closed: bool) -> str:
        q = ks._straighten([(round(x - L / 2, digits), round(y - W / 2, digits))
                            for x, y in poly], closed)
        return "(" + ";".join(f"{fmt(x)},{fmt(y)}" for x, y in q) + ")"

    items = [f"C{pts(c, False)}" for c in pat["cuts"]]
    items += [f"X{pts(h, True)}" for h in pat["holes"]]
    items += [f"+{pts(s, True)}" for s in pat["solids"]]
    if not items:
        raise ValueError("the file has no cuts, holes or solids inside the sheet")
    return ks.canonical(";".join(items))


def import_file(path: PathLike, L: float | None = None, W: float | None = None, tol: float = 0.01,
                digits: int = 3) -> dict[str, Any]:
    """String of a laser-cut file (.svg or .dxf). Returns a dict with the string, the sheet
    size, the pattern and a report, including the largest distance between the file's
    curves and the string's geometry (the price of rounding and of flattening curves)."""
    name = os.fspath(path)
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "svg":
        polys = read_svg(name, tol)
    elif ext == "dxf":
        polys = read_dxf(name, tol)
    else:
        raise ValueError(f"unknown file type .{ext}; use .svg or .dxf")
    if not polys:
        raise ValueError("no curves found in the file")
    pat, Lf, Wf, report = to_pattern(polys, L, W)
    L, W = round(Lf, digits) + 0.0, round(Wf, digits) + 0.0
    report.update(L=L, W=W)
    code = to_string(pat, L, W, digits)
    back = ks.pattern(code, L, W)
    report["max_deviation_mm"] = deviation(pat, back)
    report["curve_tolerance_mm"] = tol
    return dict(code=code, L=L, W=W, pattern=back, report=report)


def _segments(pat: Mapping[str, Sequence[Polyline]]) -> np.ndarray:
    """All straight segments of a pattern's cut lines and hole and solid outlines, (n, 2, 2)."""
    out: list[np.ndarray] = []
    for kind in ("cuts", "holes", "solids"):
        for g in pat.get(kind, ()):
            P = np.asarray(g, float)
            if kind != "cuts":
                P = np.vstack([P, P[:1]])
            out.append(np.stack([P[:-1], P[1:]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2, 2))


def deviation(a: Mapping[str, Sequence[Polyline]], b: Mapping[str, Sequence[Polyline]],
              step: float = 0.02, cap: float = 1.0) -> float:
    """Largest distance (mm) between two patterns' cut lines and outlines (symmetric
    Hausdorff distance). Points are taken every `step` mm along one pattern and measured to
    the nearest *segment* of the other, so identical geometries give zero whatever the
    sampling; distances above `cap` are reported as `cap`. step and cap are in mm (> 0)."""
    _positive(step=step, cap=cap)
    Sa, Sb = _segments(a), _segments(b)
    if not len(Sa) or not len(Sb):
        return float("inf") if len(Sa) != len(Sb) else 0.0
    return max(_one_way(Sa, Sb, step, cap), _one_way(Sb, Sa, step, cap))


def _one_way(SA: np.ndarray, SB: np.ndarray, step: float, cell: float) -> float:
    """Largest distance from points along the segments SA to the nearest segment of SB.
    SB is binned by bounding box in cells of `cell` mm; each point searches the segments
    binned in its own cell, so the result is exact up to `cell` and capped there."""
    from collections import defaultdict
    pts: list[np.ndarray] = []
    for p, q in SA:
        n = max(int(np.ceil(np.linalg.norm(q - p) / step)), 1)
        pts.append(p + np.linspace(0, 1, n + 1)[:, None] * (q - p))
    A = np.vstack(pts)
    bins: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    lo = np.floor((SB.min(1) - cell) / cell).astype(int)
    hi = np.floor((SB.max(1) + cell) / cell).astype(int)
    for i, (l, h) in enumerate(zip(lo, hi)):
        for x in range(l[0], h[0] + 1):
            for y in range(l[1], h[1] + 1):
                bins[(x, y)].append(i)
    keys = np.floor(A / cell).astype(int)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    worst = 0.0
    k0 = 0
    while k0 < len(order):                              # points of one cell at a time
        k1 = k0
        key = tuple(keys[order[k0]])
        while k1 < len(order) and tuple(keys[order[k1]]) == key:
            k1 += 1
        P = A[order[k0:k1]]
        idx = bins.get(key)
        if not idx:
            worst = cell
        else:
            s = SB[idx]
            a, d = s[:, 0], s[:, 1] - s[:, 0]
            t = np.clip(((P[:, None, :] - a[None]) * d[None]).sum(-1)
                        / np.maximum((d * d).sum(-1), 1e-30)[None], 0, 1)
            dist = np.linalg.norm(P[:, None, :] - (a[None] + t[..., None] * d[None]), axis=-1)
            worst = max(worst, min(float(dist.min(1).max()), cell))
        k0 = k1
    return worst
