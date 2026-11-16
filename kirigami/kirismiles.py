"""KiriSMILES: Kirigami Simplified Metamaterial-Input Line-Entry System.

One line of text for one cut pattern, in the spirit of SMILES for molecules.
The full description, with figures, is KIRISMILES.md. Two forms share the notation:

  lattice     P5x45S1/2O0[V42@0,0]N18E3      a motif repeated on a lattice
              (kirigami/grammar.py documents the lattice part)
  free form   (C(2,16;47.5,16;47.5,-16;2,-16)){mx};(K18.75@0){my}
              individual pieces placed in the sheet

A lattice motif may hold any free-form piece besides its slits, e.g.
P10x20S1/2O10[(C(-9,0;9,0);C(0,1;0,19))k0.5]N9E2. Slits keep their positions as
fractions of the cell (V42@0,0); every other piece is in mm from the lattice
point. One copy goes at each lattice point of columns 0 .. n-1; cuts are clipped
at the free edges like slits. Cracks (K) and edge coordinates (b, t) are not
allowed in a motif, since a lattice point does not know where the edges are.

Free form. Items are joined by ';'. Lengths are in mm. x is measured from the
middle of the sheet along the pulled direction. A y coordinate is measured from
the centre line, or from an edge when it starts with b or t: 'b+2.9' is 2.9 mm
above the bottom edge, 't-3.5' is 3.5 mm below the top edge, and 'b' / 't' are
the edges themselves.

  V80@x,y          straight slit centred at (x, y); H, D, A or R<deg>: as in grammar.py,
                   e.g. R30:24@0,0 is a 24 mm slit at 30 deg
  C(x,y;x,y;...)   cut along a polyline
  K<len>@x[b|t]    crack running <len> mm in from the bottom (b, default) or top (t) edge
  X(x,y;...)       hole: the polygon is removed
  +(x,y;...)       solid: the polygon is added back after the holes
  +E(a,b,deg)[@x,y]          ellipse added back, semi-axes a, b, rotated by deg
  +B(x,y;x,y;x,y;x,y)w<w>    cubic Bezier strip of width w added back
  (...)            group; after the ')' come symmetries, applied in order:
                     {mx} mirror x -> -x   {my} mirror y -> -y   {i} inversion
                     {t<p>x<n>} n copies, p mm apart along x, centred
  k<w>             after a slit, cut, crack or group: the cuts become slots of width w
                   (holes). A slot that starts at an edge starts 1 mm outside it.

Sheet size, thickness and material are context, kept outside the string.

Identity works on two levels, as in chemistry:
  encode(parse(s))     canonical string: numbers normalised, items of a group sorted,
                       duplicates dropped
  geometry_key(...)    hash of the cut geometry itself, the same for any two strings
                       that cut the same sheet, including mirror images; the
                       database key (the role InChIKey plays for molecules)

The paper's cells written this way (tests/test_kirismiles.py checks each one
against its generator in kirigami/geometry.py):
  single cut   V80@0,0                                                     (L 182, W 100)
  pitch        (C(2,16;47.5,16;47.5,-16;2,-16)){mx};(K18.75@0){my}         (L 180, W 60)
  roll         ((K23.25@49){my};C(41.6,b+2.9;41.6,t-3.5);(K7.5@34.4){my};
               C(41.6,0;5,0);X(1.25,-2.9;5.5,0;1.25,2.9)){mx}k1           (L 150, W 60)
  yaw          X(-30,b+4;30,b+4;30,t-4;-30,t-4);+E(14,8,20);
               (+B(-24,t-2;-24,2;-16,14;-2,4)w3;K30@-31.5t){i}            (L 150, W 60)
"""
import hashlib
import re
from dataclasses import dataclass, replace

import numpy as np

from kirigami import geometry as geo
from kirigami import grammar

EXTEND = 1.0          # mm: a slot starting at an edge starts this far outside it
N_ELLIPSE = 120       # samples, as in geometry.yaw_pattern
SYMS = ("mx", "my", "i")


def _n(x):
    return f"{round(float(x), 3) + 0.0:g}"


@dataclass(frozen=True)
class Y:
    """A y coordinate: offset v from the centre line ('c') or an edge ('b', 't')."""
    ref: str
    v: float

    def at(self, W):                       # from the centre line
        return {"c": 0.0, "b": -W / 2, "t": W / 2}[self.ref] + self.v

    def flip(self):
        return Y({"c": "c", "b": "t", "t": "b"}[self.ref], -self.v)

    def __str__(self):
        if self.ref == "c":
            return _n(self.v)
        return self.ref + ("" if self.v == 0 else ("+" if self.v > 0 else "-") + _n(abs(self.v)))


@dataclass(frozen=True)
class Slit:
    angle: int
    length: float
    x: float
    y: Y
    kerf: float = None


@dataclass(frozen=True)
class Cut:
    pts: tuple
    kerf: float = None


@dataclass(frozen=True)
class Crack:
    length: float
    x: float
    edge: str
    kerf: float = None


@dataclass(frozen=True)
class Region:                              # hole (X) or solid (+) polygon
    solid: bool
    pts: tuple


@dataclass(frozen=True)
class Ellipse:
    a: float
    b: float
    angle: float
    x: float = 0.0
    y: Y = Y("c", 0.0)


@dataclass(frozen=True)
class Strip:                               # cubic Bezier of constant width
    pts: tuple
    width: float


@dataclass(frozen=True)
class Group:
    items: tuple
    syms: tuple = ()
    kerf: float = None


@dataclass(frozen=True)
class FreeForm:
    items: tuple


@dataclass(frozen=True)
class Lattice:
    """A lattice (grammar.Design: cell, shift, offset, slits, columns, trim)
    whose motif also holds free-form pieces, in mm from the lattice point."""
    base: grammar.Design
    extra: tuple


# ------------------------------------------------------------------ parsing

_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?")


class _Parser:
    def __init__(self, text):
        self.s, self.i = re.sub(r"\s+", "", text), 0

    def fail(self, msg):
        raise ValueError(f"{msg} at position {self.i} of {self.s!r}")

    def peek(self, k=1):
        return self.s[self.i:self.i + k]

    def take(self, lit):
        if not self.s.startswith(lit, self.i):
            return False
        self.i += len(lit)
        return True

    def eat(self, lit):
        if not self.take(lit):
            self.fail(f"expected {lit!r}")

    def num(self):
        m = _NUM.match(self.s, self.i)
        if not m:
            self.fail("expected a number")
        self.i = m.end()
        return float(m[0])

    def ycoord(self):
        if self.peek() in ("b", "t"):
            ref = self.s[self.i]
            self.i += 1
            return Y(ref, self.num() if self.peek() in ("+", "-") else 0.0)
        return Y("c", self.num())

    def point(self):
        x = self.num()
        self.eat(",")
        return (x, self.ycoord())

    def points(self):
        self.eat("(")
        pts = [self.point()]
        while self.take(";"):
            pts.append(self.point())
        self.eat(")")
        return tuple(pts)

    def kerf(self):
        if not self.take("k"):
            return None
        k = self.num()
        if k <= 0:
            self.fail("a kerf needs a width above zero")
        return k

    def seq(self):
        items = [self.item()]
        while self.take(";"):
            items.append(self.item())
        return tuple(items)

    def item(self):
        if self.take("("):
            items = self.seq()
            self.eat(")")
            syms = []
            while self.take("{"):
                j = self.s.find("}", self.i)
                tok = self.s[self.i:j] if j > 0 else ""
                if tok not in SYMS and not re.fullmatch(r"t\d+(\.\d+)?x\d+", tok):
                    self.fail(f"unknown symmetry {{{tok}}}")
                if tok not in SYMS and (float(tok[1:].split("x")[0]) <= 0
                                        or int(tok.split("x")[1]) < 1):
                    self.fail("a repetition {t<p>x<n>} needs p > 0 and n >= 1")
                syms.append(tok)
                self.i = j + 1
            return Group(items, tuple(syms), self.kerf())
        if self.take("C"):
            pts = self.points()
            if len(pts) < 2:
                self.fail("a cut needs at least two points")
            return Cut(pts, self.kerf())
        if self.take("K"):
            length = self.num()
            if length <= 0:
                self.fail("a crack needs a length above zero")
            self.eat("@")
            x = self.num()
            edge = "t" if self.take("t") else "b"
            if edge == "b":
                self.take("b")                     # optional: bottom is the default
            return Crack(length, x, edge, self.kerf())
        if self.take("X"):
            return Region(False, self.points())
        if self.take("+"):
            if self.take("E("):
                a = self.num()
                self.eat(",")
                b = self.num()
                self.eat(",")
                ang = self.num()
                self.eat(")")
                x, y = self.point() if self.take("@") else (0.0, Y("c", 0.0))
                return Ellipse(a, b, ang, x, y)
            if self.take("B"):
                pts = self.points()
                if len(pts) != 4:
                    self.fail("a Bezier strip needs four control points")
                self.eat("w")
                return Strip(pts, self.num())
            return Region(True, self.points())
        m = re.compile(grammar.DIR).match(self.s, self.i)
        if m:
            self.i = m.end()
            ang = grammar.dir_angle(m[1])
            length = self.num()
            if length <= 0:
                self.fail("a slit needs a length above zero")
            self.eat("@")
            x, y = self.point()
            return Slit(ang % 180, length, x, y, self.kerf())
        if self.peek() == "R":
            self.fail("an angled slit is written R<deg>:<length>, e.g. R30:24@0,0")
        self.fail("expected an item")


def _split_top(s):
    """Split at ';' outside parentheses."""
    out, depth, start = [], 0, 0
    for k, ch in enumerate(s):
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if ch == ";" and depth == 0:
            out.append(s[start:k])
            start = k + 1
    return out + [s[start:]]


def _lattice_ok(it, code):
    """Pieces inside a lattice motif sit relative to a lattice point, so they
    cannot refer to the sheet's edges."""
    if isinstance(it, Crack):
        raise ValueError(f"cracks (K) need the sheet's edges; not allowed in a lattice motif: "
                         f"{code!r}")
    ys = ([it.y] if isinstance(it, (Slit, Ellipse)) else
          [y for _, y in it.pts] if isinstance(it, (Cut, Region, Strip)) else [])
    if any(y.ref != "c" for y in ys):
        raise ValueError(f"edge coordinates (b, t) are not allowed in a lattice motif: {code!r}")
    for c in getattr(it, "items", ()):
        _lattice_ok(c, code)


def _parse_lattice(code):
    m = grammar._TOP.fullmatch(code)
    if not m:
        raise ValueError(f"not a lattice string: {code!r}")
    slits, extra = [], []
    for tok in _split_top(m[5]):
        s = grammar._SLIT.fullmatch(tok)
        if s:
            ang = grammar.dir_angle(s[1])
            slits.append(grammar.Slit(ang, float(s[2]), grammar.Fraction(s[3]),
                                      grammar.Fraction(s[4])))
            continue
        p = _Parser(tok)
        it = p.item()
        if p.i != len(p.s):
            p.fail("unexpected text in the motif")
        _lattice_ok(it, code)
        extra.append(it)
    base = grammar.Design(float(m[1]), float(m[2]), grammar.Fraction(m[3]), float(m[4]),
                          tuple(slits), int(m[6]), float(m[7]))
    if not (base.px > 0 and base.py > 0):
        raise ValueError("the lattice cell P<px>x<py> needs px > 0 and py > 0")
    if base.n < 1:
        raise ValueError("the number of columns N must be at least 1")
    if not extra:
        return grammar.canonical(base)
    return _canonical_lattice(Lattice(base, tuple(extra)))


def _canonical_lattice(lat):
    """Slits as grammar.canonical puts them; the free pieces move with the
    lattice point when the canonical form changes its offset."""
    d = lat.base
    if d.slits:
        c = grammar.canonical(d)
    else:
        c = replace(d, px=grammar._mm(d.px), py=grammar._mm(d.py), shift=grammar._frac(d.shift),
                    offset=grammar._mm(d.offset % d.py) % grammar._mm(d.py), n=int(d.n),
                    trim=grammar._mm(d.trim))
    dy = c.offset - d.offset
    return Lattice(c, tuple(_shift(it, 0.0, -dy) for it in lat.extra))


def parse(code):
    """A KiriSMILES string -> grammar.Design (lattice of slits), Lattice
    (lattice with free-form pieces) or FreeForm."""
    code = re.sub(r"\s+", "", code)
    if code.startswith("P"):
        return _parse_lattice(code)
    p = _Parser(code)
    items = p.seq()
    if p.i != len(p.s):
        p.fail("unexpected text")
    return FreeForm(items)


# ----------------------------------------------------------------- printing

def _pts(pts):
    return "(" + ";".join(f"{_n(x)},{y}" for x, y in pts) + ")"


def _k(kerf):
    return "" if kerf is None else f"k{_n(kerf)}"


def _item(it):
    if isinstance(it, Slit):
        return f"{grammar.dir_token(it.angle)}{_n(it.length)}@{_n(it.x)},{it.y}{_k(it.kerf)}"
    if isinstance(it, Cut):
        return "C" + _pts(it.pts) + _k(it.kerf)
    if isinstance(it, Crack):
        return f"K{_n(it.length)}@{_n(it.x)}{'t' if it.edge == 't' else ''}{_k(it.kerf)}"
    if isinstance(it, Region):
        return ("+" if it.solid else "X") + _pts(it.pts)
    if isinstance(it, Ellipse):
        at = "" if (it.x, it.y) == (0.0, Y("c", 0.0)) else f"@{_n(it.x)},{it.y}"
        return f"+E({_n(it.a)},{_n(it.b)},{_n(it.angle)}){at}"
    if isinstance(it, Strip):
        return "+B" + _pts(it.pts) + f"w{_n(it.width)}"
    return ("(" + _seq(it.items) + ")" + "".join(f"{{{s}}}" for s in it.syms) + _k(it.kerf))


def _seq(items):
    return ";".join(sorted({_item(it) for it in items}))


def encode(design):
    """Canonical string of a parsed design (any form)."""
    if isinstance(design, grammar.Design):
        return grammar.encode(design)
    if isinstance(design, Lattice):
        lat = _canonical_lattice(design)
        d = lat.base
        motif = [f"{grammar.dir_token(s.angle)}{s.length:g}@"
                 f"{grammar._f(s.u)},{grammar._f(s.v)}" for s in d.slits]
        motif += sorted({_item(it) for it in lat.extra})
        return (f"P{d.px:g}x{d.py:g}S{grammar._f(d.shift)}O{d.offset:g}[{';'.join(motif)}]"
                f"N{d.n}E{d.trim:g}")
    return _seq(design.items)


def canonical(code):
    return encode(parse(code))


# ---------------------------------------------------------------- geometry

def _flip(it, sx, sy):
    """Item mirrored: x -> sx x, y -> sy y."""
    fx = (lambda x: sx * x)
    fy = (lambda y: y.flip() if sy < 0 else y)
    if isinstance(it, Slit):
        ang = it.angle if sx * sy > 0 else (-it.angle) % 180
        return replace(it, angle=ang, x=fx(it.x), y=fy(it.y))
    if isinstance(it, (Cut, Region, Strip)):
        return replace(it, pts=tuple((fx(x), fy(y)) for x, y in it.pts))
    if isinstance(it, Crack):
        return replace(it, x=fx(it.x), edge=({"b": "t", "t": "b"}[it.edge] if sy < 0 else it.edge))
    if isinstance(it, Ellipse):
        return replace(it, angle=it.angle if sx * sy > 0 else -it.angle, x=fx(it.x), y=fy(it.y))
    return replace(it, items=tuple(_flip(c, sx, sy) for c in it.items))


def _shift(it, dx, dy=0.0):
    """Item moved by (dx, dy)."""
    my = (lambda y: replace(y, v=y.v + dy))
    if isinstance(it, Crack):
        return replace(it, x=it.x + dx)
    if isinstance(it, (Slit, Ellipse)):
        return replace(it, x=it.x + dx, y=my(it.y))
    if isinstance(it, (Cut, Region, Strip)):
        return replace(it, pts=tuple((x + dx, my(y)) for x, y in it.pts))
    return replace(it, items=tuple(_shift(c, dx, dy) for c in it.items))


def _prim_key(it):
    """Identity of a primitive: its printed form; a cut is the same cut in
    either direction."""
    if isinstance(it, Cut):
        pts = tuple((_n(x), str(y)) for x, y in it.pts)
        return ("C", min(pts, pts[::-1]), _k(it.kerf))
    return _item(it)


def _unique(prims):
    """Drop repeated primitives, keeping the first: a piece that lies on a
    mirror axis is its own image, and must not be cut twice."""
    seen, out = set(), []
    for q in prims:
        k = _prim_key(q)
        if k not in seen:
            seen.add(k)
            out.append(q)
    return out


def _flatten(items, kerf=None):
    """Primitive items with symmetries applied and group kerfs passed down."""
    return _unique(_flatten_all(items, kerf))


def _flatten_all(items, kerf=None):
    out = []
    for it in items:
        if isinstance(it, Group):
            k = it.kerf if it.kerf is not None else kerf
            prims = _flatten(it.items, k)
            for s in it.syms:
                if s in SYMS:
                    sx, sy = {"mx": (-1, 1), "my": (1, -1), "i": (-1, -1)}[s]
                    prims = _unique(prims + [_flip(p, sx, sy) for p in prims])
                else:
                    p, n = map(float, s[1:].split("x"))
                    prims = [_shift(q, (j - (n - 1) / 2) * p) for j in range(int(n)) for q in prims]
            out += prims
        elif hasattr(it, "kerf") and it.kerf is None and kerf is not None:
            out.append(replace(it, kerf=kerf))
        else:
            out.append(it)
    return out


def _slots(pts, kerf):
    return [geo._slot(p, q, kerf) for p, q in zip(pts[:-1], pts[1:])]


def free_pattern(ff, L, W):
    """The cuts, holes and solids of a FreeForm on an L x W sheet."""
    return _render(_flatten(ff.items), lambda x: L / 2 + x, lambda y: W / 2 + y.at(W), W)


def _render(prims, X, Yc, W):
    """Primitives -> pattern dict; X and Yc map item coordinates to the sheet."""
    pat = geo.empty_pattern()
    for it in prims:
        if isinstance(it, (Slit, Cut, Crack)):
            if isinstance(it, Slit):
                th = np.radians(it.angle)
                h = 0.5 * it.length * np.array([np.cos(th), np.sin(th)])
                c = np.array([X(it.x), Yc(it.y)])
                pts = [tuple(c - h), tuple(c + h)]
            elif isinstance(it, Cut):
                pts = [(X(x), Yc(y)) for x, y in it.pts]
            else:
                y0, s = (0.0, 1) if it.edge == "b" else (W, -1)
                start = y0 - s * (EXTEND if it.kerf is not None else 0.0)
                pts = [(X(it.x), start), (X(it.x), y0 + s * it.length)]
            if it.kerf is None:
                pat["cuts"].append([(float(x), float(y)) for x, y in pts])
            else:
                pat["holes"] += _slots(pts, it.kerf)
        elif isinstance(it, Region):
            pat["solids" if it.solid else "holes"].append([(X(x), Yc(y)) for x, y in it.pts])
        elif isinstance(it, Ellipse):
            th = np.radians(it.angle)
            s = np.linspace(0, 2 * np.pi, N_ELLIPSE, endpoint=False)
            P = np.column_stack([it.a * np.cos(s), it.b * np.sin(s)]) @ np.array(
                [[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]) + [X(it.x), Yc(it.y)]
            pat["solids"].append([tuple(p) for p in P])
        elif isinstance(it, Strip):
            P = geo._bezier([(X(x), Yc(y)) for x, y in it.pts])
            pat["solids"].append(geo._thick_curve(P, it.width))
    return pat


def _clip_polyline(pts, W, tol=1e-9):
    """The pieces of a polyline inside 0 <= y <= W."""
    pieces, cur = [], None
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        t0, t1 = 0.0, 1.0
        if abs(y1 - y0) < tol:
            if y0 < -tol or y0 > W + tol:
                t0, t1 = 1.0, 0.0
        else:
            ta, tb = sorted(((0 - y0) / (y1 - y0), (W - y0) / (y1 - y0)))
            t0, t1 = max(t0, ta), min(t1, tb)
        if t1 - t0 <= tol:
            cur = None
            continue
        a = (x0 + t0 * (x1 - x0), y0 + t0 * (y1 - y0))
        b = (x0 + t1 * (x1 - x0), y0 + t1 * (y1 - y0))
        if cur is not None and t0 <= tol:
            cur.append(b)
        else:
            cur = [a, b]
            pieces.append(cur)
        if t1 < 1 - tol:
            cur = None
    return pieces


def _clip_polygon(pts, W):
    """Sutherland-Hodgman clip of a polygon to 0 <= y <= W."""
    for inside, cut in ((lambda p: p[1] >= 0, 0.0), (lambda p: p[1] <= W, W)):
        out = []
        for k, q in enumerate(pts):
            p = pts[k - 1]
            if inside(q) != inside(p):
                out.append((p[0] + (cut - p[1]) / (q[1] - p[1]) * (q[0] - p[0]), cut))
            if inside(q):
                out.append(q)
        pts = out
        if not pts:
            break
    return pts


def lattice_pattern(lat, L, W):
    """Cuts of a Lattice: the slits as in grammar.pattern, plus one copy of the
    free pieces at every lattice point of columns 0 .. n-1 (all rows that reach
    the sheet). As for slits, cuts are clipped at the free edges, pieces shorter
    than the trim or lying along an edge are dropped; solids are clipped too."""
    d = lat.base
    pat = grammar.pattern(d, L, W) if d.slits else geo.empty_pattern()
    prims = _flatten(lat.extra)
    one = _render(prims, lambda x: x, lambda y: y.v, W)            # around (0, 0)
    ys = [p[1] for g in one["cuts"] + one["holes"] + one["solids"] for p in g]
    if not ys:
        return pat
    lo, hi = min(ys), max(ys)
    x0 = grammar.band(d, L)[0]
    for i in range(d.n):
        xi, yb = x0 + i * d.px, W / 2 + d.offset + i * float(d.shift) * d.py
        for j in range(int(np.floor((-hi - yb) / d.py)), int(np.ceil((W - lo - yb) / d.py)) + 1):
            yi = yb + j * d.py
            g = _render(prims, lambda x: xi + x, lambda y: yi + y.v, W)
            for c in g["cuts"]:
                for piece in _clip_polyline(c, W):
                    a = np.asarray(piece)
                    on_edge = any(np.abs(a[:, 1] - e).max() < 1e-6 for e in (0, W))
                    if np.linalg.norm(np.diff(a, axis=0), axis=1).sum() >= d.trim and not on_edge:
                        pat["cuts"].append([tuple(map(float, p)) for p in piece])
            for h in g["holes"]:
                a = np.asarray(h)
                if a[:, 1].max() > 0 and a[:, 1].min() < W:
                    pat["holes"].append(h)
            for sol in g["solids"]:
                c = _clip_polygon([tuple(p) for p in sol], W)
                if len(c) >= 3:
                    pat["solids"].append(c)
    return pat


def pattern(code, L, W):
    """The cuts / holes / solids of any KiriSMILES string (or parsed design)."""
    d = parse(code) if isinstance(code, str) else code
    if isinstance(d, grammar.Design):
        return grammar.pattern(d, L, W)
    if isinstance(d, Lattice):
        return lattice_pattern(d, L, W)
    return free_pattern(d, L, W)


def spec_pattern(L, W, code):
    """Generator entry for kirigami/spec.py: {"generator": "kirismiles", "params": {"code": ...}}."""
    return pattern(code, L, W)


# ----------------------------------------------------------------- identity

def _straighten(p, closed):
    """Drop repeated points and points that lie on the straight line between
    their neighbours, which do not change the shape."""
    p = [q for k, q in enumerate(p) if k == 0 or q != p[k - 1]]
    changed = True
    while changed and len(p) > 2:
        changed = False
        for k in (range(len(p)) if closed else range(1, len(p) - 1)):
            a, b, c = p[k - 1], p[k], p[(k + 1) % len(p)]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
            if abs(cross) < 1e-9 and dot > 0:
                del p[k]
                changed = True
                break
    return p


def _poly_key(pts, closed):
    p = [tuple(q) for q in pts]
    if closed and len(p) > 1 and p[0] == p[-1]:
        p = p[:-1]
    p = _straighten(p, closed)
    if not closed:
        return min(tuple(p), tuple(p[::-1]))
    cands = []
    for seq in (p, p[::-1]):
        cands += [tuple(seq[k:] + seq[:k]) for k in range(len(seq))]
    return min(cands)


def geometry_key(pat, L, W, nd=2):
    """Hash of a pattern's geometry: independent of how it was written, of the
    order of pieces and points, and of the sheet's two mirror symmetries. Also
    accepts a KiriSMILES string."""
    if isinstance(pat, str):
        pat = pattern(pat, L, W)
    keys = []
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        def norm(g):
            a = np.asarray(g, float)
            a = np.column_stack([a[:, 0] if sx > 0 else L - a[:, 0],
                                 a[:, 1] if sy > 0 else W - a[:, 1]])
            return [tuple(r) for r in (np.round(a, nd) + 0.0).tolist()]
        keys.append(repr([sorted(_poly_key(norm(g), kind != "cuts") for g in pat.get(kind, ()))
                          for kind in ("cuts", "holes", "solids")]))
    return hashlib.sha1(min(keys).encode()).hexdigest()[:12]


# ----------------------------------------------------------------- validity

LOOSE_AREA = 0.25     # mm^2: smaller isolated specks are raster artefacts, not pieces


def _segments(pat):
    """(owner, p, q) for every straight piece of a cut or hole outline."""
    out = []
    for kind in ("cuts", "holes"):
        for i, g in enumerate(pat.get(kind, ())):
            a = [tuple(map(float, p)) for p in g]
            if kind == "holes":
                a = a + a[:1]
            out += [((kind, i), p, q) for p, q in zip(a[:-1], a[1:]) if p != q]
    return out


def _raster(pat, L, W, h):
    """Material on an h-mm pixel grid: the sheet minus holes, plus solids, with
    each zero-width cut drawn as a one-pixel line."""
    from matplotlib.path import Path
    nx, ny = int(np.ceil(L / h)), int(np.ceil(W / h))
    xs, ys = (np.arange(nx) + 0.5) * h, (np.arange(ny) + 0.5) * h
    mat = np.ones((ny, nx), bool)

    def fill(poly, value):
        P = np.asarray(poly, float)
        i0, i1 = np.searchsorted(xs, [P[:, 0].min(), P[:, 0].max()])
        j0, j1 = np.searchsorted(ys, [P[:, 1].min(), P[:, 1].max()])
        if i1 <= i0 or j1 <= j0:
            return
        X, Yg = np.meshgrid(xs[i0:i1], ys[j0:j1])
        inside = Path(P).contains_points(np.column_stack([X.ravel(), Yg.ravel()]))
        mat[j0:j1, i0:i1][inside.reshape(X.shape)] = value

    for g in pat.get("holes", ()):
        fill(g, False)
    for g in pat.get("solids", ()):
        fill(g, True)
    for g in pat.get("cuts", ()):
        for p, q in zip(g[:-1], g[1:]):
            p, q = np.asarray(p, float), np.asarray(q, float)
            t = np.linspace(0, 1, int(np.ceil(np.linalg.norm(q - p) / (h / 4))) + 2)[:, None]
            pts = p + t * (q - p)
            i = np.clip((pts[:, 0] / h).astype(int), 0, nx - 1)
            j = np.clip((pts[:, 1] / h).astype(int), 0, ny - 1)
            mat[j, i] = False
    return mat


def _closest(p1, q1, p2, q2, n=200):
    """Midpoint of the narrowest gap between two segments (for messages)."""
    t = np.linspace(0, 1, n)[:, None]
    A, B = p1 + t * (q1 - p1), p2 + t * (q2 - p2)
    D = np.linalg.norm(A[:, None] - B[None], axis=-1)
    i, j = np.unravel_index(D.argmin(), D.shape)
    return (A[i] + B[j]) / 2


def check(code, L, W, min_gap=1.0, h=None):
    """Problems that make a design impossible to cut or to clamp, as readable
    messages; an empty list means the design is valid. Accepts a string, a
    parsed design or a pattern dict.

      - a piece touches a clamped end (x = 0 or x = L), or a cut leaves the sheet
      - zero-width cuts cross or touch each other, or run into a hole
        (where cuts meet they must be slots: give them a kerf)
      - a ligament narrower than min_gap mm between two pieces, or between a
        piece and a free edge it does not open onto
      - the clamped ends are no longer joined through material
      - a piece of material is cut loose from both clamped ends
    """
    try:
        pat = code if isinstance(code, dict) else pattern(code, L, W)
    except ValueError as e:
        return [str(e)]
    tol = 1e-6
    errs = []
    for kind in ("cuts", "holes", "solids"):
        for i, g in enumerate(pat.get(kind, ())):
            a = np.asarray(g, float)
            if ((a[:, 0] <= tol) | (a[:, 0] >= L - tol)).any():
                errs.append(f"{kind}[{i}] reaches a clamped end (x = 0 or x = {_n(L)})")
            if kind == "cuts" and ((a[:, 1] < -tol) | (a[:, 1] > W + tol)).any():
                errs.append(f"cuts[{i}] leaves the sheet across a free edge")
    if not (pat.get("cuts") or pat.get("holes")):
        errs.append("the pattern has no cuts or holes")
    if errs:
        return errs

    # ligaments between pieces: nearest approach of every pair of pieces
    segs = _segments(pat)
    if len(segs) > 1:
        own = [s[0] for s in segs]
        oid = np.unique([f"{k}{i}" for k, i in own], return_inverse=True)[1]
        P = np.array([s[1] for s in segs])
        Q = np.array([s[2] for s in segs])
        lo, hi = np.minimum(P, Q), np.maximum(P, Q)
        near = {}
        for i0 in range(0, len(segs), 256):            # blocks keep memory small
            a = np.arange(i0, min(i0 + 256, len(segs)))[:, None]
            b = np.arange(len(segs))[None, :]
            gap = np.maximum(lo[b] - hi[a], lo[a] - hi[b]).max(-1)
            a, b = np.nonzero((b > a) & (oid[b] != oid[a]) & (gap < min_gap))
            a += i0
            if not len(a):
                continue
            dist = grammar._seg_dist(P[a], Q[a], P[b], Q[b])
            for i, j, d in zip(a[dist < min_gap], b[dist < min_gap], dist[dist < min_gap]):
                k = (own[i], own[j])
                if k not in near or d < near[k][0]:
                    near[k] = (d, i, j)
        for (o1, o2), (d, i, j) in sorted(near.items()):
            where = "near ({:.1f}, {:.1f})".format(*_closest(P[i], Q[i], P[j], Q[j]))
            n1, n2 = f"{o1[0]}[{o1[1]}]", f"{o2[0]}[{o2[1]}]"
            if d < tol:
                if o1[0] == o2[0] == "cuts":
                    errs.append(f"{n1} and {n2} cross or touch {where}; where cuts meet "
                                "they must be slots (give them a kerf)")
                elif "cuts" in (o1[0], o2[0]):
                    errs.append(f"{n1} runs into {n2} {where}; give the cut a kerf")
                # two holes that overlap simply merge into one opening
            elif d < min_gap - tol:
                errs.append(f"ligament of {d:.2f} mm < {_n(min_gap)} mm between {n1} and {n2} "
                            f"{where}")

    # slivers along the free edges y = 0 and y = W
    for kind in ("cuts", "holes"):
        for i, g in enumerate(pat.get(kind, ())):
            y = np.asarray(g, float)[:, 1]
            for edge, d in (("y = 0", y.min()), (f"y = {_n(W)}", W - y.max())):
                if tol < d < min_gap - tol:
                    errs.append(f"{kind}[{i}] leaves a {d:.2f} mm sliver along the free edge "
                                f"{edge}")

    # the clamped ends must stay joined, and nothing may be cut loose
    from scipy import ndimage
    h = h or min(0.1, min_gap / 4)
    lab, n = ndimage.label(_raster(pat, L, W, h))
    left, right = set(lab[:, 0][lab[:, 0] > 0]), set(lab[:, -1][lab[:, -1] > 0])
    if not left & right:
        errs.append("the clamped ends are not joined: the pattern cuts the sheet in two")
    area = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * h * h
    loose = [k + 1 for k in range(n) if k + 1 not in left | right and area[k] >= LOOSE_AREA]
    for k in loose:
        jj, ii = np.nonzero(lab == k)
        errs.append(f"a piece of {area[k - 1]:.1f} mm^2 near ({ii.mean() * h:.1f}, "
                    f"{jj.mean() * h:.1f}) is cut loose from both clamped ends")
    return errs


def is_valid(code, L, W, min_gap=1.0):
    return not check(code, L, W, min_gap)
