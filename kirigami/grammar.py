"""A line notation for periodic kirigami cut patterns (SMILES-like).

A pattern is a motif of straight slits repeated on an oblique lattice inside a
band of the sheet:

    P5x45S1/2O0[V42@0,0]N18E3

  P<px>x<py>   lattice cell (mm): px along the pulled direction x, py across
  S<s>         column shift: column i is shifted by i * s * py along y
  O<o>         y offset (mm) of column 0's lattice point from the sheet's centre line
  [...]        the motif: slits joined by ';', each <dir><length>@<u>,<v>
                 dir: H (along x), V (along y), D (45 deg), A (135 deg), or R<deg>: for
                 any other whole angle, e.g. R30:24@0,0 (the ':' ends the angle)
                 u, v: position in the cell as fractions of px and py
  N<n>         number of lattice columns; the band spans (n - 1) * px, centred
  E<e>         trim (mm): every cut piece shorter than e is dropped (mainly the stubs clipped by the free edges)

Rules that make the string a complete description: the lattice fills the
plane; a slit is kept only if it lies entirely inside the band's x-range;
slits are clipped to the sheet; a slit lying along a free edge is dropped.

The example above is the classic alternating pattern (b = 42, g = 3, d = 5, 18 rows);
rotating squares (panel a, hinge t) is P{a}x{2a}S1/2O..[H{2a-t}@0,0;V{2a-t}@0,1/2].
Nothing in the notation favours either: they are two points among many.

Two identities:
  encode(design)        canonical string: slits sorted, positions reduced mod 1, and the
                        motif re-anchored so equivalent descriptions print identically
  geometry_hash(cuts)   hash of the actual cut set, the same for mirror images
                        (y -> W - y, x -> L - x); the de-duplication key
Sheet size, material and thickness are context, kept outside the string.
"""
import hashlib
import re
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from kirigami.geometry import clip_to_sheet

LETTER = {0: "H", 45: "D", 90: "V", 135: "A"}
ANGLE = {v: k for k, v in LETTER.items()}
DIR = r"(H|D|V|A|R\d+:)"         # slit direction; 'R30:' is 30 deg, the ':' ends the angle


def dir_angle(tok):
    """'H', 'D', 'V', 'A' or 'R<deg>:' -> angle in degrees, in [0, 180)."""
    return ANGLE[tok] if tok in ANGLE else int(tok[1:-1]) % 180


def dir_token(angle):
    """Angle in degrees -> its direction token (a letter, or 'R<deg>:')."""
    angle = int(angle) % 180
    return LETTER.get(angle, f"R{angle}:")
TOL = 1e-6


@dataclass(frozen=True)
class Slit:
    angle: int                  # degrees from x, in [0, 180)
    length: float               # mm
    u: Fraction                 # position in the cell (fraction of px)
    v: Fraction                 # (fraction of py)


@dataclass(frozen=True)
class Design:
    px: float
    py: float
    shift: Fraction
    offset: float
    slits: tuple
    n: int
    trim: float


def _mm(x):
    return round(float(x), 3) + 0.0


def _frac(x):
    """A lattice fraction, reduced to [0, 1). Fractions are kept exact up to a
    denominator of 24; a written fraction with a larger one is refused rather
    than silently moved to the nearest allowed value."""
    q = Fraction(x)
    r = q.limit_denominator(24)
    if r != q and q.denominator < 10**6:              # a float such as 0.3333... is rounded
        raise ValueError(f"lattice fraction {q} has a denominator above 24")
    return r % 1


def _f(q):
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def _slit_key(s):
    return (s.angle, s.length, s.u, s.v)


def canonical(d):
    if not (d.px > 0 and d.py > 0):
        raise ValueError("the lattice cell P<px>x<py> needs px > 0 and py > 0")
    if int(d.n) < 1:
        raise ValueError("the number of columns N must be at least 1")
    if any(s.length <= 0 for s in d.slits):
        raise ValueError("a slit needs a length above zero")
    slits = {Slit(int(s.angle) % 180, _mm(s.length), _frac(s.u), _frac(s.v)) for s in d.slits}
    best = None
    for anchor in sorted(slits, key=_slit_key):      # put each slit at v = 0, keep the smallest
        moved = tuple(sorted((replace(s, v=(s.v - anchor.v) % 1) for s in slits), key=_slit_key))
        off = _mm(float(d.offset + anchor.v * d.py) % d.py) % _mm(d.py)
        cand = (moved, _mm(off))
        if best is None or (tuple(map(_slit_key, cand[0])), cand[1]) < \
                (tuple(map(_slit_key, best[0])), best[1]):
            best = cand
    return Design(_mm(d.px), _mm(d.py), _frac(d.shift), best[1], best[0], int(d.n), _mm(d.trim))


def encode(d):
    d = canonical(d)
    motif = ";".join(f"{dir_token(s.angle)}{s.length:g}@{_f(s.u)},{_f(s.v)}"
                     for s in d.slits)
    return f"P{d.px:g}x{d.py:g}S{_f(d.shift)}O{d.offset:g}[{motif}]N{d.n}E{d.trim:g}"


_NUM, _FR = r"\d+(?:\.\d+)?", r"\d+(?:/\d+)?"
_TOP = re.compile(rf"P({_NUM})x({_NUM})S({_FR})O({_NUM})\[([^\]]+)\]N(\d+)E({_NUM})")
_SLIT = re.compile(rf"{DIR}({_NUM})@({_FR}),({_FR})")


def decode(code):
    m = _TOP.fullmatch(code)
    if not m:
        raise ValueError(f"not a pattern string: {code!r}")
    slits = []
    for tok in m[5].split(";"):
        s = _SLIT.fullmatch(tok)
        if not s:
            raise ValueError(f"bad slit {tok!r} in {code!r}")
        ang = dir_angle(s[1])
        slits.append(Slit(ang, float(s[2]), Fraction(s[3]), Fraction(s[4])))
    return canonical(Design(float(m[1]), float(m[2]), Fraction(m[3]), float(m[4]), tuple(slits),
                            int(m[6]), float(m[7])))


def band(d, L):
    x0 = L / 2 - (d.n - 1) * d.px / 2
    return x0, x0 + (d.n - 1) * d.px


def pattern(d, L, W):
    """The cuts of a design on an L x W sheet (kirigami/geometry.py pattern dict)."""
    if isinstance(d, str):
        d = decode(d)
    x0, x1 = band(d, L)
    reach = max(s.length for s in d.slits) / 2 + d.px + d.py
    cuts = []
    for i in range(-int(reach // d.px) - 2, d.n + int(reach // d.px) + 2):
        xi, yi = x0 + i * d.px, W / 2 + d.offset + i * float(d.shift) * d.py
        for s in d.slits:
            th = np.radians(s.angle)
            hx, hy = 0.5 * s.length * np.cos(th), 0.5 * s.length * np.sin(th)
            cx, cy0 = xi + float(s.u) * d.px, yi + float(s.v) * d.py
            if cx - abs(hx) < x0 - TOL or cx + abs(hx) > x1 + TOL:
                continue
            for j in range(int(np.floor((-abs(hy) - cy0) / d.py)),
                           int(np.ceil((W + abs(hy) - cy0) / d.py)) + 1):
                cy = cy0 + j * d.py
                cuts.append([(cx - hx, cy - hy), (cx + hx, cy + hy)])
    pat = clip_to_sheet(dict(cuts=cuts, holes=[], solids=[]), L, W, min_len=d.trim)
    pat["cuts"] = [c for c in pat["cuts"]
                   if not any(abs(c[0][1] - e) < TOL and abs(c[1][1] - e) < TOL for e in (0, W))]
    return pat


def geometry_hash(cuts, L, W, nd=2):
    """Hash of the cut set, invariant to cut order, endpoint order and the
    sheet's two mirror symmetries."""
    segs = np.asarray(cuts, float).reshape(-1, 2, 2)
    keys = []
    for fx, fy in ((False, False), (False, True), (True, False), (True, True)):
        s = segs.copy()
        if fx:
            s[..., 0] = L - s[..., 0]
        if fy:
            s[..., 1] = W - s[..., 1]
        r = np.round(s, nd) + 0.0
        keys.append(repr(sorted(tuple(sorted(map(tuple, c.tolist()))) for c in r)))
    return hashlib.sha1(min(keys).encode()).hexdigest()[:12]


def _seg_dist(P1, Q1, P2, Q2):
    """Distance between segments P1Q1 and P2Q2 (arrays of shape (..., 2))."""
    d1, d2, r = Q1 - P1, Q2 - P2, P1 - P2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    den = a * e - b * b
    ok = den > 1e-12
    s = np.where(ok, np.clip((b * f - c * e) / np.where(ok, den, 1), 0, 1), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
    t = np.clip(t, 0, 1)
    return np.linalg.norm(P1 + d1 * s[..., None] - P2 - d2 * t[..., None], axis=-1)


def clearance(cuts):
    """Smallest distance between two different cuts (0 if any touch or cross):
    the narrowest ligament, which sets the mesh size."""
    c = np.asarray(cuts, float).reshape(-1, 2, 2)
    if len(c) < 2:
        return np.inf
    i, j = np.triu_indices(len(c), 1)
    return float(_seg_dist(c[i, 0], c[i, 1], c[j, 0], c[j, 1]).min())


def features(code, L, W):
    """Generic, family-free descriptors of a pattern."""
    d = decode(code)
    cuts = pattern(d, L, W)["cuts"]
    seg = np.asarray(cuts, float).reshape(-1, 2, 2)
    dv = seg[:, 1] - seg[:, 0]
    lens = np.linalg.norm(dv, axis=1)
    x0, x1 = band(d, L)
    return dict(n_cuts=len(cuts), cut_length=float(lens.sum()), clearance=clearance(cuts),
                longest=float(lens.max()),
                transverse=float((np.abs(dv[:, 1])).sum() / lens.sum()),   # share across the pull
                coverage=(x1 - x0) / L, density=float(lens.sum()) / max((x1 - x0) * W, 1e-9),
                geom=geometry_hash(cuts, L, W) if cuts else None)


# ---------------------------------------------------- the earlier generator families

def from_legacy(family, p, W):
    if family == "alternating":
        return Design(p["d"], p["b"] + p["g"], Fraction(1, 2), 0.0,
                      (Slit(90, p["b"], Fraction(0), Fraction(0)),), p["n_rows"], p["g"])
    if family == "rotating_squares":
        a, t = p["a"], p["t"]
        return Design(a, 2 * a, Fraction(1, 2), (-W / 2) % (2 * a),
                      (Slit(0, 2 * a - t, Fraction(0), Fraction(0)),
                       Slit(90, 2 * a - t, Fraction(0), Fraction(1, 2))), p["nx"] + 1, t)
    raise ValueError(f"no grammar form for family {family!r}")


def lattice_pattern(L, W, code):
    """Generator entry for kirigami/spec.py: {"generator": "lattice", "params": {"code": ...}}."""
    return pattern(code, L, W)
