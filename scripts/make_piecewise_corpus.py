"""Free-form strings for corpus designs that have no compact lattice form.

    python scripts/make_piecewise_corpus.py

Prints two strings, both derived from the rotating-squares lattice
P8x16S1/2O2[H12@0,0;V12@0,1/2]N9E4 on a 120 x 60 sheet:

  graded   every slit is shortened or lengthened about its centre, so the hinge
           width grows linearly from about 1 mm at the left of the band to
           about 3 mm at the right (cf. the non-uniform tessellations of
           Choi et al. 2019, which are likewise written piece by piece);
  random   each slit is kept with probability p = 0.7 (seed 1), one
           realization of the random cut sets of Chen et al. 2020.

Edge cuts (clipped by a free edge) are written as cracks K<len>@x[b|t].
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kirigami import kirismiles as ks   # noqa: E402

L, W = 120, 60
BASE = "P8x16S1/2O2[H12@0,0;V12@0,1/2]N9E4"
A = 8                      # panel side; a slit of length 2a - t leaves hinges of about t/2


def num(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def piece(c, length):
    """One clipped base cut as a free-form slit or crack with a new length."""
    (x0, y0), (x1, y1) = c[0], c[-1]
    xc, yc = (x0 + x1) / 2 - L / 2, (y0 + y1) / 2 - W / 2
    horiz = abs(y1 - y0) < 1e-9
    on_b, on_t = min(y0, y1) <= 1e-6, max(y0, y1) >= W - 1e-6
    if not horiz and (on_b or on_t):                    # vertical edge stub -> crack
        frac = length / (2 * A - 4)                     # scale stub like the full slit
        k = abs(y1 - y0) * frac
        return f"K{num(k)}@{num(xc)}" + ("t" if on_t else "")
    return f"{'H' if horiz else 'V'}{num(length)}@{num(xc)},{num(yc)}"


def graded():
    cuts = ks.pattern(BASE, L, W)["cuts"]
    xs = [np.mean(np.asarray(c)[:, 0]) for c in cuts]
    lo, hi = min(xs), max(xs)
    out = []
    for c, x in zip(cuts, xs):
        t = 2 + 4 * (x - lo) / (hi - lo)                # hinge parameter t: 2 -> 6 (hinge ~1 -> 3 mm)
        out.append(piece(c, 2 * A - t))
    return ";".join(out)


def random_cuts(p=0.7, seed=1):
    """Old ensemble (whole slits removed); kept for reference, not used in the corpus."""
    rng = random.Random(seed)
    cuts = ks.pattern(BASE, L, W)["cuts"]
    kept = [c for c in cuts if rng.random() < p]
    return ";".join(piece(c, 2 * A - 4) for c in kept)


def random_links(p=0.8, seed=1, kerf=0.5):
    """One realization of the random link patterns of Chen, Choi & Mahadevan (PNAS 2020):
    every hinge (link) between neighbouring squares of the rotating-squares lattice is
    kept with probability p and cut otherwise. A hinge sits at each slit end, between the
    slit tip and the perpendicular slit; cutting it extends the slit by the ligament width
    t/2 = 2 mm so that it meets the perpendicular slit. Cuts are slots of width `kerf`
    so that they may meet."""
    rng = random.Random(seed)
    gap = 2.0
    out = []
    for c in ks.pattern(BASE, L, W)["cuts"]:
        (x0, y0), (x1, y1) = c[0], c[-1]
        horiz = abs(y1 - y0) < 1e-9
        lo, hi = (min(x0, x1), max(x0, x1)) if horiz else (min(y0, y1), max(y0, y1))
        edge = 0.0 if horiz else W
        on_lo = (not horiz) and lo <= 1e-6           # vertical stub starting at y = 0
        on_hi = (not horiz) and hi >= edge - 1e-6    # vertical stub ending at y = W
        if not on_lo and rng.random() > p:
            lo -= gap
        if not on_hi and rng.random() > p:
            hi += gap
        if horiz:
            xc, yc = (lo + hi) / 2 - L / 2, y0 - W / 2
            out.append(f"H{num(hi - lo)}@{num(xc)},{num(yc)}")
        elif on_lo or on_hi:
            out.append(f"K{num(hi - lo)}@{num(x0 - L / 2)}" + ("t" if on_hi else ""))
        else:
            out.append(f"V{num(hi - lo)}@{num(x0 - L / 2)},{num((lo + hi) / 2 - W / 2)}")
    return "(" + ";".join(out) + f")k{num(kerf)}"


if __name__ == "__main__":
    for name, code in (("graded", graded()), ("random", random_cuts())):
        problems = ks.check(code, L, W)
        print(f"{name}: {len(code)} chars, {len(ks.pattern(code, L, W)['cuts'])} cuts, "
              f"{'valid' if not problems else problems[:3]}")
        print(code)
