"""A 2 x 2 supercell of the paper's pitch cell (A) and roll cell (B), written as one free-form
KiriSMILES string, and saved as the spec specs/pitch_roll_2x2.json.

    python scripts/build_pitch_roll.py            # writes specs/pitch_roll_2x2.json
    python scripts/build_pitch_roll.py --print    # prints the string only

Layout on a 290 x 121 mm sheet (free-form coordinates: x from the middle, y from the centre
line, in mm):

    top row    (y from 0.5 to 60.5):    A at x = -62.5,  B at x = +62.5
    separator  (y from -0.5 to 0.5):    a 1 mm slot from x = -135 to +135
    bottom row (y from -60.5 to -0.5):  B at x = -62.5,  A at x = +62.5

Both cells were designed for a 60 mm strip with two free edges. The slot gives each row its
own free edges; the rows stay joined by 10 mm pads next to the clamps. Each cell's pieces are
those of the published cell on its own 60 mm sheet, translated. An edge crack that meets the
slot is written as a 1 mm slot that overlaps it, since a zero-width cut cannot end inside an
opening.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kirigami import kirismiles as K  # noqa: E402
from kirigami._types import Point, Spec  # noqa: E402

PITCH = "(C(2,16;47.5,16;47.5,-16;2,-16)){mx};(K18.75@0){my}"
ROLL = ("((K23.25@49){my};(K7.5@34.4){my};C(41.6,0;5,0);C(41.6,b+2.9;41.6,t-3.5);"
        "X(1.25,-2.9;5.5,0;1.25,2.9)){mx}k1")
CELLS = {"A": (PITCH, 180.0), "B": (ROLL, 150.0)}    # string, length of its own sheet (mm)
LAYOUT = (("A", "B"), ("B", "A"))                    # top row, bottom row
L, ROW, SLOT, XC, PAD = 290.0, 60.0, 1.0, 62.5, 10.0
W = 2 * ROW + SLOT
OUT = ROOT / "specs" / "pitch_roll_2x2.json"
NAME = "pitch_roll_2x2"


def _f(v: float) -> str:
    return f"{round(v, 2):g}"


def _pts(p: list[Point]) -> str:
    return ";".join(f"{_f(x)},{_f(y)}" for x, y in p)


def place(name: str, xc: float, top: bool) -> list[str]:
    """The pieces of cell `name` ("A" or "B") centred at free-form x = xc (mm), in the top or
    the bottom row, as free-form pieces."""
    if name not in CELLS:
        raise ValueError(f"the cell must be one of {sorted(CELLS)}, not {name!r}")
    code, Lc = CELLS[name]
    pat = K.pattern(code, Lc, ROW)
    y0 = SLOT / 2 if top else -SLOT / 2 - ROW        # free-form y of the row's local y = 0
    inner = 0.0 if top else ROW                      # local y of the row edge that meets the slot

    def tr(x: float, y: float) -> Point:
        return (x - Lc / 2 + xc, y + y0)

    out = []
    for c in pat["cuts"]:
        if len(c) == 2 and inner in (c[0][1], c[1][1]):   # a crack from the inner edge
            (xa, ya), (_, yb) = c
            far = yb if ya == inner else ya
            reach = inner - 0.5 if top else inner + 0.5   # overlap the separator by 0.5 mm
            lo, hi = sorted((reach, far))
            out.append("X(" + _pts([tr(xa - 0.5, lo), tr(xa + 0.5, lo),
                                    tr(xa + 0.5, hi), tr(xa - 0.5, hi)]) + ")")
        else:
            out.append("C(" + _pts([tr(x, y) for x, y in c]) + ")")
    for h in pat["holes"]:                            # slots past an edge stop 1 mm beyond it
        out.append("X(" + _pts([tr(x, min(max(y, -1.0), ROW + 1.0)) for x, y in h]) + ")")
    return out


def code() -> str:
    """The free-form string of the whole 2 x 2 design."""
    pieces = [f"X({_f(-L / 2 + PAD)},-0.5;{_f(L / 2 - PAD)},-0.5;"
              f"{_f(L / 2 - PAD)},0.5;{_f(-L / 2 + PAD)},0.5)"]
    for row, top in zip(LAYOUT, (True, False)):
        for name, xc in zip(row, (-XC, XC)):
            pieces += place(name, xc, top)
    return ";".join(pieces)


def spec() -> Spec:
    """The spec of the design: the default material, and the mesh of the paper's roll cell."""
    return {
        "name": NAME,
        "sheet": {"L": L, "W": W},
        "pattern": {"generator": "kirismiles", "params": {"code": code()}},
        "material": {"E": 3500.0, "nu": 0.38, "h": 0.127},
        "mesh": {"h_cut": 0.6, "h_far": 3.0, "refine_dist": [1.0, 6.0]},
        "load": {"Dmax": 29.0, "steps": 45},
        "ell": 30,
    }


def main() -> None:
    s = code()
    errs = K.check(s, L, W)
    if errs:
        raise SystemExit("the design is not valid: " + "; ".join(errs))
    if "--print" in sys.argv:
        print(s)
        return
    if OUT.exists() and json.loads(OUT.read_text()).get("name") != NAME:
        raise SystemExit(f"{OUT} exists and is not this design's spec; not overwriting it")
    OUT.write_text(json.dumps(spec(), indent=2) + "\n")
    print(f"{len(s)} characters, geometry key {K.geometry_key(s, L, W)}; wrote {OUT}")


if __name__ == "__main__":
    main()
