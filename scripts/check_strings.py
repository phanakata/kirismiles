"""Round trip of every design simulated in the paper: geometry -> KiriSMILES string -> geometry.

    python scripts/check_strings.py              geometry only (no solver needed)
    ./run.sh python3 scripts/check_strings.py --buckle     also compare the buckling threshold
                                                            of the two inputs (runs one at a time)

For each design, the reference is the geometry the simulation actually used (the generator in
kirigami/geometry.py, or for the star the string itself, re-expanded from its canonical form).
The string is expanded with kirismiles.pattern and compared with the reference:

  pieces     the numbers of cuts, holes and solids must be equal;
  distance   the largest distance between the two geometries (symmetric Hausdorff distance
             between the cut lines and outlines, measured to the segment), in mm;
  key        whether the geometry keys (vertices rounded to 0.01 mm) agree.

With --buckle, the buckling threshold Delta_c of each design is computed from both inputs with
the same mesh settings, and the relative difference is printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kirigami import geometry as geo          # noqa: E402
from kirigami import kirismiles as ks         # noqa: E402

PITCH = "(C(2,16;47.5,16;47.5,-16;2,-16)){mx};(K@C@0){my}"
STAR = {s: c for s, c in [
    ("0.028", "((C(-0.63,6.49;-1.66,1.66;-6.49,0.63);C(-0.63,20.49;-1.66,15.66;-6.49,14.63);"
              "C(-0.63,7.51;-1.66,12.34;-6.49,13.37)){mx}{my}{t14x3};(D13.8@7,7;A13.8@7,7){mx}{my};"
              "(C(2.12,16.12;7,21;11.88,16.12);C(16.12,2.12;21,7;16.12,11.88);C(16.12,16.12;21,21))"
              "{mx}{my};(H12.4@0,7){my}{t14x3};(V12.4@7,0;V12.4@21,0;V12.4@7,14;V12.4@21,14){mx}{my})k0.2"),
    ("0.039", "((C(-0.63,6.33;-1.63,1.63;-6.33,0.63);C(-0.63,20.33;-1.63,15.63;-6.33,14.63);"
              "C(-0.63,7.67;-1.63,12.37;-6.33,13.37)){mx}{my}{t14x3};(D13.52@7,7;A13.52@7,7){mx}{my};"
              "(C(2.22,16.22;7,21;11.78,16.22);C(16.22,2.22;21,7;16.22,11.78);C(16.22,16.22;21,21))"
              "{mx}{my};(H11.96@0,7){my}{t14x3};(V11.96@7,0;V11.96@21,0;V11.96@7,14;V11.96@21,14){mx}{my})k0.2"),
    ("0.050", "((C(-0.63,6.18;-1.6,1.6;-6.18,0.63);C(-0.63,20.18;-1.6,15.6;-6.18,14.63);"
              "C(-0.63,7.82;-1.6,12.4;-6.18,13.37)){mx}{my}{t14x3};(D13.24@7,7;A13.24@7,7){mx}{my};"
              "(C(2.32,16.32;7,21;11.68,16.32);C(16.32,2.32;21,7;16.32,11.68);C(16.32,16.32;21,21))"
              "{mx}{my};(H11.54@0,7){my}{t14x3};(V11.54@7,0;V11.54@21,0;V11.54@7,14;V11.54@21,14){mx}{my})k0.2"),
]}


def rs(t: float) -> str:
    """Rotating squares of side 8 mm with slits 16 - t long (hinges about t/2 wide), 0 < t < 16."""
    if not 0 < t < 16:
        raise ValueError(f"the hinge t must be between 0 and 16 mm, not {t}")
    return f"P8x16S1/2O0[H{16 - t:g}@0,0;V{16 - t:g}@0,1/2]N9E4"


# (name, string, L, W, reference geometry or None for a string-only design)
DESIGNS = [
    ("single cut", "V80@0,0", 182, 100, geo.single_cut_pattern(182, 100, 80)),
    ("two cuts, 10 mm apart", "(V40@5,0){mx}", 182, 100, geo.two_cuts_pattern(182, 100, 40, 10)),
    ("two cuts, 30 mm apart", "(V40@15,0){mx}", 182, 100, geo.two_cuts_pattern(182, 100, 40, 30)),
    ("pitch, concave", PITCH.replace("@C", "18.75"), 180, 60, geo.pitch_pattern(180, 60, 18.75, 32, 95, 4)),
    ("pitch, convex", PITCH.replace("@C", "8"), 180, 60, geo.pitch_pattern(180, 60, 8, 32, 95, 4)),
    ("pitch chain", "(" + PITCH.replace("@C", "18.75") + "){t125x3}", 415, 60,
     geo.pitch_chain_pattern(415, 60, 3, 125, 18.75, 32, 95, 4)),
    ("roll", "((K23.25@49){my};C(41.6,b+2.9;41.6,t-3.5);(K7.5@34.4){my};C(41.6,0;5,0);"
             "X(1.25,-2.9;5.5,0;1.25,2.9)){mx}k1", 150, 60, geo.roll_pattern(150, 60, 1.0)),
    ("yaw", "X(-30,b+4;30,b+4;30,t-4;-30,t-4);+E(14,8,20);"
            "(+B(-24,t-2;-24,2;-16,14;-2,4)w3;K30@-31.5t){i}", 150, 60, geo.yaw_pattern(150, 60)),
    ("alternating slits", "P5x40S1/2O0[V36@0,0]N20E4", 120, 60,
     geo.alternating_pattern(120, 60, 36, 4, 5, 20)),
] + [
    (f"rotating squares, hinge {t / 2:g} mm, {L} x 64", rs(t), L, 64,
     geo.rotating_squares_pattern(L, 64, 8, t, 8))
    for t, L in ((4, 72), (3, 72), (2, 72), (1, 72), (4, 120))
] + [
    (f"star, s/l = {s}", code, 50, 42, None) for s, code in STAR.items()
]


def distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Largest distance between two geometries, in mm (symmetric Hausdorff distance between
    the cut lines and outlines; kirigami.importer.deviation, exact to the segment)."""
    from kirigami.importer import deviation
    return deviation(a, b)


def counts(pat: dict[str, Any]) -> tuple[int, int, int]:
    return len(pat.get("cuts", ())), len(pat.get("holes", ())), len(pat.get("solids", ()))


def check(name: str, code: str, L: float, W: float, ref: dict[str, Any] | None) -> dict[str, Any]:
    """Round trip of one design: the string's geometry against the reference (or, for a
    string-only design, against its own canonical form)."""
    got = ks.pattern(code, L, W)
    if ref is None:                                     # string-only design: canonical round trip
        ref, code = got, ks.canonical(code)
        got = ks.pattern(code, L, W)
    d = distance(ref, got)
    same_key = ks.geometry_key(got, L, W) == ks.geometry_key(ref, L, W)
    ok = counts(got) == counts(ref) and d < 0.01 and same_key
    return dict(name=name, code=code, L=L, W=W, pieces=counts(got), ref_pieces=counts(ref),
                distance=d, key=same_key, ok=ok, ref=ref, got=got)


def buckle_both(r: dict[str, Any]) -> list[float]:
    """Delta_c from the reference geometry and from the string, same mesh settings."""
    from kirigami import api
    from kirigami import spec as S
    base = dict(sheet=dict(L=r["L"], W=r["W"]), material=dict(E=4000.0, nu=0.4, h=0.127),
                mesh=dict(h_cut=1.0, h_far=4.0))
    raw = {k: [list(map(list, g)) for g in r["ref"].get(k, ())] for k in ("cuts", "holes", "solids")}
    from_string = dict(generator="kirismiles", params=dict(code=r["code"]))   # the string itself
    out = []
    for pat in (raw, from_string):
        out.append(api.buckle(S.load(dict(base, pattern=pat)))["Delta_c_mm"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckle", action="store_true")
    a = ap.parse_args()
    print(f"{'design':42s} {'pieces (c,h,s)':>16s} {'distance':>10s}  key  result")
    bad = 0
    for row in DESIGNS:
        r = check(*row)
        bad += not r["ok"]
        line = (f"{r['name']:42s} {str(r['pieces']):>16s} {r['distance']:9.4f}mm  "
                f"{'yes' if r['key'] else 'NO ':3s}  {'ok' if r['ok'] else 'MISMATCH'}")
        if a.buckle:
            d0, d1 = buckle_both(r)
            line += f"   Delta_c {d0:.5f} / {d1:.5f} mm ({abs(d1 - d0) / d0 * 100:.3f}%)"
        print(line, flush=True)
    print(f"\n{len(DESIGNS) - bad} of {len(DESIGNS)} designs round-trip")
    return bad


if __name__ == "__main__":
    sys.exit(main())
