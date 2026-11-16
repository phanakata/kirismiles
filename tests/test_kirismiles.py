"""KiriSMILES strings must cut exactly what the paper's generators cut.

    python -m pytest tests/test_kirismiles.py      (or: python tests/test_kirismiles.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kirigami import geometry as geo          # noqa: E402
from kirigami import grammar                   # noqa: E402
from kirigami import kirismiles as ks          # noqa: E402

PITCH = "(C(2,16;47.5,16;47.5,-16;2,-16)){mx};(K18.75@0){my}"

# (name, string, L, W, generator output)
CASES = [
    ("single cut", "V80@0,0", 182, 100, geo.single_cut_pattern(182, 100, 80)),
    ("two cuts", "(V40@5,0){mx}", 182, 100, geo.two_cuts_pattern(182, 100, 40, 10)),
    ("pitch", PITCH, 180, 60, geo.pitch_pattern(180, 60, 18.75, 32, 95, 4)),
    ("pitch chain", f"({PITCH}){{t125x3}}", 415, 60,
     geo.pitch_chain_pattern(415, 60, 3, 125, 18.75, 32, 95, 4)),
    ("roll", "((K23.25@49){my};C(41.6,b+2.9;41.6,t-3.5);(K7.5@34.4){my};C(41.6,0;5,0);"
             "X(1.25,-2.9;5.5,0;1.25,2.9)){mx}k1", 150, 60, geo.roll_pattern(150, 60, 1.0)),
    ("yaw", "X(-30,b+4;30,b+4;30,t-4;-30,t-4);+E(14,8,20);"
            "(+B(-24,t-2;-24,2;-16,14;-2,4)w3;K30@-31.5t){i}", 150, 60, geo.yaw_pattern(150, 60)),
    ("alternating", "P5x40S1/2O0[V36@0,0]N20E4", 120, 60,
     geo.alternating_pattern(120, 60, 36, 4, 5, 20)),
    ("rotating squares", grammar.encode(grammar.from_legacy(
        "rotating_squares", dict(a=10, t=2, nx=8), 60)), 120, 60,
     geo.rotating_squares_pattern(120, 60, 10, 2, 8)),
    ("rotating squares, benchmark", "P8x16S1/2O0[H12@0,0;V12@0,1/2]N9E4", 72, 64,
     geo.rotating_squares_pattern(72, 64, 8, 4, 8)),
]


def check(name, code, L, W, ref):
    got = ks.pattern(code, L, W)
    assert ks.geometry_key(got, L, W) == ks.geometry_key(ref, L, W), name
    canon = ks.canonical(code)
    assert ks.canonical(canon) == canon, (name, "canonical form is not stable")
    assert ks.geometry_key(canon, L, W) == ks.geometry_key(ref, L, W), (name, "canonical")
    return canon


def test_paper_cells():
    for case in CASES:
        check(*case)


def test_mirror_images_share_a_key():
    a = ks.geometry_key("C(0,b;10,0)", 100, 50)
    b = ks.geometry_key("C(0,t;10,0)", 100, 50)            # mirrored in y
    c = ks.geometry_key("C(0,b;-10,0)", 100, 50)           # mirrored in x
    assert a == b == c
    assert a != ks.geometry_key("C(0,b;10,1)", 100, 50)


def test_equivalent_strings_share_a_key():
    a = ks.geometry_key("(V40@6,0){mx}", 120, 60)
    b = ks.geometry_key("V40@-6,0;V40@6,0", 120, 60)
    c = ks.geometry_key("C(6,-20;6,20);C(-6,20;-6,-20)", 120, 60)
    assert a == b == c
    assert ks.canonical("V40@6,0;V40@-6,0") == ks.canonical("V40@-6,0;V40@6,0")


def test_bad_strings_are_rejected():
    for bad in ["", "Q1@0,0", "C(0,0)", "(V4@0,0){zz}", "+B(0,0;1,1)w1", "V4@0,0;"]:
        try:
            ks.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_angled_slits_round_trip():
    import numpy as np
    # free form: R<deg>:<length>, and the canonical string parses back to the same cut
    for code in ["R30:24@0,0", "R150:12.5@-10,b+20k0.5", "(R30:24@10,5){mx}"]:
        c = ks.canonical(code)
        assert ks.canonical(c) == c and ks.geometry_key(c, 120, 60) == \
            ks.geometry_key(code, 120, 60), (code, c)
    (p, q), = ks.pattern("R30:24@0,0", 120, 60)["cuts"]
    assert np.allclose(np.subtract(q, p), [24 * np.cos(np.pi / 6), 12])
    # the four named angles print as their letters; angles are taken mod 180
    assert ks.canonical("R90:20@0,0") == "V20@0,0"
    assert ks.canonical("R210:20@0,0") == "R30:20@0,0"
    # lattice: same syntax, read by both parsers
    lat = "P10x20S0O0[R30:8@0,0]N5E1"
    assert ks.canonical(lat) == grammar.encode(grammar.decode(lat)) == lat
    assert grammar.decode(lat).slits[0].angle == 30
    # the old form without ':' was ambiguous (R3024 = 30 deg x 24 mm, or 302 deg x 4 mm?)
    for bad in ["R3024@0,0", "P10x20S0O0[R302@0,0]N3E0"]:
        try:
            ks.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_paper_cells_are_valid():
    for name, code, L, W, _ in CASES:
        assert ks.check(code, L, W) == [], name


INVALID = [                                      # (string, start of the expected message)
    ("K60@0", "the clamped ends are not joined"),
    ("C(-5,-5;5,-5;5,5;-5,5;-5,-5)", "a piece of"),            # closed loop: loose square
    ("V20@0,0;V20@0.5,0", "ligament of 0.50 mm"),
    ("V20@0,0;H20@0,0", "cuts[0] and cuts[1] cross"),
    ("V20@0,b+10.6", "cuts[0] leaves a 0.60 mm sliver"),
    ("H130@0,0", "cuts[0] reaches a clamped end"),
    ("X(-10,-10;10,-10;10,10;-10,10);+E(5,3,0)", "a piece of"),  # island in a window
    ("X(-10,-10;10,-10;10,10;-10,10);V30@0,0", "cuts[0] runs into holes[0]"),
    ("(K29.6@0){my}", "ligament of 0.80 mm"),
]


def test_invalid_designs_are_caught():
    assert ks.check("(K29@0){my}", 120, 60) == []               # 2 mm neck: fine
    for code, msg in INVALID:
        errs = ks.check(code, 120, 60)
        assert errs and errs[0].startswith(msg), (code, errs)


def test_lattice_motifs():
    k = ks.geometry_key
    # a slit written as a fraction of the cell or as a cut in mm is the same cut
    assert k("P5x40S1/2O0[V36@0,0]N20E4", 120, 60) == k("P5x40S1/2O0[C(0,-18;0,18)]N20E4", 120, 60)
    # the pitch chain as a lattice of pitch cells
    chain = "P120x200S0O0[(C(2,16;47.5,16;47.5,-16;2,-16)){mx};(C(0,-30;0,-11.25)){my}]N3E0"
    assert k(chain, 400, 60) == k(geo.pitch_chain_pattern(400, 60, 3, 120, 18.75, 32, 95, 4),
                                  400, 60)
    # re-anchoring the slits moves the free pieces with the lattice point
    for code in ["P5x40S1/2O3[V36@0,1/2;C(0,-5;1,5)]N20E4",
                 "P10x20S1/2O10[(C(-9,0;9,0);C(0,1;0,19))k0.5]N9E2",
                 "P8x16S1/2O0[+E(3,1.5,30);X(-3,-3;3,-3;3,3;-3,3)]N10E1"]:
        c = ks.canonical(code)
        assert ks.canonical(c) == c and k(c, 120, 60) == k(code, 120, 60), code
    # pieces in a motif cannot refer to the sheet's edges
    for bad in ["P5x40S1/2O0[K10@0]N3E1", "P5x40S1/2O0[C(0,b;0,5)]N3E1"]:
        try:
            ks.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    # plain slit lattices print exactly as grammar.py prints them
    code = "P3x83S1/3O41.5[V79.5@1/4,0]N13E2.5"
    assert ks.canonical(code) == grammar.encode(grammar.decode(code))


if __name__ == "__main__":
    for case in CASES:
        print(f"{case[0]:18s} {check(*case)}")
    test_mirror_images_share_a_key()
    test_equivalent_strings_share_a_key()
    test_bad_strings_are_rejected()
    print("all ok")
