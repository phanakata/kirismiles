"""Import of laser-cut files (kirigami/importer.py): file -> string -> geometry must give back
what was drawn."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_strings as C                       # noqa: E402
from kirigami import export as X                # noqa: E402
from kirigami import importer as I              # noqa: E402
from kirigami import kirismiles as ks           # noqa: E402


def test_paper_designs_through_svg_and_dxf(tmp_path):
    """Every design of the paper without solids, written by our exporter and read back."""
    for name, code, L, W, _ in C.DESIGNS:
        pat = ks.pattern(code, L, W)
        if pat["solids"]:
            continue                            # their exact export needs gmsh
        paths = X.cut_paths(L, W, pat, exact=False)
        for ext in ("svg", "dxf"):
            f = tmp_path / f"d.{ext}"
            X.write_svg(paths, L, W, f) if ext == "svg" else X.write_dxf(paths, f)
            r = I.import_file(f)
            assert (r["L"], r["W"]) == (L, W), (name, ext)
            assert C.counts(r["pattern"]) == C.counts(pat), (name, ext)
            assert I.deviation(pat, r["pattern"]) < 1e-3, (name, ext)


SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="5in" height="2.5in" viewBox="0 0 500 250">
<g transform="translate(10,5) scale(0.96)">
  <rect x="-10.4167" y="-5.2083" width="520.8333" height="260.4167"/></g>
<circle cx="100" cy="125" r="50"/>
<circle cx="100" cy="125" r="20"/>
<path d="M 250 60 C 300 60 300 190 350 190"/>
<path d="M 400 100 A 40 40 0 0 1 440 140"/>
</svg>"""


def test_curves_units_transforms_and_islands(tmp_path):
    f = tmp_path / "c.svg"
    f.write_text(SVG)
    r = I.import_file(f, tol=0.01)
    rep = r["report"]
    assert (r["L"], r["W"]) == (127.0, 63.5)                       # 5 x 2.5 in
    assert (rep["cuts"], rep["holes"], rep["solids"]) == (2, 1, 1)   # the small disc is an island
    for kind, radius in (("holes", 12.7), ("solids", 5.08)):
        P = np.asarray(r["pattern"][kind][0])
        assert np.allclose(P.mean(0), (25.4, 31.75), atol=0.01)
        rr = np.linalg.norm(P - (25.4, 31.75), axis=1)
        assert np.all(np.abs(rr - radius) < 0.002)                 # vertices on the circle
        mid = (P + np.roll(P, -1, 0)) / 2           # chords within tol, plus 3-decimal rounding
        assert radius - np.linalg.norm(mid - (25.4, 31.75), axis=1).min() <= 0.01 + 0.001


def test_dxf_entities(tmp_path):
    """LWPOLYLINE with a bulge (a half circle), CIRCLE and ARC, in inches."""
    def g(*pairs):
        return "\n".join(f"{c}\n{v}" for c, v in pairs)
    dxf = "\n".join([
        g((0, "SECTION"), (2, "HEADER"), (9, "$INSUNITS"), (70, 1), (0, "ENDSEC")),
        g((0, "SECTION"), (2, "ENTITIES")),
        g((0, "LWPOLYLINE"), (90, 4), (70, 1), (10, 0), (20, 0), (10, 4), (20, 0),
          (10, 4), (20, 2), (10, 0), (20, 2)),                                  # the sheet
        g((0, "LWPOLYLINE"), (90, 2), (70, 0), (10, 1), (20, 1), (42, 1), (10, 2), (20, 1)),
        g((0, "CIRCLE"), (10, 3), (20, 1), (40, 0.25)),
        g((0, "ARC"), (10, 2), (20, 1.5), (40, 0.3), (50, 0), (51, 90)),
        g((0, "ENDSEC"), (0, "EOF"))]) + "\n"
    f = tmp_path / "e.dxf"
    f.write_text(dxf)
    r = I.import_file(f, tol=0.005)
    rep = r["report"]
    assert (r["L"], r["W"]) == (101.6, 50.8)
    assert (rep["cuts"], rep["holes"], rep["solids"]) == (2, 1, 0)
    semi = max(r["pattern"]["cuts"], key=len)                       # the bulge: radius 0.5 in
    P = np.asarray(semi)
    assert np.allclose(np.linalg.norm(P - (38.1, 25.4), axis=1), 12.7, atol=0.002)
    assert P[:, 1].min() < 25.4 - 12.0                             # bulge 1 = clockwise below


def test_sheet_size_without_outline(tmp_path):
    f = tmp_path / "n.svg"
    f.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="50mm" '
                 'viewBox="0 0 100 50"><line x1="50" y1="10" x2="50" y2="40"/></svg>')
    try:
        I.import_file(f)
        raise AssertionError("expected a request for the sheet size")
    except ValueError as e:
        assert "L and W" in str(e)
    r = I.import_file(f, L=100, W=50)
    assert r["code"] == ks.canonical("V30@0,0") or ks.geometry_key(r["code"], 100, 50) == \
        ks.geometry_key("V30@0,0", 100, 50)


def test_chain_keeps_t_junctions_apart():
    a = np.array([[0, 0], [10, 0]]), np.array([[10, 0], [20, 0]]), np.array([[10, 0], [10, 5]])
    assert len(I.chain(list(a))) == 3                    # three ends meet: not joined
    assert len(I.chain(list(a[:2]))) == 1                # two ends meet: joined
    assert math.isclose(np.ptp(I.chain(list(a[:2]))[0][:, 0]), 20)
