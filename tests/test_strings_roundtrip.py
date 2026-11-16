"""Every design simulated in the paper must survive geometry -> string -> geometry
(scripts/check_strings.py): same pieces, largest distance below 0.01 mm, same geometry key."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_strings as C      # noqa: E402


def test_paper_designs_round_trip():
    for row in C.DESIGNS:
        r = C.check(*row)
        assert r["ok"], (r["name"], r["pieces"], r["ref_pieces"], r["distance"], r["key"])


def test_check_detects_errors():
    from kirigami import geometry as geo
    ref = geo.single_cut_pattern(182, 100, 80)
    assert not C.check("shorter", "V79@0,0", 182, 100, ref)["ok"]
    assert not C.check("moved", "V80@0.02,0", 182, 100, ref)["ok"]
