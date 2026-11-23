"""Malformed or out-of-range input is reported with a readable message, never a crash
deep inside the code: spec.validate, the notation parsers, kirismiles.check, the
database records and the STL export."""
import pytest

from kirigami import api, grammar
from kirigami import kirismiles as K
from kirigami import spec as S

GOOD = {"name": "t", "sheet": {"L": 120, "W": 60},
        "pattern": {"generator": "kirismiles", "params": {"code": "P5x45S1/2O0[V42@0,0]N18E3"}}}


def test_good_spec_has_no_problems():
    assert S.validate(S.load(GOOD)) == []


@pytest.mark.parametrize("spec, expect", [
    ("not a dict", "JSON object"),
    ({"sheet": {"L": 120, "W": 60}}, "pattern is required"),            # used to raise KeyError
    ({**GOOD, "sheet": None}, "sheet is required"),
    ({**GOOD, "sheet": {"L": "120", "W": 60}}, "required numbers"),
    ({**GOOD, "sheet": {"L": 0, "W": 60}}, "positive"),
    ({**GOOD, "material": "PET"}, "must be an object"),
    ({**GOOD, "material": {"E": "3500"}}, "must be numbers"),           # used to raise TypeError
    ({**GOOD, "material": {"h": -0.1}}, "h > 0"),
    ({**GOOD, "mesh": {"h_cut": 0}}, "h_cut"),
    ({**GOOD, "load": {"steps": 0}}, "load.steps"),
    ({**GOOD, "load": {"steps": 2.5}}, "load.steps"),
    ({**GOOD, "boundary": {"type": "glued"}}, "boundary.type"),
    ({**GOOD, "boundary": {"in_plane": "yes"}}, "in_plane"),
])
def test_validate_reports_instead_of_raising(spec, expect):
    errs = S.validate(spec)
    assert errs and any(expect in e for e in errs), errs


def test_load_rejects_a_section_that_is_not_an_object():
    with pytest.raises(ValueError, match="must be an object"):
        S.load({**GOOD, "mesh": [1, 2]})


@pytest.mark.parametrize("code, expect", [
    ("P0x45S1/2O0[V42@0,0]N18E3", "px > 0"),
    ("P5x45S1/2O0[V42@0,0]N0E3", "at least 1"),
    ("P5x45S1/2O0[V0@0,0]N18E3", "length above zero"),
    ("P5x45S1/2O0[V42@1/25,0]N18E3", "denominator above 24"),
    ("P5x45S1/2O0[Q42@0,0]N18E3", "bad slit"),
])
def test_grammar_rejects_out_of_range_designs(code, expect):
    with pytest.raises(ValueError, match=expect):
        grammar.decode(code)


def test_y_coordinate_reference_is_checked():
    assert K.Y("b", 2.0).at(60) == -28.0
    with pytest.raises(ValueError, match="'c', 'b' or 't'"):
        K.Y("x", 1.0)


@pytest.mark.parametrize("L, W, gap, expect", [(0, 60, 1.0, "L > 0"), (120, -1, 1.0, "W > 0"),
                                               (120, 60, 0.0, "min_gap")])
def test_check_reports_bad_sheet_or_gap(L, W, gap, expect):
    errs = K.check("V80@0,0", L, W, min_gap=gap)
    assert len(errs) == 1 and expect in errs[0]


def test_check_reports_a_string_that_does_not_parse():
    errs = K.check("V80@0,", 120, 60)
    assert errs and not K.is_valid("V80@0,", 120, 60)


@pytest.mark.parametrize("thickness, kerf, expect", [(0.0, 0.1, "thickness"), (1.0, -0.1, "kerf")])
def test_stl_export_checks_its_sizes(thickness, kerf, expect):
    with pytest.raises(ValueError, match=expect):
        api.export_stl(S.load(GOOD), thickness, kerf)


def test_database_records_refuse_impossible_sizes():
    pytest.importorskip("pydantic")
    from pydantic import ValidationError

    from kirigami.kdb import Cylinder, FlatSheet
    assert FlatSheet(L=120, W=60).L == 120
    for bad in ({"L": 0, "W": 60}, {"L": 120, "W": -5}):
        with pytest.raises(ValidationError):
            FlatSheet(**bad)
    with pytest.raises(ValidationError):
        Cylinder(circumference=0)
    with pytest.raises(ValidationError):
        Cylinder(circumference=100, n_around=0)
