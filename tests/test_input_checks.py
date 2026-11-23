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


# ---------------------------------------------------------------- run engine and database

def test_run_progress_survives_missing_or_half_written_history(tmp_path, monkeypatch):
    from kirigami import jobs
    monkeypatch.setattr(jobs, "RUNS", tmp_path)
    spec = S.load(GOOD)
    assert jobs.run_progress("none", spec) == (0.0, 0, 0.0)              # not written yet
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "history.npz").write_bytes(b"PK\x03\x04 truncated")
    assert jobs.run_progress("half", spec) == (0.0, 0, 0.0)              # caught mid-write


def test_run_engine_rejects_bad_arguments():
    from kirigami import jobs
    with pytest.raises(ValueError, match="Runner.submit"):
        jobs.run_now("simulate", S.load(GOOD))
    with pytest.raises(ValueError, match="label"):
        jobs.Runner("web app")                                          # names docker containers
    with pytest.raises(ValueError, match="eps_y"):
        jobs.analyse_run("x", eps_y=0)
    assert jobs.safe_name("my design!") == "my_design" and jobs.safe_name(None) == "design"


def test_database_records_check_counts_and_hashes():
    pytest.importorskip("pydantic")
    from pydantic import ValidationError

    from kirigami import kdb
    assert kdb.Loading(steps=40).steps == 40
    with pytest.raises(ValidationError):
        kdb.Loading(steps=0)
    with pytest.raises(ValidationError):
        kdb.Method(seconds=-1.0)
    with pytest.raises(ValidationError):
        kdb.ArrayInfo(shape=[3, -1], dtype="float64")
    with pytest.raises(ValidationError):
        kdb.DataRef(path="a.npz", sha256="not-a-hash")
    with pytest.raises(ValueError, match="design ID"):
        kdb.Design.make_id()


# ---------------------------------------------------------------- cut-file import and provenance

def test_importer_checks_its_settings(tmp_path):
    from kirigami import importer as I
    svg = tmp_path / "a.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="120mm" height="60mm" viewBox="0 0 120 60">'
                   '<rect x="0" y="0" width="120" height="60"/><line x1="60" y1="10" x2="60" y2="50"/></svg>')
    assert I.import_file(svg)["code"]                                     # a valid file still imports
    for kw, expect in ((dict(tol=0), "tol"), (dict(digits=9), "digits"), (dict(L=120), "both L and W"),
                       (dict(L=-1, W=60), "L = -1")):
        with pytest.raises(ValueError, match=expect):
            I.import_file(svg, **kw)
    with pytest.raises(ValueError, match="step"):
        I.deviation({"cuts": []}, {"cuts": []}, step=0)
    with pytest.raises(ValueError, match="start with a command"):
        I._path("10 10 L 20 20", 0.01)


def test_provenance_needs_an_existing_run_folder(tmp_path):
    from kirigami import provenance
    with pytest.raises(FileNotFoundError):
        provenance.write(tmp_path / "missing")
    assert len(provenance.source_hash()) == 40
