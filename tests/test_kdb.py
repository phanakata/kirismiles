"""kirigami/kdb.py: records validate, IDs are stable content hashes, and a legacy
generator converts to a string that cuts the same geometry."""
import pytest

pytest.importorskip("pydantic")

from kirigami import grammar, kirismiles as K
from kirigami.kdb import Design, FlatSheet, Material, Quantity, Specimen


def test_ids_are_content_hashes():
    a = Design.make_id(kirismiles="V80@0,0")
    assert a == Design.make_id(kirismiles="V80@0,0") != Design.make_id(kirismiles="V60@0,0")
    m = Material.make_id(3500, 0.38, "svk")
    assert m == Material.make_id(3500.0, 0.38, "svk") != Material.make_id(3500, 0.38, "neo_hookean")


def test_specimen_roundtrip():
    g = FlatSheet(L=120, W=60)
    did, mid = Design.make_id(kirismiles="V80@0,0"), Material.make_id(3500, 0.38, "svk")
    sp = Specimen(specimen_id=Specimen.make_id(did, "mm", g.model_dump(), 0.127, mid, None),
                  design_id=did, geometry=g, thickness=0.127, material_id=mid)
    again = Specimen.model_validate_json(sp.model_dump_json())
    assert again == sp and again.geometry.type == "flat"
    with pytest.raises(Exception):
        Specimen.model_validate({**sp.model_dump(), "colour": "red"})     # extra fields refused
    assert Quantity(value=1.0, unit="1").unit == "1"


def test_legacy_generator_to_string():
    from kirigami import spec as S
    spec = {"sheet": {"L": 120, "W": 60},
            "pattern": {"generator": "alternating", "params": {"b": 20, "g": 4, "d": 5, "n_rows": 12}}}
    code = K.canonical(grammar.encode(grammar.from_legacy("alternating", spec["pattern"]["params"], 60)))
    assert K.geometry_key(code, 120, 60) == K.geometry_key(S.pattern(spec), 120, 60)


def test_data_ref_npz_and_old_paths(tmp_path):
    import numpy as np
    from kirigami.kdb import Result, data_ref
    np.savez(tmp_path / "f.npz", X=np.zeros((5, 3)), eps0_0100=np.ones((5, 3), np.float32))
    r = data_ref("f.npz", tmp_path, array_descriptions={"X": "coords (mm)", "eps*": "u (mm)"})
    assert r.available and r.format == "npz" and len(r.sha256) == 64 and r.bytes > 0
    assert r.arrays["X"].shape == [5, 3] and r.arrays["X"].description == "coords (mm)"
    assert r.arrays["eps0_0100"].dtype == "float32" and r.arrays["eps0_0100"].description == "u (mm)"
    assert data_ref("missing.npz", tmp_path).available is False
    base = dict(result_id="kr-1", specimen_id="ks-1", kind="simulation", loading={},
                method={}, provenance={})
    old = Result.model_validate({**base, "data": {"curve": "a/b.json#experiment"}})   # kdb/0.1
    assert old.data["curve"].path == "a/b.json" and old.data["curve"].key == "experiment"
