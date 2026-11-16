"""Options for squeezed hinges: neo-Hookean membrane, hinge-only refinement, equal load steps."""
import numpy as np

from kirigami import geometry as geo
from kirigami import spec as S


def _sq(x0, y0, a=2.0):
    return [(x0, y0), (x0 + a, y0), (x0 + a, y0 + a), (x0, y0 + a)]


def test_hinge_between_two_holes():
    # two squares 0.5 mm apart: one neck, centred between them
    H = geo.hinge_points(20, 10, holes=[_sq(5, 4), _sq(7.5, 4)], gap=0.8)
    assert len(H) >= 1
    assert np.all(np.abs(H[:, 0] - 7.25) < 0.25)               # midpoints of near pairs
    assert np.all((H[:, 1] >= 4 - 1e-9) & (H[:, 1] <= 6 + 1e-9))


def test_no_hinge_when_far_apart_or_touching():
    assert len(geo.hinge_points(20, 10, holes=[_sq(5, 4), _sq(10, 4)], gap=0.8)) == 0
    assert len(geo.hinge_points(20, 10, holes=[_sq(5, 4), _sq(6.5, 4)], gap=0.8)) == 0   # overlap


def test_hinge_at_sheet_edge():
    H = geo.hinge_points(20, 10, holes=[_sq(5, 0.4)], gap=0.8)
    assert len(H) >= 1 and np.all(np.abs(H[:, 1] - 0.2) < 1e-6)


def _spec(**kw):
    s = {"sheet": {"L": 20, "W": 10}, "pattern": {"holes": [_sq(5, 4)]}}
    for k, v in kw.items():
        s[k] = v
    return S.load(s)


def test_validate_new_options():
    assert not S.validate(_spec(material={"model": "neo_hookean"},
                                mesh={"h_hinge": 0.1, "hinge_gap": 0.8},
                                load={"Dmax": 1, "steps": 4, "spacing": "linear"}))
    assert any("material.model" in e for e in S.validate(_spec(material={"model": "rubber"})))
    assert any("load.spacing" in e for e in S.validate(_spec(load={"Dmax": 1, "steps": 4,
                                                                   "spacing": "log"})))
    assert any("h_hinge" in e for e in S.validate(_spec(mesh={"h_hinge": -1})))


def test_defaults_unchanged():
    s = _spec()
    assert s["material"].get("model", "svk") == "svk"
    assert s["load"].get("spacing", "geometric") == "geometric"
    assert s["mesh"].get("h_hinge") is None
