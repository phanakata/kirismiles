"""The in-plane stiffness command (kirigami/api.py: stiffness, stiffness_batch).

The batch logic runs anywhere (the solve is replaced by a fake). The physics checks need the
solver and are skipped on the host; run them in the FEniCSx container:

    python -m pytest -q tests/test_api_stiffness.py
    ./run.sh bash -c "pip install -q pytest && python3 -m pytest -q tests/test_api_stiffness.py"
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kirigami import api            # noqa: E402
from kirigami import spec as S      # noqa: E402


def spec(name, code):
    return S.load({"name": name, "sheet": {"L": 120, "W": 60},
                   "pattern": {"generator": "kirismiles", "params": {"code": code}}})


def write(tmp_path, s):
    p = tmp_path / f"{s['name']}.json"
    S.save(s, p)
    return str(p)


def test_batch_keeps_order_and_isolates_failures(tmp_path, monkeypatch):
    def fake(s):
        if "boom" in s["name"]:
            raise RuntimeError("solver blew up")
        return dict(name=s["name"], analysis="stiffness", stiffness_N_per_mm=100.0, n_cells=1, seconds=0.0)
    monkeypatch.setattr(api, "stiffness", fake)
    paths = [write(tmp_path, spec("a", "V20@0,0")), str(tmp_path / "missing.json"),
             write(tmp_path, spec("boom", "V20@0,0")), write(tmp_path, spec("b", "H20@0,0"))]
    out = api.stiffness_batch(paths)
    assert out["analysis"] == "stiffness_batch"
    r = out["results"]
    assert [x["ok"] for x in r] == [True, False, False, True]
    assert r[0]["name"] == "a" and r[3]["name"] == "b" and r[0]["stiffness_N_per_mm"] == 100.0
    assert "solver blew up" in r[2]["errors"][0]
    json.dumps(out)                                   # the CLI prints it as one JSON line


# ------------------------------------------------------------------ with the solver (container only)

solver = pytest.mark.skipif(not __import__("importlib").util.find_spec("dolfinx"),
                            reason="needs the FEniCSx solver (run in the container)")


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RUNS", tmp_path)
    return tmp_path


@solver
def test_stiffness_physics(runs):
    s = {k: spec(k, c) for k, c in (("along", "H2@0,0"), ("across", "V30@0,0"), ("across_long", "V40@0,0"))}
    k = {n: api.stiffness(x)["stiffness_N_per_mm"] for n, x in s.items()}
    mat = s["along"]["material"]
    uncut = mat["E"] * mat["h"] * 60 / 120          # E h W / L, uniaxial stress estimate
    assert abs(k["along"] / uncut - 1) < 0.05        # a short slit along the pull changes almost nothing
    assert k["across_long"] < k["across"] < k["along"]   # cuts across the pull soften, longer more
    assert (runs / "along" / "stiffness.json").exists()


@solver
def test_stiffness_matches_buckle(runs):
    s = spec("same", "V30@0,0")
    assert api.stiffness(s)["stiffness_N_per_mm"] == pytest.approx(api.buckle(s)["stiffness_N_per_mm"], rel=1e-9)
