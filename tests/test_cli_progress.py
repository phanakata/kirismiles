"""Progress output of the command line: the solver's progress goes to stderr and to
log.txt, stdout keeps exactly one JSON line, and `kirismiles` runs the same CLI.
A small Python script stands in for the solver container, so no Docker is needed and
nothing is written to results/runs/."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import kirigami_cli  # noqa: E402

from kirigami import api  # noqa: E402

FAKE_OK = r"""
import json, sys
print("mesh: 100 triangles", file=sys.stderr, flush=True)
print(" step   Delta(mm)", file=sys.stderr, flush=True)
print("    1      0.1000", file=sys.stderr, flush=True)
print("some library chatter")
print(json.dumps({"name": "t", "analysis": "buckle", "files": ["results/runs/t/buckling.npz"]}))
"""
FAKE_FAIL = r"""
import sys
print("mesh: 100 triangles", file=sys.stderr, flush=True)
print("Traceback: something broke", file=sys.stderr, flush=True)
sys.exit(3)
"""


def fake_solver(monkeypatch, code):
    """Make in_container start `python -c code` instead of the container."""
    real = subprocess.Popen

    def popen(cmd, **kw):
        kw.pop("env", None)
        return real([sys.executable, "-c", code], **kw)
    monkeypatch.setattr(kirigami_cli.subprocess, "Popen", popen)


def test_print_every_reads_the_environment(monkeypatch):
    for value, expect in (("1", 1), ("5", 5), ("0", 0), ("-2", 1), ("x", 1)):
        monkeypatch.setenv("KIRISMILES_PRINT_EVERY", value)
        assert api.print_every() == expect
    monkeypatch.delenv("KIRISMILES_PRINT_EVERY")
    assert api.print_every() == 1


def test_progress_row_matches_the_header():
    row = api.progress_row(3, 0.25, 120.0, 0.5, 0.01, 0.002, 12.3)
    assert len(row.split()) == len(api.PROGRESS_HEADER.split()) == 7
    assert row.split()[2] == f"{100 * 0.25 / 120:.3f}"


def test_in_container_streams_progress_and_writes_the_log(monkeypatch, tmp_path, capsys):
    fake_solver(monkeypatch, FAKE_OK)
    res = kirigami_cli.in_container(["buckle", str(tmp_path / "spec.json")], echo=True)
    assert res["ok"] is True and res["analysis"] == "buckle"
    log = (tmp_path / "log.txt").read_text()
    assert "mesh: 100 triangles" in log and '"analysis": "buckle"' in log and "chatter" in log
    assert res["log"] == (tmp_path / "log.txt").as_posix()
    cap = capsys.readouterr()
    assert "mesh: 100 triangles" in cap.err and cap.out == ""     # progress on stderr only


def test_in_container_reports_a_failure_with_the_log(monkeypatch, tmp_path):
    fake_solver(monkeypatch, FAKE_FAIL)
    res = kirigami_cli.in_container(["simulate", str(tmp_path / "spec.json")])
    assert res["ok"] is False and "code 3" in res["error"]
    assert any("something broke" in ln for ln in res["log_tail"])
    assert "something broke" in (tmp_path / "log.txt").read_text()


def test_in_container_rejects_a_negative_print_every():
    with pytest.raises(ValueError, match="print_every"):
        kirigami_cli.in_container(["buckle", "x"], print_every=-1)


@pytest.mark.parametrize("quiet", [False, True])
def test_solver_command_prints_one_json_line(monkeypatch, tmp_path, capsys, quiet):
    fake_solver(monkeypatch, FAKE_OK)
    monkeypatch.setattr(kirigami_cli, "stage_spec", lambda spec: str(tmp_path / "spec.json"))
    argv = ["buckle", "specs/alternating.json"] + (["--quiet"] if quiet else [])
    assert kirigami_cli.main(argv) == 0
    cap = capsys.readouterr()
    lines = cap.out.strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["ok"] is True
    if quiet:
        assert cap.err == ""
    else:
        assert "Job started" in cap.err and "Job ended" in cap.err and "completed" in cap.err
        assert "mesh: 100 triangles" in cap.err and "buckling.npz" in cap.err


def test_kirismiles_command_runs_the_cli(monkeypatch, capsys):
    from kirigami import cli
    monkeypatch.setattr(sys, "argv", ["kirismiles", "validate", "specs/alternating.json"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
