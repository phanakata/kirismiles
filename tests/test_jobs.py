"""The shared run engine kirigami/jobs.py, with a fake solver (no Docker needed).

    python -m pytest -q tests/test_jobs.py
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kirigami import jobs          # noqa: E402
from kirigami import spec as S     # noqa: E402

SPEC = S.load({"name": "t", "sheet": {"L": 120, "W": 60},
               "pattern": {"generator": "kirismiles", "params": {"code": "V40@0,0"}}})


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """jobs.py pointed at a temporary results/runs, with a fake solver that
    takes `delay` seconds and records how many solves overlapped."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(jobs, "RUNS", runs)
    monkeypatch.setattr(jobs, "LOCK_FILE", runs / ".solver.lock")
    monkeypatch.setattr(jobs, "docker_ok", lambda: None)
    monkeypatch.setattr(jobs, "foreign_solver_running", lambda: False)
    monkeypatch.setattr(jobs, "STALE_AFTER", 2.0)
    state = dict(active=0, peak=0, delay=0.3, ok=True)
    mu = threading.Lock()

    class CLI:
        @staticmethod
        def stage_spec(spec):
            d = runs / spec["name"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "spec.json").write_text(json.dumps(spec))
            return str(d / "spec.json")

        @staticmethod
        def run_dir(spec):
            return runs / spec["name"]

        @staticmethod
        def in_container(args, name=None):
            with mu:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(state["delay"])
            with mu:
                state["active"] -= 1
            if not state["ok"]:
                return dict(ok=False, error="solver exited with code 1")
            spec = json.loads(Path(args[1]).read_text())
            summary = dict(name=spec["name"], status="completed", strain_final=0.1)
            (runs / spec["name"] / "summary.json").write_text(json.dumps(summary))
            return dict(ok=True, **summary)

    monkeypatch.setattr(jobs, "_scripts", lambda: (CLI, None, None))
    monkeypatch.setattr(jobs, "finish_run", lambda name, res: res.update(files=["gif"]) or res)
    return state


def wait(runner, jid, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = runner.get(jid)
        if job["state"] not in jobs.ACTIVE:
            return job
        time.sleep(0.05)
    raise TimeoutError(jid)


def test_lock_excludes_threads_and_processes(tmp_path):
    path = tmp_path / "x.lock"
    a, b = jobs.SolverLock(path), jobs.SolverLock(path)
    assert a.try_acquire()
    assert not b.try_acquire()                       # same process, other handle
    a.release()
    assert b.try_acquire()
    b.release()
    holder = subprocess.Popen(                       # another process holds it
        [sys.executable, "-c", "import sys, time; sys.path.insert(0, sys.argv[1]);"
         "from kirigami import jobs; l = jobs.SolverLock(sys.argv[2]);"
         "assert l.try_acquire(); print('held', flush=True); time.sleep(30)",
         str(ROOT), str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert not a.try_acquire()
    finally:
        holder.kill()
        holder.wait()
    t0 = time.time()                                 # the OS frees it when the holder dies
    while not a.try_acquire():
        assert time.time() - t0 < 5
        time.sleep(0.05)
    a.release()


def test_a_job_runs_and_its_record_is_shared(fake):
    r = jobs.Runner("t")
    job = r.submit(SPEC)
    assert job["state"] == "queued" and job["name"].startswith("t_")
    done = wait(r, job["id"])
    assert done["state"] == "done" and done["result"]["files"] == ["gif"]
    other = jobs.Runner("other")                     # another front end reads it from disk
    assert other.get(job["id"])["state"] == "done"
    assert any(j["id"] == job["id"] for j in other.list())


def test_one_solve_at_a_time_across_runners(fake):
    a, b = jobs.Runner("a"), jobs.Runner("b")        # e.g. the web app and the MCP server
    ids = [(a, a.submit(SPEC)["id"]) for _ in range(2)] + [(b, b.submit(SPEC)["id"])]
    for r, jid in ids:
        assert wait(r, jid)["state"] == "done"
    assert fake["peak"] == 1


def test_failed_and_cancelled_jobs(fake):
    r = jobs.Runner("t")
    fake["ok"] = False
    bad = wait(r, r.submit(SPEC)["id"])
    assert bad["state"] == "failed" and "code 1" in bad["message"]
    fake.update(ok=True, delay=1.0)
    first, second = r.submit(SPEC), r.submit(SPEC)
    assert r.cancel(second["id"])["state"] == "cancelled"
    assert wait(r, first["id"])["state"] == "done"
    assert r.get(second["id"])["state"] == "cancelled"


def test_records_left_by_an_exited_process_are_reconciled(fake, monkeypatch):
    runs = jobs.RUNS
    monkeypatch.setattr(jobs, "container_running", lambda name: False)

    def orphan(name, state, summary):
        d = runs / name
        d.mkdir()
        (d / "spec.json").write_text(json.dumps(dict(SPEC, name=name)))
        if summary:
            (d / "summary.json").write_text(json.dumps({"status": "completed"}))
        rec = dict(id=name[-8:], name=name, state=state, message="", progress=0, steps=0,
                   delta=0, created=time.time() - 100, started=time.time() - 100,
                   finished=None, beat=time.time() - 100, container="gone", result=None)
        (d / "job.json").write_text(json.dumps(rec))
        return rec["id"]

    r = jobs.Runner("t")
    finished = orphan("run_aaaaaaaa", "running", summary=True)
    died = orphan("run_bbbbbbbb", "running", summary=False)
    never = orphan("run_cccccccc", "queued", summary=False)
    assert r.get(finished)["state"] == "done"        # finished while nobody watched
    assert r.get(died)["state"] == "interrupted"
    assert r.get(never)["state"] == "interrupted"
    monkeypatch.setattr(jobs, "container_running", lambda name: True)
    still = orphan("run_dddddddd", "running", summary=False)
    assert r.get(still)["state"] == "running"        # its container is still solving


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_stored_runs_are_found_by_design_not_by_spelling(tmp_path, monkeypatch):
    """find_runs matches on the geometry key: the same two slits written with a
    mirror, or listed one by one, are one design; other designs are not found."""
    runs = tmp_path / "runs"
    monkeypatch.setattr(jobs, "RUNS", runs)

    def spec(code, h=0.127, name="r"):
        return S.load({"name": name, "sheet": {"L": 120, "W": 60}, "material": {"h": h},
                       "pattern": {"generator": "kirismiles", "params": {"code": code}}})

    def store(name, sp, status="completed"):
        d = runs / name
        d.mkdir(parents=True)
        S.save(sp, d / "spec.json")
        (d / "summary.json").write_text(json.dumps(dict(analysis="simulate", status=status,
                                                        strain_final=0.1, uz_max_final_mm=3.0)))

    store("a", spec("(V40@6,0){mx}"))
    store("b", spec("V40@-6,0;V40@6,0", h=0.2))
    store("c", spec("V40@0,0"))
    store("d", spec("(V40@6,0){mx}"), status="stopped early: Newton failed")
    found = {r["name"]: r for r in jobs.find_runs(spec("V40@6,0;V40@-6,0"))}
    assert set(found) == {"a", "b", "d"}
    assert found["a"]["completed"] and found["a"]["same_settings"]
    assert found["b"]["completed"] and not found["b"]["same_settings"]      # other thickness
    assert not found["d"]["completed"]
    assert (runs / "a" / "geometry_key.txt").read_text() == jobs.design_key(spec("(V40@6,0){mx}"))
