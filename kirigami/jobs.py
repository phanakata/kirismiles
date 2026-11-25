"""Solver jobs on the host: a queue, one solve at a time, and job records on disk.

This is the run engine behind every front end: the web app (webapp/server.py),
the MCP server (mcp_server/server.py), and batch scripts. They share three
guarantees:

- **One solve at a time on the machine.** A solve holds an exclusive lock on
  results/runs/.solver.lock. The operating system releases the lock when the
  holding process exits, so a crash never leaves it stuck. A solve also waits
  while any other container labelled kirigami=1 is running, for tools that do
  not take the lock.
- **Job records on disk.** Each job is saved as results/runs/<name>/job.json and
  rewritten as it progresses. Any process can read a job's state, including
  after the process that started it has exited.
- **Runs outlive their server.** If the process that started a simulation exits
  (for example when an MCP session ends), the solver container keeps running.
  Reading the job later reconciles the record with what is actually there:
  it reports the container as still running, finishes the job (GIF and
  stress-strain plot) once the solver has written summary.json, or marks it
  interrupted.

Usage:

    from kirigami import jobs
    runner = jobs.Runner()                    # starts one worker thread
    job = runner.submit(spec)                 # -> {"id", "name", "state", ...}
    runner.get(job["id"])                     # state, progress, result
    runner.cancel(job["id"])
    jobs.run_now("buckle", spec)              # short solver commands, if the solver is free

Host side only: it starts the FEniCSx container through scripts/kirigami_cli.py
and post-processes with scripts/make_gif.py and scripts/compare_designs.py.
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from types import ModuleType
from typing import IO, Any

import numpy as np

from kirigami._types import Spec, Summary
from kirigami.stiffness import from_history as stiffness_fit

Job = dict[str, Any]        # a job record: id, name, state, message, progress, ..., result, spec

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
LOCK_FILE = RUNS / ".solver.lock"
STALE_AFTER = 60.0          # s: a queued or running record not refreshed for this long is orphaned
ACTIVE = ("queued", "running")


def _scripts() -> tuple[ModuleType, ModuleType, ModuleType]:
    """The host-side helpers in scripts/ (the CLI, GIF rendering, run analysis)."""
    p = str(ROOT / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)
    import compare_designs
    import kirigami_cli
    import make_gif
    return kirigami_cli, make_gif, compare_designs


def rel(path: str | os.PathLike[str]) -> str:
    """A path inside the repository, relative to its root, with forward slashes."""
    return Path(path).resolve().relative_to(ROOT).as_posix()


# ------------------------------------------------------------- the solver lock

class SolverLock:
    """Exclusive lock on LOCK_FILE, held for the length of a solve.

    The OS drops the lock when the process exits, so it is never left stale.
    Within one process, every acquire opens its own handle, so two threads
    exclude each other too."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path or LOCK_FILE)
        self.fh: IO[str] | None = None

    def try_acquire(self) -> bool:
        """Take the lock if it is free; never waits."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self.fh = fh
        return True

    def release(self) -> None:
        if self.fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.fh.close()
            self.fh = None

    def __enter__(self) -> SolverLock:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def docker_ok() -> str | None:
    """None if Docker answers, otherwise a readable reason."""
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        return "Docker is not installed (the solver runs in the dolfinx/dolfinx container)"
    except subprocess.TimeoutExpired:
        return "Docker did not answer within 20 s"
    if r.returncode != 0:
        return "Docker is not running: start Docker Desktop and try again"
    return None


def _docker_ps(*filters: str) -> bool:
    """True if `docker ps` lists a container matching every filter; False also when
    Docker is missing or does not answer (docker_ok() reports why)."""
    args = ["docker", "ps", "-q"] + [a for f in filters for a in ("--filter", f)]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):     # OSError includes Docker not installed
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def foreign_solver_running() -> bool:
    """True if a solver container is running that does not hold the lock
    (for example one started by a tool that predates it)."""
    return _docker_ps("label=kirigami=1")


def container_running(name: str) -> bool:
    """True if the container with exactly this name is running."""
    return _docker_ps(f"name=^{name}$")


RUN_NOW_COMMANDS = ("buckle", "stiffness", "export", "stl", "mesh")


def run_now(cmd: str, spec: Spec, *extra: object) -> Summary:
    """Run a short solver command (buckle, export, stl, mesh) right away, if the
    solver is free. Returns the command's JSON result, with ok=False and a
    reason when it cannot run."""
    if cmd not in RUN_NOW_COMMANDS:
        raise ValueError(f"run_now runs {list(RUN_NOW_COMMANDS)}; a full simulation goes through Runner.submit")
    cli, _, _ = _scripts()
    why = docker_ok()
    if why:
        return dict(ok=False, errors=[why])
    lock = SolverLock()
    if not lock.try_acquire() or foreign_solver_running():
        lock.release()
        return dict(ok=False, errors=["the solver is busy with another run; "
                                      "try again when it finishes"])
    with lock:
        return cli.in_container([cmd, cli.stage_spec(spec), *map(str, extra)])


# --------------------------------------------------------------- run analysis

def run_progress(name: str, spec: Spec) -> tuple[float, int, float]:
    """(fraction of the load ramp done, steps saved, current Delta) from the
    history the solver saves after every step; (0, 0, 0) while there is none yet."""
    try:
        with np.load(RUNS / name / "history.npz") as d:
            D, Dc = d["Delta"], float(d["Delta_c"])
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
        return 0.0, 0, 0.0              # not written yet, or caught half-written
    if len(D) < 2:
        return 0.02, len(D), float(D[-1]) if len(D) else 0.0
    Dmax = spec["load"]["Dmax"]
    if not (Dc > 0 and Dmax > 0.5 * Dc):
        return 0.0, len(D), float(D[-1])
    frac = np.log(D[-1] / (0.5 * Dc)) / np.log(Dmax / (0.5 * Dc))
    return float(np.clip(frac, 0, 1)), len(D), float(D[-1])


def analyse_run(name: str, eps_y: float = 0.02) -> dict[str, Any]:
    """Stress-strain plot (stress_strain.png) and stretchability of a finished run.
    eps_y: yield strain of the material (> 0)."""
    if not eps_y > 0:
        raise ValueError(f"the yield strain eps_y must be positive, not {eps_y}")
    _, _, C = _scripts()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = C.analyse(str(RUNS / name / "history.npz"), eps_y)
    if r is None:
        return {}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.1))
    a1.plot(r["eps"] * 100, r["sigma"], color="#2a78d6", lw=1.8)
    a1.set(xlabel="applied strain Δ/L (%)", ylabel="nominal stress F/(Wh) (MPa)",
           title="Stress–strain")
    a2.semilogy(r["eps"] * 100, r["eloc"], color="#e34948", lw=1.8)
    a2.axhline(eps_y, color="#0b0b0b", ls="--", lw=1)
    a2.set(xlabel="applied strain Δ/L (%)", ylabel="peak local strain",
           title=f"Local strain (yield {eps_y:.0%})")
    for ax in (a1, a2):
        ax.grid(color="#e4e3df", lw=0.6)
    fig.tight_layout()
    out = RUNS / name / "stress_strain.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    out_d = dict(stretchability=r["stretch"], stretchability_note=r["note"],
                 stress_at_stretchability_MPa=r["sigma_star"], initial_modulus_MPa=r["E_eff"],
                 plot=rel(out))
    try:                                   # stiffness before and after buckling (N/mm)
        fit = stiffness_fit(RUNS / name / "history.npz")
        out_d.update(stiffness_before_buckling_N_per_mm=fit["k_before_N_per_mm"],
                     stiffness_after_buckling_N_per_mm=fit["k_after_N_per_mm"],
                     stiffness_fit_range_pct=[fit["fit_from_pct"], fit["fit_to_pct"]])
    except (KeyError, ValueError):         # no buckling stretch stored, or a run under 1 % strain
        pass
    return out_d


def finish_run(name: str, res: Summary) -> Summary:
    """Render the GIF and the analysis of a finished simulation; adds them to res."""
    _, make_gif, _ = _scripts()
    d = RUNS / name
    make_gif.make(name, src=d / "history.npz", dst=d / "deformation.gif")
    res["analysis"] = analyse_run(name)
    res["files"] = [rel(d / f) for f in ("spec.json", "provenance.json", "summary.json",
                                         "history.npz", "deformation.gif", "stress_strain.png")
                    if (d / f).exists()]
    prov = d / "provenance.json"                  # written by kirigami_cli.stage_spec
    if prov.exists():
        p = json.loads(prov.read_text())
        res["provenance"] = {k: p.get(k) for k in ("code_version", "source_hash", "image", "image_id")}
    return res


# ------------------------------------------------- results stored by design

SETTINGS = ("sheet", "material", "mesh", "boundary")      # with the load, what decides a result


def design_key(spec: Spec) -> str:
    """Geometry key of a spec's pattern on its sheet (kirismiles.geometry_key):
    the same for any two specs that cut the same sheet, however written."""
    from kirigami import kirismiles
    from kirigami import spec as S
    return kirismiles.geometry_key(S.pattern(spec), spec["sheet"]["L"], spec["sheet"]["W"])


def _settings(spec: Spec) -> dict[str, Any]:
    out: dict[str, Any] = {k: spec.get(k) for k in SETTINGS}
    load = spec.get("load", {})
    out["load"] = {k: load.get(k) for k in ("Dmax", "steps", "stop_local_strain")}
    if load.get("spacing", "geometric") != "geometric":       # older runs have no spacing key
        out["load"]["spacing"] = load["spacing"]
    out["imperfection_sign"] = spec.get("imperfection_sign", 1.0)
    out["ell"] = spec.get("ell")
    return json.loads(json.dumps(out))                    # tuples and lists compare equal


def _run_key(d: Path) -> str:
    """The geometry key of a stored run, computed once and kept beside it."""
    from kirigami import spec as S
    f = d / "geometry_key.txt"
    if f.exists():
        return f.read_text().strip()
    key = design_key(S.load(d / "spec.json"))
    try:
        f.write_text(key)
    except OSError:
        pass
    return key


def find_runs(spec: Spec) -> list[dict[str, Any]]:
    """Stored simulations of the same design as `spec` (same geometry key),
    newest first. Each entry says whether the run completed and whether it
    used the same sheet, material, mesh, boundary and load as `spec`, in which
    case its result can stand for a new solve."""
    from kirigami import spec as S
    key, want = design_key(spec), _settings(spec)
    out: list[dict[str, Any]] = []
    dirs = [d for d in RUNS.glob("*") if (d / "spec.json").exists() and (d / "summary.json").exists()]
    for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True):
        try:
            if _run_key(d) != key:
                continue
            sp = S.load(d / "spec.json")
            summ = json.loads((d / "summary.json").read_text())
        except (OSError, ValueError, KeyError, TypeError):      # an unreadable or outdated record
            continue
        if summ.get("analysis") != "simulate":
            continue
        out.append(dict(name=d.name, geometry_key=key, status=summ.get("status"),
                        completed=str(summ.get("status", "")).startswith("completed"),
                        same_settings=_settings(sp) == want,
                        thickness_mm=sp["material"]["h"], boundary=sp["boundary"],
                        max_stretch=summ.get("strain_final"),
                        max_out_of_plane_mm=summ.get("uz_max_final_mm")))
    return out


# ------------------------------------------------------------------- the queue

def safe_name(name: str | None) -> str:
    """A run folder name: letters, digits, _ and -, at most 40 characters."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name or "design").strip("_")[:40] or "design"


def unique_name(name: str | None) -> str:
    """name_<MMDD_HHMMSS>, with a suffix if that run folder already exists."""
    base = f"{safe_name(name)}_{time.strftime('%m%d_%H%M%S')}"
    out, k = base, 1
    while (RUNS / out).exists():
        k += 1
        out = f"{base}_{k}"
    return out


def _save(job: Job) -> None:
    d = RUNS / job["name"]
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"job.{uuid.uuid4().hex[:6]}.tmp"         # one per write: threads may save at once
    tmp.write_text(json.dumps({k: v for k, v in job.items() if k != "spec"}, indent=2))
    os.replace(tmp, d / "job.json")


def _find(job_id: str | None) -> Job | None:
    if not re.fullmatch(r"[0-9a-f]{8}", job_id or ""):
        return None
    for f in RUNS.glob("*/job.json"):
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if rec.get("id") == job_id:
            return rec
    return None


class Runner:
    """A job queue served by one worker thread in this process."""

    def __init__(self, label: str = "jobs") -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError(f"the label names containers: letters, digits, _ and - only, not {label!r}")
        self.label = label
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue[str] = queue.Queue()
        self.mu = threading.Lock()
        threading.Thread(target=self._work, daemon=True, name=f"{label}-worker").start()
        threading.Thread(target=self._heartbeat, daemon=True, name=f"{label}-beat").start()

    # -- public
    def submit(self, spec: Spec, unique: bool = True) -> Job:
        """Queue a full simulation. The spec is saved with the job; its name
        gets a timestamp unless unique=False."""
        spec = dict(spec)
        spec["name"] = unique_name(spec.get("name")) if unique else safe_name(spec.get("name"))
        jid = uuid.uuid4().hex[:8]
        job = dict(id=jid, name=spec["name"], state="queued", message="waiting in queue",
                   progress=0.0, steps=0, delta=0.0, created=time.time(), started=None,
                   finished=None, beat=time.time(), container=f"kirigami-{self.label}-{jid}",
                   result=None, spec=spec)
        cli, _, _ = _scripts()
        cli.stage_spec(spec)                       # the spec is on disk from the start
        with self.mu:
            self.jobs[jid] = job
        _save(job)
        self.q.put(jid)
        return self._public(job)

    def get(self, job_id: str) -> Job | None:
        """The job's current state, from this process or from disk."""
        with self.mu:
            job = self.jobs.get(job_id)
        if job is not None:
            return self._public(job)
        rec = _find(job_id)
        return None if rec is None else self._reconcile(rec)

    def cancel(self, job_id: str) -> Job | None:
        """Cancel a queued or running job (stops its container); None if there is no such job."""
        with self.mu:
            job = self.jobs.get(job_id)
        if job is None:
            job = _find(job_id)
            if job is None:
                return None
        if job["state"] in ACTIVE:
            was_running = job["state"] == "running"
            job.update(state="cancelled", message="cancelled", finished=time.time())
            if was_running:
                subprocess.run(["docker", "stop", job["container"]], capture_output=True)
            _save(job)
        return self._public(job)

    def list(self, limit: int = 20) -> list[Job]:
        """Recent jobs from every process, newest first (at most limit, >= 1)."""
        if limit < 1:
            raise ValueError(f"limit must be at least 1, not {limit}")
        recs: list[Job] = []
        for f in RUNS.glob("*/job.json"):
            try:
                recs.append(json.loads(f.read_text()))
            except (OSError, ValueError):
                pass
        recs.sort(key=lambda r: r.get("created", 0), reverse=True)
        return [self.get(r["id"]) or r for r in recs[:limit]]

    # -- internals
    @staticmethod
    def _public(job: Job) -> Job:
        return {k: v for k, v in job.items() if k not in ("spec", "beat")}

    def _heartbeat(self) -> None:
        """Refresh the records of this process's active jobs, so other
        processes can tell them from orphans."""
        while True:
            time.sleep(STALE_AFTER / 4)
            with self.mu:
                active = [j for j in self.jobs.values() if j["state"] in ACTIVE]
            for job in active:
                try:                               # cancelled from another process?
                    disk = json.loads((RUNS / job["name"] / "job.json").read_text())
                except (OSError, ValueError):
                    disk = {}
                if disk.get("state") == "cancelled":
                    was_running = job["state"] == "running"
                    job.update(state="cancelled", message="cancelled",
                               finished=disk.get("finished") or time.time())
                    if was_running:
                        subprocess.run(["docker", "stop", job["container"]], capture_output=True)
                    continue
                job["beat"] = time.time()
                _save(job)

    def _reconcile(self, rec: Job) -> Job:
        """Bring a record left by another process up to date."""
        if rec["state"] not in ACTIVE or time.time() - rec.get("beat", 0) < STALE_AFTER:
            return self._public(rec)
        d = RUNS / rec["name"]
        summary = d / "summary.json"
        if rec["state"] == "queued":
            rec.update(state="interrupted", message="its server exited before the run started",
                       finished=time.time())
        elif container_running(rec["container"]):
            spec = json.loads((d / "spec.json").read_text())
            rec["progress"], rec["steps"], rec["delta"] = run_progress(rec["name"], spec)
            rec["message"] = "still solving after its server exited; check again later"
            return self._public(rec)
        elif summary.exists() and summary.stat().st_mtime >= (rec.get("started") or 0):
            res = dict(ok=True, **json.loads(summary.read_text()))
            try:
                finish_run(rec["name"], res)
            except Exception as e:      # the run itself is done; report a failed GIF or plot with it
                res["analysis_error"] = f"{type(e).__name__}: {e}"
            rec.update(state="done", message=res.get("status", "completed"), progress=1.0,
                       result=res, finished=time.time())
        else:
            rec.update(state="interrupted", finished=time.time(),
                       message="the solver stopped without finishing; history.npz has the "
                               "steps it saved")
        rec["beat"] = time.time()
        _save(rec)
        return self._public(rec)

    def _work(self) -> None:
        while True:
            jid = self.q.get()
            job = self.jobs[jid]
            lock = SolverLock()
            try:
                if job["state"] == "cancelled":
                    continue
                why = docker_ok()
                if why:
                    job.update(state="failed", message=why, finished=time.time())
                    continue
                while not lock.try_acquire() or foreign_solver_running():
                    lock.release()
                    job["message"] = "waiting for another solver run to finish"
                    time.sleep(5)
                    if job["state"] == "cancelled":
                        break
                if job["state"] == "cancelled":
                    continue
                self._solve(job)
            except Exception as e:      # last resort: the worker thread must survive any job's failure
                job.update(state="failed", message=f"{type(e).__name__}: {e}",
                           result=dict(trace=traceback.format_exc()[-2000:]),
                           finished=time.time())
            finally:
                lock.release()
                job["beat"] = time.time()
                _save(job)

    def _solve(self, job: Job) -> None:
        cli, _, _ = _scripts()
        spec = job["spec"]
        job.update(state="running", message="meshing and solving", started=time.time())
        _save(job)
        done = threading.Event()

        def watch() -> None:
            while not done.is_set():
                job["progress"], job["steps"], job["delta"] = run_progress(spec["name"], spec)
                time.sleep(3)
        threading.Thread(target=watch, daemon=True).start()
        try:
            res = cli.in_container(["simulate", cli.stage_spec(spec)], name=job["container"])
        finally:
            done.set()
        if job["state"] == "cancelled":
            return
        if not res.get("ok"):
            job.update(state="failed", message=res.get("error", "solver failed"), result=res,
                       finished=time.time())
            return
        job.update(message="rendering the animation", progress=1.0)
        finish_run(spec["name"], res)
        job.update(state="done", message=res.get("status", "completed"), result=res,
                   finished=time.time())
