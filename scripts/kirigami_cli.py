"""One command-line entry point for designing, simulating and fabricating
kirigami sheets. Meant for people and for agents: every command prints a single
JSON line on stdout, and all outputs land in results/runs/<name>/ next to a copy
of the spec that produced them.

  python scripts/kirigami_cli.py generators                 # available pattern generators
  python scripts/kirigami_cli.py validate  specs/x.json     # cheap checks (host only)
  python scripts/kirigami_cli.py preview   specs/x.json     # pattern.png (host only)
  python scripts/kirigami_cli.py import    cut.svg [--spec specs/x.json]   # a laser-cut file as a string
  python scripts/kirigami_cli.py export    specs/x.json     # cut.svg + cut.dxf for a laser cutter
  python scripts/kirigami_cli.py mesh      specs/x.json     # the FEM mesh: .msh, .inp, .bdf, .vtk, .xdmf
  python scripts/kirigami_cli.py stl       specs/x.json --thickness 0.5 --kerf 0.4
  python scripts/kirigami_cli.py buckle    specs/x.json     # seconds: stiffness, Delta_c, F_c
  python scripts/kirigami_cli.py simulate  specs/x.json     # minutes: post-buckling + GIF

buckle / simulate / stl / mesh (and export of patterns with solids) run in the FEniCSx
Docker container automatically; the rest needs only numpy + matplotlib.

After `pip install -e .` the same commands are available as `kirismiles <command>`.

Progress: the solver commands print "Job started" / "Job ended" lines, the solver's
progress (for simulate, one row per load step) and the files
written, all on stderr; --print-every N prints every N-th step, --quiet only the JSON line.
Everything the solver prints is also saved to results/runs/<name>/log.txt.

Exit status: 0 when the JSON line was printed (check its "ok"), 1 when the command could
not run at all (an unreadable spec, Docker missing); a JSON line with ok=false is printed
then too.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kirigami import spec as S  # noqa: E402
from kirigami.argtypes import count0, positive  # noqa: E402

from kirigami.provenance import IMAGE  # noqa: E402  (one place for the solver image)
CONTAINER_PYTHONPATH = "/work:/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib"


Result = dict[str, Any]


def emit(obj: Result) -> None:
    print(json.dumps(obj))


def run_dir(spec: dict[str, Any]) -> Path:
    """results/runs/<name>/, created if needed; ValueError for a name that is not a plain
    folder name (spec.check_name)."""
    return S.run_folder(ROOT / "results" / "runs", spec)


def stage_spec(spec: dict[str, Any]) -> str:
    """Save the (default-filled) spec with the run: the record of what was
    simulated, and the path handed to the container. provenance.json beside it
    records the code version and the solver image (kirigami/provenance.py)."""
    p = run_dir(spec) / "spec.json"
    S.save(spec, p)
    try:                                   # which code and image produce this run
        from kirigami import provenance
        provenance.write(p.parent)
    except Exception:                      # never let bookkeeping stop a run
        pass
    return p.relative_to(ROOT).as_posix()


def rel(p: Path) -> str:
    """p relative to the repository when it is inside it, else as given."""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def log_path(args: Sequence[str]) -> Path | None:
    """results/runs/<name>/log.txt for a solver command whose second argument is the
    staged spec (stage_spec), else None."""
    if len(args) > 1 and args[1].endswith("spec.json"):
        return (ROOT / args[1]).parent / "log.txt"
    return None


def in_container(args: Sequence[str], name: str | None = None, echo: bool = False,
                 print_every: int = 1) -> Result:
    """Run python3 -m kirigami.api <args> in the FEniCSx container (or directly
    if we already are in one) and return its JSON result. `name` names the
    container, so that it can be stopped with `docker stop <name>`.

    Everything the solver prints is saved to log.txt in the run folder as it comes;
    with echo=True its progress (stderr) is also shown live on our stderr. Our stdout
    is left alone: the caller prints the one JSON line. print_every: a progress row every
    print_every-th load step (0: none), passed to the solver as KIRISMILES_PRINT_EVERY."""
    if print_every < 0:
        raise ValueError(f"print_every must be 0 (no rows) or a positive step interval, not {print_every}")
    try:
        import dolfinx  # noqa: F401
        cmd = [sys.executable, "-m", "kirigami.api", *args]
        env = dict(os.environ, KIRISMILES_PRINT_EVERY=str(print_every), PYTHONUNBUFFERED="1")
    except ImportError:
        cmd = ["docker", "run", "--rm", "--init", "--label", "kirigami=1", "--memory", "3g",
               *(["--name", name] if name else []),
               "-v", f"{ROOT}:/work", "-v", f"{ROOT / '.cache'}:/root/.cache/fenics",
               "-w", "/work", "-e", f"PYTHONPATH={CONTAINER_PYTHONPATH}",
               "-e", "OMP_NUM_THREADS=1", "-e", "PYTHONUNBUFFERED=1",
               "-e", f"KIRISMILES_PRINT_EVERY={print_every}",
               IMAGE, "python3", "-m", "kirigami.api", *args]
        env = None
    log = log_path(args)
    try:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
    except FileNotFoundError:
        return dict(ok=False, error="Docker is not installed (the solver runs in the "
                                    f"{IMAGE} container)")
    out_lines: list[str] = []
    err_lines: list[str] = []
    lock = threading.Lock()
    logf = open(log, "w", encoding="utf-8") if log else None

    def keep(line: str) -> None:
        with lock:
            if logf:
                logf.write(line)
                logf.flush()

    def pump_stderr() -> None:            # progress, warnings, tracebacks
        assert p.stderr is not None
        for line in p.stderr:
            err_lines.append(line)
            keep(line)
            if echo:
                sys.stderr.write(line)
                sys.stderr.flush()

    t = threading.Thread(target=pump_stderr, daemon=True)
    t.start()
    try:
        assert p.stdout is not None
        for line in p.stdout:             # the JSON summary, plus anything a library prints
            out_lines.append(line)
            keep(line)
        p.wait()
        t.join()
    finally:
        if logf:
            logf.close()
    stdout = "".join(out_lines)
    lines = [ln for ln in stdout.strip().splitlines() if ln.startswith("{")]
    tail = (stdout + "".join(err_lines)).strip().splitlines()[-15:]
    log_rel = {"log": rel(log)} if log else {}
    if p.returncode != 0 or not lines:
        return dict(ok=False, error=f"solver exited with code {p.returncode}", log_tail=tail, **log_rel)
    try:
        out = json.loads(lines[-1])
    except ValueError:
        return dict(ok=False, error="the solver's last line is not valid JSON", log_tail=tail, **log_rel)
    return dict(ok=True, **out, **log_rel)


def say(msg: str) -> None:
    """A human-readable line on stderr; stdout keeps its single JSON line."""
    print(msg, file=sys.stderr, flush=True)


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h} h {m} min {s} s" if h else f"{m} min {s} s" if m else f"{s} s"


def cmd_generators(_: argparse.Namespace) -> None:
    emit({name: str(inspect.signature(fn)) for name, fn in S.GENERATORS.items()})


def cmd_validate(a: argparse.Namespace) -> None:
    spec = S.load(a.spec)
    errs = S.validate(spec)
    pat = S.pattern(spec) if not errs else {}
    emit(dict(name=spec["name"], ok=not errs, errors=errs,
              counts={k: len(v) for k, v in pat.items()}))


def cmd_preview(a: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    spec = S.load(a.spec)
    L, W = spec["sheet"]["L"], spec["sheet"]["W"]
    pat = S.pattern(spec)
    fig, ax = plt.subplots(figsize=(8, 8 * W / L + 0.8))
    ax.add_patch(Rectangle((0, 0), L, W, fc="#d9d8d4", ec="#52514e"))
    for h in pat["holes"]:
        ax.add_patch(Polygon(h, fc="white", ec="#52514e", lw=0.5))
    for s in pat["solids"]:
        ax.add_patch(Polygon(s, fc="#d9d8d4", ec="#52514e", lw=0.5))
    for c in pat["cuts"]:
        xy = list(zip(*c))
        ax.plot(*xy, color="#e34948", lw=1.2)
    for x in (0, L):                                     # clamped / pulled edges
        ax.plot([x, x], [0, W], color="#2a78d6", lw=4)
    for name, (x, y) in spec["probes"].items():
        ax.plot(x, y, "k.", ms=4)
    ax.set(xlim=(-2, L + 2), ylim=(-2, W + 2), aspect="equal",
           title=f"{spec['name']}: red = cuts, blue = clamped (x=0) / pulled (x=L) edges")
    out = run_dir(spec) / "pattern.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    emit(dict(name=spec["name"], ok=True, files=[out.relative_to(ROOT).as_posix()]))


def cmd_export(a: argparse.Namespace) -> None:
    from kirigami import export as X
    spec = S.load(a.spec)
    errs = S.validate(spec)
    if errs:
        return emit(dict(name=spec["name"], ok=False, errors=errs))
    pat = S.pattern(spec)
    if pat["solids"]:
        return emit(in_container(["export", stage_spec(spec)]))
    L, W = spec["sheet"]["L"], spec["sheet"]["W"]
    paths = X.cut_paths(L, W, pat, exact=False)
    d = run_dir(spec)
    stage_spec(spec)
    X.write_svg(paths, L, W, d / "cut.svg")
    X.write_dxf(paths, d / "cut.dxf")
    emit(dict(name=spec["name"], ok=True, n_paths=len(paths),
              files=[(d / f).relative_to(ROOT).as_posix() for f in ("cut.svg", "cut.dxf")]))


def cmd_import(a: argparse.Namespace) -> None:
    """A laser-cut file (SVG or DXF) as a KiriSMILES string, checked; with --spec, also
    written as a spec ready to validate, preview, mesh or simulate."""
    from kirigami import importer as I
    from kirigami import kirismiles as ks
    try:
        r = I.import_file(a.file, L=a.L, W=a.W, tol=a.tol, digits=a.digits)
    except (ValueError, OSError, SyntaxError) as e:
        return emit(dict(ok=False, file=a.file, errors=[str(e)]))
    problems = ks.check(r["code"], r["L"], r["W"], min_gap=a.min_gap)
    out = dict(ok=True, file=a.file, code=r["code"], L=r["L"], W=r["W"], chars=len(r["code"]),
               problems=problems, **{k: v for k, v in r["report"].items() if k not in ("L", "W")})
    if a.spec:
        name = Path(a.spec).stem
        spec = S.load(dict(name=name, sheet=dict(L=r["L"], W=r["W"]),
                           pattern=dict(generator="kirismiles", params=dict(code=r["code"]))))
        S.save(spec, a.spec)
        out["spec"] = a.spec
    emit(out)


def cmd_mesh(a: argparse.Namespace) -> None:
    res = _solver("mesh", a)
    job_ended(a, res)
    emit(res)


def _solver(cmd: str, a: argparse.Namespace, *extra: str) -> Result:
    """Validate, stage and run one solver command. Unless --quiet, frame it with
    "Job started" / "Job ended" lines and show the solver's progress, all on stderr."""
    spec = S.load(a.spec)
    errs = S.validate(spec)
    if errs:
        return dict(name=spec["name"], ok=False, errors=errs)
    staged = stage_spec(spec)
    echo = not a.quiet
    t0 = time.time()
    if echo:
        say(f"Job started  {stamp()}   {cmd} {spec['name']}")
        log = log_path([cmd, staged])
        if log:
            say(f"log: {rel(log)}")
    res = in_container([cmd, staged, *extra], echo=echo, print_every=a.print_every)
    a._t0 = t0                                     # for job_ended(), after any post-processing
    return res


def job_ended(a: argparse.Namespace, res: Result) -> None:
    """The closing lines (files written, end time, duration) unless --quiet."""
    if a.quiet:
        return
    files = [str(f) for f in res.get("files", [])]
    if res.get("log"):
        files.append(res["log"])
    for i, f in enumerate(files):
        say(("files: " if i == 0 else "       ") + Path(f).as_posix())
    state = "completed" if res.get("ok") else "FAILED: " + str(res.get("error") or res.get("errors"))
    t0 = getattr(a, "_t0", None)
    took = f" in {duration(time.time() - t0)}" if t0 else ""
    say(f"Job ended    {stamp()}   {state}{took}")


def cmd_stl(a: argparse.Namespace) -> None:
    res = _solver("stl", a, str(a.thickness), str(a.kerf))
    job_ended(a, res)
    emit(res)


def cmd_buckle(a: argparse.Namespace) -> None:
    res = _solver("buckle", a)
    job_ended(a, res)
    emit(res)


def cmd_simulate(a: argparse.Namespace) -> None:
    res = _solver("simulate", a)
    if res.get("ok") and not a.no_gif:
        import make_gif
        name = res["name"]
        d = ROOT / "results" / "runs" / name
        if not a.quiet:
            say("rendering deformation.gif ...")
        make_gif.make(name, src=d / "history.npz", dst=d / "deformation.gif")
        res["files"] = res.get("files", []) + [(d / "deformation.gif").relative_to(ROOT).as_posix()]
    job_ended(a, res)
    emit(res)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; returns the exit status (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generators").set_defaults(fn=cmd_generators)
    def progress_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--quiet", action="store_true",
                       help="print only the JSON line (no start/end lines, no progress)")
        p.add_argument("--print-every", type=count0, default=1, metavar="N",
                       help="a progress row every N load steps (0: none; default 1)")

    for name, fn in (("validate", cmd_validate), ("preview", cmd_preview), ("export", cmd_export),
                     ("buckle", cmd_buckle), ("mesh", cmd_mesh)):
        p = sub.add_parser(name)
        p.add_argument("spec")
        if name in ("buckle", "mesh"):
            progress_options(p)
        p.set_defaults(fn=fn)
    p = sub.add_parser("import", help="convert a laser-cut file (.svg, .dxf) to a string")
    p.add_argument("file")
    p.add_argument("--L", type=positive, help="sheet length, mm (with --W; default: the outline in the file)")
    p.add_argument("--W", type=positive, help="sheet width, mm")
    p.add_argument("--tol", type=positive, default=0.01, help="curves to polylines within tol mm")
    p.add_argument("--digits", type=int, default=3, help="decimals of the coordinates")
    p.add_argument("--min-gap", type=positive, default=0.5, help="smallest hinge checked, mm")
    p.add_argument("--spec", help="also write a spec JSON here")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("stl")
    p.add_argument("spec")
    p.add_argument("--thickness", type=positive, default=0.5, help="mm")
    p.add_argument("--kerf", type=positive, default=0.4, help="slot width replacing each slit, mm")
    progress_options(p)
    p.set_defaults(fn=cmd_stl)
    p = sub.add_parser("simulate")
    p.add_argument("spec")
    p.add_argument("--no-gif", action="store_true")
    progress_options(p)
    p.set_defaults(fn=cmd_simulate)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (OSError, KeyError, TypeError, ValueError) as e:    # e.g. a missing or malformed spec
        emit(dict(ok=False, command=a.cmd, errors=[f"{type(e).__name__}: {e}"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
