"""KiriSMILES MCP server: kirigami design, checking and simulation as tools for an AI assistant.

    claude mcp add kirismiles -s project -- python mcp_server/server.py
    (or use the .mcp.json at the repository root, which does the same)

Speaks MCP over stdio by default; needs the `mcp` package (pip install "mcp>=2.3,<3").
To keep one server running for several clients (people or agents), serve it over HTTP:

    python mcp_server/server.py --http [--host 127.0.0.1] [--port 8765]
    claude mcp add --transport http kirismiles http://127.0.0.1:8765/mcp

The HTTP server has no authentication: keep it on 127.0.0.1 or a trusted network.
Simulations go through the shared run engine kirigami/jobs.py, the same one the
web app uses: one solve at a time on the machine, a record for every job in
results/runs/<name>/job.json, and runs that keep going if the session ends.

Designs are written as KiriSMILES strings (KIRISMILES.md) on an L x W sheet in
mm; every design tool also accepts a full JSON spec (kirigami/spec.py) instead.

Tools report problems with the input as {"ok": false, "errors": [...]} (or "problems" for
check_design) rather than raising, so an assistant can read and fix them.
"""
from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import Image, MCPServer  # noqa: E402
from mcp_types import ToolAnnotations  # noqa: E402

from kirigami import jobs  # noqa: E402
from kirigami import kirismiles as ks  # noqa: E402
from kirigami import spec as S  # noqa: E402
from kirigami.stiffness import from_history as stiffness_fit  # noqa: E402

RUNS = jobs.RUNS
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

INSTRUCTIONS = """\
Kirigami sheets: design cut patterns, check them, and simulate how they deform when stretched.

- Write designs as KiriSMILES strings (read the kirismiles://notation resource first), e.g.
  P5x45S1/2O0[V42@0,0]N18E3 for the classic alternating pattern. Lengths are in mm; the sheet
  is L x W (default 120 x 60), clamped at x = 0 and pulled at x = L.
- Always run check_design before simulating: it is instant, and the solver needs a valid design.
- preview_design shows the pattern as an image.
- import_cut_file turns a laser-cut file (SVG or DXF) into a string; export_mesh writes the
  finite element mesh of a design for other codes (gmsh, Abaqus, Nastran, VTK, XDMF).
- buckle takes seconds; start_simulation takes minutes and returns a job id. Follow it with
  wait_for_job or job_status, then show_run to see the result.
- Results are stored by design. find_runs lists the stored simulations of a design, and
  start_simulation returns a stored result instead of solving again when one matches exactly.
- The machine runs one solve at a time; a new job waits for the current one. Simulation tools
  need Docker running.
"""

server = MCPServer("kirismiles", instructions=INSTRUCTIONS)
_runner: jobs.Runner | None = None
_runner_lock = threading.Lock()
MAX_WAIT_S = 300.0                          # wait_for_job never blocks a session longer


def runner() -> jobs.Runner:
    """The job queue, started on first use (tools run in parallel threads)."""
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = jobs.Runner("mcp")
    return _runner


# ------------------------------------------------------------------ helpers

def make_spec(code: str | None = None, L: float = 120.0, W: float = 60.0, spec: dict[str, Any] | None = None,
              name: str | None = None, settings: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """A complete spec from a KiriSMILES string or a spec dict, with defaults filled in.
    Raises ValueError for neither or a section that is not an object."""
    if spec is None:
        if not code:
            raise ValueError("give a KiriSMILES string (code) or a full spec")
        spec = {"sheet": {"L": L, "W": W},
                "pattern": {"generator": "kirismiles", "params": {"code": code}}}
    spec = json.loads(json.dumps(spec))                       # a deep copy
    if name:
        spec["name"] = name
    spec.setdefault("name", "design")
    for key, val in (settings or {}).items():
        section = spec.setdefault(key, {})
        if not isinstance(section, dict):
            raise ValueError(f"spec[{key!r}] must be an object")
        section.update(val)
    spec = S.load(spec)
    spec["name"] = jobs.safe_name(spec["name"])
    return spec


def _sheet_errors(L: float, W: float) -> list[str]:
    return [] if L > 0 and W > 0 else [f"the sheet needs L > 0 and W > 0 (got L = {L}, W = {W})"]


def counts(pat: dict[str, Any]) -> dict[str, int]:
    return {k: len(pat.get(k, ())) for k in ("cuts", "holes", "solids")}


def render(pat: dict[str, Any], L: float, W: float, title: str = "") -> bytes:
    """PNG of a pattern: sheet grey, cuts red, clamped and pulled ends blue."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    fig, ax = plt.subplots(figsize=(8, 8 * W / L + 0.7))
    ax.add_patch(Rectangle((0, 0), L, W, fc="#d9d8d4", ec="#52514e"))
    for h in pat.get("holes", ()):
        ax.add_patch(Polygon(h, fc="white", ec="#52514e", lw=0.6))
    for s in pat.get("solids", ()):
        ax.add_patch(Polygon(s, fc="#c4c3bd", ec="#52514e", lw=0.6))
    for c in pat.get("cuts", ()):
        ax.plot(*zip(*c), color="#e34948", lw=1.2)
    for x in (0, L):
        ax.plot([x, x], [0, W], color="#2a78d6", lw=4)
    ax.set(xlim=(-2, L + 2), ylim=(-2, W + 2), aspect="equal", title=title)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def summarize(job: jobs.Job | None) -> dict[str, Any]:
    """The parts of a job record an assistant needs."""
    if job is None:
        return {"error": "no such job"}
    out: dict[str, Any] = {k: job.get(k) for k in ("id", "name", "state", "message", "steps")}
    out["progress_percent"] = round(100 * (job.get("progress") or 0))
    out["delta_mm"] = job.get("delta")
    res = job.get("result") or {}
    if job.get("state") == "done":
        a = res.get("analysis") or {}
        force = res.get("force_N") or [None]
        out["result"] = {
            "status": res.get("status"),
            "buckles_at_Delta_c_mm": res.get("Delta_c_mm"),
            "stiffness_N_per_mm": res.get("stiffness_N_per_mm"),
            "final_stretch_Delta_over_L": res.get("strain_final"),
            "max_out_of_plane_mm": res.get("uz_max_final_mm"),
            "final_force_N": force[-1],
            "stiffness_before_buckling_N_per_mm": a.get("stiffness_before_buckling_N_per_mm"),
            "stiffness_after_buckling_N_per_mm": a.get("stiffness_after_buckling_N_per_mm"),
            "stiffness_fit_range_pct": a.get("stiffness_fit_range_pct"),
            "stretchability": a.get("stretchability"),
            "stretchability_note": a.get("stretchability_note"),
            "files": res.get("files"),
        }
    elif job.get("state") == "failed" and res:
        out["details"] = res.get("log_tail") or res.get("trace")
    return out


def run_dir(name: str) -> Path:
    """The folder of a stored run; ValueError for a name outside results/runs or not there."""
    d = (RUNS / name).resolve()
    if RUNS.resolve() not in d.parents or not d.is_dir():
        raise ValueError(f"no run named {name!r}")
    return d


def _read_summary(d: Path) -> dict[str, Any] | None:
    """A run's summary.json, or None while it is being written or if it is damaged."""
    try:
        s = json.loads((d / "summary.json").read_text())
    except (OSError, ValueError):
        return None
    return s if isinstance(s, dict) else None


def last_frame(gif_path: Path) -> bytes:
    """The last frame of an animation, as PNG."""
    from PIL import Image as PILImage
    g = PILImage.open(gif_path)
    g.seek(getattr(g, "n_frames", 1) - 1)          # a still image has one frame
    buf = io.BytesIO()
    g.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# -------------------------------------------------------------- design tools

@server.tool(annotations=READ_ONLY)
def check_design(code: str, L: float = 120.0, W: float = 60.0, min_gap: float = 1.0) -> dict[str, Any]:
    """Check a KiriSMILES design on an L x W mm sheet: instant.

    Returns valid (bool), the readable problems that make it impossible to cut or clamp
    (crossing cuts, ligaments narrower than min_gap mm, pieces cut loose, cuts reaching a
    clamped end), its canonical string, its geometry key (the design's identity), and the
    number of cuts, holes and solids."""
    bad = _sheet_errors(L, W) + ([] if min_gap > 0 else [f"min_gap must be positive, not {min_gap}"])
    if bad:
        return {"valid": False, "problems": bad}
    try:
        d = ks.parse(code)
    except ValueError as e:
        return {"valid": False, "problems": [f"does not parse: {e}"]}
    problems = ks.check(d, L, W, min_gap=min_gap)
    pat = ks.pattern(d, L, W)
    return {"valid": not problems, "problems": problems, "canonical": ks.encode(d),
            "geometry_key": ks.geometry_key(pat, L, W), "counts": counts(pat)}


@server.tool(annotations=READ_ONLY, structured_output=False)
def preview_design(code: str | None = None, L: float = 120.0, W: float = 60.0,
                   spec: dict[str, Any] | None = None) -> list[str | Image]:
    """Draw a design as an image: grey sheet, red cuts, white holes, blue clamped (left) and
    pulled (right) ends. Give a KiriSMILES string with L and W, or a full JSON spec."""
    sp = make_spec(code, L, W, spec)
    pat = S.pattern(sp)
    Ls, Ws = sp["sheet"]["L"], sp["sheet"]["W"]
    label = code or sp["pattern"].get("generator", "custom")
    return [f"{label} on {Ls:g} x {Ws:g} mm: {counts(pat)}",
            Image(data=render(pat, Ls, Ws, label), format="png")]


@server.tool(annotations=READ_ONLY)
def list_samples() -> list[dict[str, Any]]:
    """The example designs in specs/: name, sheet size, and how the pattern is made."""
    out: list[dict[str, Any]] = []
    for f in sorted((ROOT / "specs").glob("*.json")):
        try:
            sp = S.load(f)
            p = sp["pattern"]
        except (OSError, KeyError, TypeError, ValueError) as e:     # one bad file: report it, go on
            out.append({"name": f.stem, "file": f"specs/{f.name}", "error": str(e)})
            continue
        out.append({"name": sp["name"], "file": f"specs/{f.name}", "sheet_mm": sp["sheet"],
                    "generator": p.get("generator", "explicit geometry"),
                    "params": p.get("params")})
    return out


@server.tool(annotations=WRITES)
def export_cut_files(code: str | None = None, L: float = 120.0, W: float = 60.0,
                     spec: dict[str, Any] | None = None, name: str | None = None) -> dict[str, Any]:
    """Write laser-cut files (SVG and DXF) for a design to results/runs/<name>/.
    Patterns with solids need Docker, the others do not."""
    from kirigami import export as X
    sp = make_spec(code, L, W, spec, name)
    errs = S.validate(sp)
    if errs:
        return {"ok": False, "errors": errs}
    pat = S.pattern(sp)
    if pat["solids"]:
        res = jobs.run_now("export", sp)
    else:
        cli, _, _ = jobs._scripts()
        d = cli.run_dir(sp)
        cli.stage_spec(sp)
        Ls, Ws = sp["sheet"]["L"], sp["sheet"]["W"]
        paths = X.cut_paths(Ls, Ws, pat, exact=False)
        X.write_svg(paths, Ls, Ws, d / "cut.svg")
        X.write_dxf(paths, d / "cut.dxf")
        res = {"ok": True, "n_paths": len(paths)}
    if res.get("ok"):
        res["files"] = [f"results/runs/{sp['name']}/cut.svg", f"results/runs/{sp['name']}/cut.dxf"]
    return res


@server.tool(annotations=READ_ONLY)
def import_cut_file(path: str | None = None, content: str | None = None, fmt: str = "svg",
                    L: float | None = None, W: float | None = None, tol: float = 0.01,
                    digits: int = 3, min_gap: float = 0.5) -> dict[str, Any]:
    """Convert a laser-cut file (SVG or DXF) into a KiriSMILES string. Give either `path` (a
    .svg or .dxf file on this machine) or `content` (the file's text, with fmt "svg" or
    "dxf"). The sheet is the closed rectangle around everything, unless L and W are given.
    Open curves become cuts, closed curves holes (or solids when nested inside a hole);
    curves are replaced by polylines within `tol` mm. Returns the string, the sheet size,
    the counts, max_deviation_mm (string geometry against the file) and check problems."""
    import tempfile
    from kirigami import importer as I
    if (path is None) == (content is None):
        return {"ok": False, "errors": ["give either path or content"]}
    if path is not None:
        if not path.lower().endswith((".svg", ".dxf")):
            return {"ok": False, "errors": ["path must be a .svg or .dxf file"]}
        src, tmp = path, None
    else:
        if fmt not in ("svg", "dxf"):
            return {"ok": False, "errors": ['fmt must be "svg" or "dxf"']}
        tmp = tempfile.NamedTemporaryFile("w", suffix="." + fmt, delete=False, encoding="utf-8")
        tmp.write(content or "")
        tmp.close()
        src = tmp.name
    try:
        r = I.import_file(src, L=L, W=W, tol=tol, digits=digits)
    except (ValueError, OSError, SyntaxError) as e:
        return {"ok": False, "errors": [str(e)]}
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)
    rep = {k: v for k, v in r["report"].items() if k not in ("L", "W")}
    return {"ok": True, "code": r["code"], "L": r["L"], "W": r["W"], "chars": len(r["code"]),
            "problems": ks.check(r["code"], r["L"], r["W"], min_gap=min_gap), **rep}


@server.tool(annotations=WRITES)
def export_mesh(code: str | None = None, L: float = 120.0, W: float = 60.0,
                spec: dict[str, Any] | None = None, name: str | None = None) -> dict[str, Any]:
    """Write the finite element mesh of a design, exactly as the solver uses it, to
    results/runs/<name>/: mesh.msh (gmsh), mesh.inp (Abaqus), mesh.bdf (Nastran), mesh.vtk
    and mesh.xdmf. Linear triangles in mm, with the nodes along every cut duplicated so that
    its lips are free. Seconds; needs Docker."""
    sp = make_spec(code, L, W, spec, name)
    errs = S.validate(sp)
    if errs:
        return {"ok": False, "errors": errs}
    return jobs.run_now("mesh", sp)


# ------------------------------------------------------------ solver tools

@server.tool(annotations=WRITES)
def buckle(code: str | None = None, L: float = 120.0, W: float = 60.0,
           spec: dict[str, Any] | None = None, thickness_mm: float | None = None,
           name: str | None = None) -> dict[str, Any]:
    """Quick linear analysis (seconds, needs Docker): the stretched sheet's stiffness before
    buckling, the stretch Delta_c at which it first buckles out of plane, and the force F_c.
    Fails with a reason if the solver is busy with another run."""
    settings = {"material": {"h": thickness_mm}} if thickness_mm is not None else None   # 0 is checked
    sp = make_spec(code, L, W, spec, name, settings)
    errs = S.validate(sp)
    if errs:
        return {"ok": False, "errors": errs}
    return jobs.run_now("buckle", sp)


@server.tool(annotations=WRITES)
def start_simulation(code: str | None = None, L: float = 120.0, W: float = 60.0,
                     spec: dict[str, Any] | None = None, max_stretch: float = 0.1,
                     steps: int = 40, thickness_mm: float | None = None,
                     E_MPa: float | None = None, nu: float | None = None,
                     boundary: str | None = None, in_plane: bool | None = None,
                     name: str | None = None, reuse: bool = True) -> dict[str, Any]:
    """Queue a full nonlinear shell simulation (minutes, needs Docker): the sheet is stretched
    to max_stretch = Delta/L in `steps` load steps and its post-buckled shape is followed.
    Material: thickness_mm, E_MPa, nu (default PET: 0.127 mm, 3500 MPa, 0.38).
    boundary: "clamped" (grips, default), "roller" (ends move only along x: uniaxial stress)
    or "biaxial" (all edges pulled out equally). in_plane=True keeps the sheet flat (membrane
    model, no buckling), like plane 2D calculations of rotating-unit kirigami.
    Returns a job id at once; use wait_for_job or job_status to follow it, and show_run to
    see the result. Jobs run one at a time and keep running if this session ends.
    If a completed run of the same design (same geometry key) with the same sheet, material,
    mesh, boundary and load is already stored, it is returned instead (reused: true) and
    nothing is solved; pass reuse=False to solve again."""
    if not 0 < max_stretch <= 0.6:
        return {"ok": False, "errors": ["max_stretch is Delta/L and must be in (0, 0.6]"]}
    if steps < 1:
        return {"ok": False, "errors": [f"steps must be at least 1, not {steps}"]}
    settings: dict[str, dict[str, Any]] = {"load": {"Dmax_over_L": max_stretch, "steps": int(steps)}}
    bnd = {k: v for k, v in (("type", boundary), ("in_plane", in_plane)) if v is not None}
    if bnd:
        settings["boundary"] = bnd
    mat = {k: v for k, v in (("h", thickness_mm), ("E", E_MPa), ("nu", nu)) if v is not None}
    if mat:
        settings["material"] = mat
    sp = make_spec(code, L, W, spec, name, settings)
    sp["load"]["Dmax"] = max_stretch * sp["sheet"]["L"]
    errs = S.validate(sp)
    if errs:
        return {"ok": False, "errors": errs}
    if reuse:
        for r in jobs.find_runs(sp):
            if r["completed"] and r["same_settings"]:
                return {"ok": True, "reused": True, "state": "done", **r,
                        "note": "stored result of the same design and settings; "
                                "show_run(name) shows it"}
    why = jobs.docker_ok()
    if why:
        return {"ok": False, "errors": [why]}
    job = runner().submit(sp)
    return {"ok": True, "reused": False, **summarize(job)}


@server.tool(annotations=READ_ONLY)
def job_status(job_id: str) -> dict[str, Any]:
    """A simulation job's state (queued, running, done, failed, cancelled, interrupted),
    progress, and when done its key results and files."""
    return summarize(runner().get(job_id))


@server.tool(annotations=READ_ONLY)
def wait_for_job(job_id: str, timeout_s: float = 120.0) -> dict[str, Any]:
    """Wait up to timeout_s seconds (0 to 300) for a job to finish, then report its
    status. Saves repeated job_status calls while a simulation runs."""
    import time
    limit = min(max(float(timeout_s), 0.0), MAX_WAIT_S)
    t0 = time.time()
    while True:
        job = runner().get(job_id)
        if job is None or job["state"] not in jobs.ACTIVE or time.time() - t0 >= limit:
            return summarize(job)
        time.sleep(min(5.0, max(limit - (time.time() - t0), 0.1)))


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True,
                                      open_world_hint=False))
def cancel_job(job_id: str) -> dict[str, Any]:
    """Stop a queued or running simulation. Steps already solved stay in history.npz."""
    return summarize(runner().cancel(job_id))


@server.tool(annotations=READ_ONLY)
def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    """Recent simulation jobs from every front end (this server, the web app, scripts);
    limit from 1 to 200."""
    return [summarize(j) for j in runner().list(min(max(int(limit), 1), 200))]


# ------------------------------------------------------------------ results

@server.tool(annotations=READ_ONLY)
def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Finished simulations in results/runs/, newest first: name, status and key numbers;
    limit from 1 to 200."""
    limit = min(max(int(limit), 1), 200)
    out: list[dict[str, Any]] = []
    dirs = sorted((d for d in RUNS.glob("*") if (d / "summary.json").exists()),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs:
        if len(out) == limit:
            break
        s = _read_summary(d)
        if s is None:                               # being written, or damaged: skip it
            continue
        out.append({"name": d.name, "status": s.get("status"),
                    "final_stretch": s.get("strain_final"),
                    "max_out_of_plane_mm": s.get("uz_max_final_mm"),
                    "has_animation": (d / "deformation.gif").exists()})
    return out


@server.tool(annotations=READ_ONLY)
def find_runs(code: str | None = None, L: float = 120.0, W: float = 60.0,
              spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stored simulations of this design, found by its geometry key, so the same design
    written in another way (or mirrored) is found too. Each entry gives the run name, whether
    it completed, its thickness, boundary and final stretch. Use it before start_simulation."""
    sp = make_spec(code, L, W, spec)
    errs = S.validate(sp)
    if errs:
        return {"ok": False, "errors": errs}
    runs = jobs.find_runs(sp)
    return {"ok": True, "geometry_key": jobs.design_key(sp), "runs": runs}


@server.tool(annotations=READ_ONLY, structured_output=False)
def show_run(name: str) -> list[str | Image]:
    """A finished run: its summary, the final deformed shape, and the stress-strain plot."""
    d = run_dir(name)
    out: list[str | Image] = []
    if (d / "summary.json").exists():
        s = _read_summary(d)
        if s is None:
            out.append("summary.json could not be read (still being written, or damaged)")
        else:
            s.pop("force_N", None)                  # long series; the plot shows it
            s.pop("Delta_mm", None)
            out.append(json.dumps(s, indent=1))
    if (d / "history.npz").exists():
        try:
            fit = stiffness_fit(d / "history.npz")
            out.append("stiffness, fitted to the force history (N/mm): " + json.dumps(fit))
        except (KeyError, OSError, ValueError):     # no buckling stretch stored, or under 1 % strain
            pass
    if (d / "spec.json").exists():
        out.append("spec: " + (d / "spec.json").read_text())
    if (d / "deformation.gif").exists():
        out.append(Image(data=last_frame(d / "deformation.gif"), format="png"))
    if (d / "stress_strain.png").exists():
        out.append(Image(path=d / "stress_strain.png"))
    return out or [f"run {name} has no results yet"]


# ---------------------------------------------------------------- resources

@server.resource("kirismiles://notation", mime_type="text/markdown",
              description="The KiriSMILES notation: pieces, operations, lattice form, grammar")
def notation() -> str:
    return (ROOT / "KIRISMILES.md").read_text(encoding="utf-8")


@server.resource("kirismiles://spec-format", mime_type="text/plain",
              description="The JSON spec format: sheet, material, pattern, mesh, load, defaults")
def spec_format() -> str:
    return (S.__doc__ or "") + "\nDefaults: " + json.dumps(S.DEFAULTS)


def _port(text: str) -> int:
    import argparse
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a port number") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"the port must be 1 to 65535, not {port}")
    return port


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--http", action="store_true",
                    help="serve over HTTP for several clients, instead of stdio for one")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on (no authentication: trusted networks only)")
    ap.add_argument("--port", type=_port, default=8765)
    a = ap.parse_args()
    if a.http:
        server.run(transport="streamable-http", host=a.host, port=a.port)
    else:
        server.run()
