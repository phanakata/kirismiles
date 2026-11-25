"""Local web app: browse sample designs, draw your own, run the FEM, see the GIF.

    python webapp/server.py            # then open http://localhost:8000

Uses only the Python standard library plus this project's host-side modules
(numpy, matplotlib, Pillow). This is the demo front end: simulations go
through the shared run engine kirigami/jobs.py, which also serves the MCP
server and batch scripts. It runs one solve at a time on the machine, whoever
started it, and keeps each job's record in results/runs/<name>/job.json.

API (JSON):
  GET  /api/samples            sample specs (specs/*.json) with resolved geometry
  GET  /api/generators         pattern generators and their parameters
  POST /api/pattern   {spec}   validate a spec and return its geometry
  POST /api/buckle    {spec}   quick linear analysis (seconds), synchronous
  POST /api/simulate  {spec}   queue a full post-buckling run -> {job}
  GET  /api/jobs/<id>          job status, progress, results
  POST /api/jobs/<id>/cancel   stop a queued or running job
  POST /api/export    {spec}   laser-cut SVG / DXF
  GET  /api/runs               previous runs (results/runs/*)
  GET  /files/<path>           files under results/runs/

Errors: 400 for a request the app cannot use (bad JSON, a body that is not an object, a
spec that cannot be read), 404 for an unknown path or job, 413 for a body over MAX_BODY,
500 for anything else; each with {"ok": false, "errors": [...]}.
"""
from __future__ import annotations

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
RUNS = ROOT / "results" / "runs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import kirigami_cli as cli  # noqa: E402
from kirigami import importer, jobs  # noqa: E402
from kirigami import spec as S  # noqa: E402

RUNNER: jobs.Runner | None = None      # started in main()
MAX_BODY = 5 * 1024 * 1024              # bytes: a spec with explicit geometry is far smaller


class BadRequest(ValueError):
    """A request the app cannot use; answered with 400 (or `status`) and the message."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def runner() -> jobs.Runner:
    if RUNNER is None:
        raise RuntimeError("the job runner is not started (run the app with main())")
    return RUNNER


# ------------------------------------------------------------------ helpers

def as_jsonable(pattern: dict[str, Any]) -> dict[str, list[list[list[float]]]]:
    """A pattern with plain floats, for JSON."""
    return {k: [[[float(x), float(y)] for x, y in g] for g in pattern.get(k, [])]
            for k in ("cuts", "holes", "solids")}


def kirismiles_string(spec: dict[str, Any], geometry: dict[str, Any] | None) -> str | None:
    """The design as a KiriSMILES string: the one typed for a kirismiles pattern, otherwise
    the free-form string of its cuts, holes and solids (None if there is nothing to show)."""
    pat = spec.get("pattern", {})
    if pat.get("generator") == "kirismiles":
        return str(pat.get("params", {}).get("code", ""))
    if not geometry:
        return None
    try:
        return importer.to_string(geometry, float(spec["sheet"]["L"]), float(spec["sheet"]["W"]))
    except (KeyError, TypeError, ValueError):     # an empty design, or a sheet size that is not a number
        return None


def web_job(job: jobs.Job) -> jobs.Job:
    """A job record with its result files as URLs the page can load."""
    res = job.get("result")
    if job.get("state") == "done" and res:
        n = job["name"]
        res = dict(res, gif=f"/files/{n}/deformation.gif?t={int(time.time())}")
        a = dict(res.get("analysis") or {})
        if a.get("plot"):
            a["files"] = [f"/files/{n}/stress_strain.png"]
        res["analysis"] = a
        job = dict(job, result=res)
    return job


# ------------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    raw: bytes = b""                    # the body of the current POST (read_body)

    def log_message(self, format: str, *args: Any) -> None:   # quieter console: no job polling
        if "/api/jobs/" not in str(args[0] if args else ""):
            sys.stderr.write("%s\n" % (format % args))

    def send_json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json(dict(ok=False, errors=[message], error=message), status)

    def send_file(self, path: Path, ctype: str | None = None) -> None:
        if not path.is_file():
            return self.send_json(dict(error="not found"), 404)
        types = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css",
                 ".gif": "image/gif", ".png": "image/png", ".svg": "image/svg+xml",
                 ".json": "application/json", ".dxf": "application/dxf",
                 ".stl": "model/stl"}
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype or types.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_body(self) -> bytes:
        """Read the whole request body. Every POST reads it before answering: on Windows a
        socket closed with unread bytes is reset, and the client can lose the answer."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise BadRequest("Content-Length is not a number") from None
        if n < 0:                       # rfile.read(-1) would wait for the client to close
            raise BadRequest("Content-Length must not be negative")
        if n > MAX_BODY:
            raise BadRequest(f"the request body is over {MAX_BODY // 2**20} MB", 413)
        return self.rfile.read(n)

    def body(self) -> dict[str, Any]:
        """The request's JSON object; BadRequest for a malformed body or one that is not an object."""
        try:
            obj = json.loads(self.raw or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise BadRequest(f"the request body is not valid JSON: {e}") from None
        if not isinstance(obj, dict):
            raise BadRequest(f"the request body must be a JSON object, not {type(obj).__name__}")
        return obj

    def spec_from_body(self) -> dict[str, Any]:
        try:
            spec = S.load(self.body())
        except (KeyError, TypeError, ValueError) as e:
            if isinstance(e, BadRequest):
                raise
            raise BadRequest(f"not a usable spec: {type(e).__name__}: {e}") from None
        spec["name"] = jobs.safe_name(spec.get("name"))
        return spec

    # -- GET
    def do_GET(self) -> None:
        try:
            self._get(self.path.split("?")[0])
        except BadRequest as e:
            self.send_error_json(e.status, str(e))
        except Exception as e:                  # the server keeps serving; the client gets a reason
            self.send_error_json(500, f"{type(e).__name__}: {e}")

    def _get(self, p: str) -> None:
        if p in ("/", "/index.html"):
            return self.send_file(STATIC / "index.html")
        if p.startswith("/static/"):
            f = (STATIC / p[len("/static/"):]).resolve()
            if STATIC.resolve() not in f.parents:          # no ../ out of the static folder
                return self.send_json(dict(error="forbidden"), 403)
            return self.send_file(f)
        if p.startswith("/files/"):
            f = (RUNS / p[len("/files/"):]).resolve()
            if RUNS.resolve() not in f.parents:
                return self.send_json(dict(error="forbidden"), 403)
            return self.send_file(f)
        if p == "/api/generators":
            import inspect
            return self.send_json({k: str(inspect.signature(v)) for k, v in S.GENERATORS.items()})
        if p == "/api/samples":
            out: list[dict[str, Any]] = []
            for f in sorted((ROOT / "specs").glob("*.json")):
                try:
                    spec = S.load(f)
                    out.append(dict(spec=spec, pattern=as_jsonable(S.pattern(spec))))
                except (OSError, KeyError, TypeError, ValueError) as e:     # one bad sample file
                    out.append(dict(name=f.stem, error=str(e)))
            return self.send_json(out)
        if p == "/api/runs":
            runs: list[dict[str, Any]] = []
            for d in sorted(RUNS.glob("*"), key=lambda d: d.stat().st_mtime, reverse=True):
                s = d / "summary.json"
                if s.exists() and (d / "deformation.gif").exists():
                    try:
                        summary = json.loads(s.read_text())
                    except (OSError, ValueError):       # being written, or damaged: skip it
                        continue
                    runs.append(dict(name=d.name, summary=summary, gif=f"/files/{d.name}/deformation.gif"))
                    if len(runs) == 30:
                        break
            return self.send_json(runs)
        m = re.match(r"^/api/jobs/(\w+)$", p)
        if m:
            job = runner().get(m.group(1))
            if not job:
                return self.send_error_json(404, "no such job")
            return self.send_json(web_job(job))
        return self.send_error_json(404, "not found")

    # -- POST
    def do_POST(self) -> None:
        try:
            self.raw = self.read_body()
            self._post(self.path.split("?")[0])
        except BadRequest as e:
            self.send_error_json(e.status, str(e))
        except Exception as e:                  # the server keeps serving; the client gets a reason
            self.send_error_json(500, f"{type(e).__name__}: {e}")

    def _post(self, p: str) -> None:
        m = re.match(r"^/api/jobs/(\w+)/cancel$", p)
        if m:
            if runner().cancel(m.group(1)) is None:
                return self.send_error_json(404, "no such job")
            return self.send_json(dict(ok=True))
        if p == "/api/pattern":
            spec = self.spec_from_body()
            errs = S.validate(spec)
            geometry = as_jsonable(S.pattern(spec)) if not any(
                e.startswith("pattern could not") for e in errs) else None
            return self.send_json(dict(ok=not errs, errors=errs, pattern=geometry, spec=spec,
                                       kirismiles=kirismiles_string(spec, geometry)))
        if p == "/api/buckle":
            spec = self.spec_from_body()
            errs = S.validate(spec)
            if errs:
                return self.send_json(dict(ok=False, errors=errs))
            return self.send_json(jobs.run_now("buckle", spec))
        if p == "/api/simulate":
            spec = self.spec_from_body()
            errs = S.validate(spec)
            if errs:
                return self.send_json(dict(ok=False, errors=errs))
            job = runner().submit(spec)
            return self.send_json(dict(ok=True, job=job["id"], name=job["name"]))
        if p == "/api/export":
            spec = self.spec_from_body()
            errs = S.validate(spec)
            if errs:
                return self.send_json(dict(ok=False, errors=errs))
            from kirigami import export as X
            pat = S.pattern(spec)
            if pat["solids"]:
                res = jobs.run_now("export", spec)
            else:
                L, W = spec["sheet"]["L"], spec["sheet"]["W"]
                d = cli.run_dir(spec)
                cli.stage_spec(spec)
                paths = X.cut_paths(L, W, pat, exact=False)
                X.write_svg(paths, L, W, d / "cut.svg")
                X.write_dxf(paths, d / "cut.dxf")
                res = dict(ok=True)
            n = spec["name"]
            if res.get("ok"):
                res["files"] = [f"/files/{n}/cut.svg", f"/files/{n}/cut.dxf"]
            return self.send_json(res)
        return self.send_error_json(404, "not found")


def _port(text: str) -> int:
    import argparse
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a port number") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"the port must be 1 to 65535, not {port}")
    return port


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=_port, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to accept other machines (no authentication: trusted networks only)")
    a = ap.parse_args()
    global RUNNER
    RUNNER = jobs.Runner("web")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Kirigami FEM app on http://localhost:{a.port}  (Ctrl+C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
