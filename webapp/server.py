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
"""
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
RUNS = ROOT / "results" / "runs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import kirigami_cli as cli  # noqa: E402
from kirigami import jobs  # noqa: E402
from kirigami import spec as S  # noqa: E402

RUNNER = None                    # jobs.Runner, started in main()


# ------------------------------------------------------------------ helpers

def as_jsonable(pattern):
    return {k: [[[float(x), float(y)] for x, y in g] for g in pattern.get(k, [])]
            for k in ("cuts", "holes", "solids")}


def web_job(job):
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
    def log_message(self, fmt, *args):                 # quieter console
        if "/api/jobs/" not in (args[0] if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype=None):
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

    def body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def spec_from_body(self):
        spec = S.load(self.body())
        spec["name"] = jobs.safe_name(spec.get("name"))
        return spec

    # -- GET
    def do_GET(self):
        p = self.path.split("?")[0]
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
            out = []
            for f in sorted((ROOT / "specs").glob("*.json")):
                try:
                    spec = S.load(f)
                    out.append(dict(spec=spec, pattern=as_jsonable(S.pattern(spec))))
                except Exception as e:
                    out.append(dict(name=f.stem, error=str(e)))
            return self.send_json(out)
        if p == "/api/runs":
            runs = []
            for d in sorted(RUNS.glob("*"), key=lambda d: d.stat().st_mtime, reverse=True):
                s = d / "summary.json"
                if s.exists() and (d / "deformation.gif").exists():
                    runs.append(dict(name=d.name, summary=json.loads(s.read_text()),
                                     gif=f"/files/{d.name}/deformation.gif"))
            return self.send_json(runs[:30])
        m = re.match(r"^/api/jobs/(\w+)$", p)
        if m:
            job = RUNNER.get(m.group(1))
            if not job:
                return self.send_json(dict(error="no such job"), 404)
            return self.send_json(web_job(job))
        return self.send_json(dict(error="not found"), 404)

    # -- POST
    def do_POST(self):
        p = self.path.split("?")[0]
        try:
            m = re.match(r"^/api/jobs/(\w+)/cancel$", p)
            if m:
                RUNNER.cancel(m.group(1))
                return self.send_json(dict(ok=True))
            if p == "/api/pattern":
                spec = self.spec_from_body()
                errs = S.validate(spec)
                pat = as_jsonable(S.pattern(spec)) if not any(
                    e.startswith("pattern could not") for e in errs) else None
                return self.send_json(dict(ok=not errs, errors=errs, pattern=pat, spec=spec))
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
                job = RUNNER.submit(spec)
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
                res["files"] = [f"/files/{n}/cut.svg", f"/files/{n}/cut.dxf"]
                return self.send_json(res)
            return self.send_json(dict(error="not found"), 404)
        except Exception as e:
            return self.send_json(dict(ok=False, errors=[f"{type(e).__name__}: {e}"]), 500)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
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
