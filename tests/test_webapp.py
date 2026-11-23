"""The web app's HTTP layer (webapp/server.py), served in this process on a free port.
No solver is started: only the endpoints that check designs and report errors are called.

    python -m pytest -q tests/test_webapp.py
"""
import http.client
import importlib.util
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GOOD = {"name": "web test", "sheet": {"L": 120, "W": 60},
        "pattern": {"generator": "kirismiles", "params": {"code": "P5x45S1/2O0[V42@0,0]N18E3"}}}


class NoJobs:
    """Stands in for the job runner: no job exists."""

    def get(self, job_id):
        return None

    def cancel(self, job_id):
        return None


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("webapp_server", ROOT / "webapp" / "server.py")
    web = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(web)
    web.RUNNER = NoJobs()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def request(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    c.request(method, path, body=data, headers=headers or {})
    r = c.getresponse()
    out = r.status, r.read()
    c.close()
    return out


def test_pattern_endpoint_checks_a_good_spec(app):
    status, body = request(app, "POST", "/api/pattern", GOOD)
    r = json.loads(body)
    assert status == 200 and r["ok"] and r["errors"] == [] and len(r["pattern"]["cuts"]) == 45
    assert r["spec"]["name"] == "web_test"                     # made safe for a folder name


def test_pattern_endpoint_reports_a_bad_design(app):
    bad = {**GOOD, "material": {"h": -1}}
    status, body = request(app, "POST", "/api/pattern", bad)
    r = json.loads(body)
    assert status == 200 and not r["ok"] and any("h > 0" in e for e in r["errors"])


@pytest.mark.parametrize("body, headers, code, expect", [
    (b"{not json", None, 400, "not valid JSON"),
    (b"[1, 2]", None, 400, "JSON object"),
    (json.dumps({"pattern": {}}).encode(), None, 400, "not a usable spec"),      # no sheet
    # no body with a bad length: the server answers without reading one, and on Windows a socket
    # closed with unread bytes is reset, which can drop the answer before the client reads it
    (b"", {"Content-Length": "-5"}, 400, "negative"),
    (b"", {"Content-Length": "abc"}, 400, "not a number"),
    (b"", {"Content-Length": str(64 * 2**20)}, 413, "over"),
])
def test_bad_requests_get_a_4xx_and_a_reason(app, body, headers, code, expect):
    status, raw = request(app, "POST", "/api/pattern", body, headers)
    r = json.loads(raw)
    assert status == code and not r["ok"] and expect in r["errors"][0]


def test_unknown_paths_jobs_and_files(app):
    assert request(app, "GET", "/api/nothing")[0] == 404
    assert request(app, "GET", "/api/jobs/00000000")[0] == 404
    assert request(app, "POST", "/api/jobs/00000000/cancel", {})[0] == 404
    assert request(app, "GET", "/files/../../README.md")[0] in (403, 404)       # no way out of results/runs
    assert request(app, "GET", "/static/../server.py")[0] in (403, 404)


def test_generators_and_samples_list(app):
    status, body = request(app, "GET", "/api/generators")
    assert status == 200 and "kirismiles" in json.loads(body)
    status, body = request(app, "GET", "/api/samples")
    assert status == 200 and any("spec" in s for s in json.loads(body))
