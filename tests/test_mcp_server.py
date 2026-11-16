"""The MCP server, end to end: started over stdio and driven by the MCP client,
as Claude Code drives it. The solver tools are only checked to answer cleanly;
they need Docker.

    python -m pytest -q tests/test_mcp_server.py
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
import anyio                                                   # noqa: E402
from mcp import ClientSession, StdioServerParameters           # noqa: E402
from mcp.client.stdio import stdio_client                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOOLS = {"check_design", "preview_design", "list_samples", "export_cut_files", "buckle",
         "start_simulation", "job_status", "wait_for_job", "cancel_job", "list_jobs",
         "list_runs", "show_run", "find_runs", "import_cut_file", "export_mesh"}


def call(fn):
    async def main():
        params = StdioServerParameters(command=sys.executable,
                                       args=[str(ROOT / "mcp_server" / "server.py")], cwd=str(ROOT))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)
    return anyio.run(main)


def text(result):
    return "".join(c.text for c in result.content if c.type == "text")


def test_the_server_lists_its_tools_and_resources():
    async def go(s):
        tools = await s.list_tools()
        resources = await s.list_resources()
        doc = await s.read_resource("kirismiles://notation")
        return tools, resources, doc
    tools, resources, doc = call(go)
    assert {t.name for t in tools.tools} == TOOLS
    assert {str(r.uri) for r in resources.resources} == {"kirismiles://notation",
                                                         "kirismiles://spec-format"}
    assert "KiriSMILES" in doc.contents[0].text


def test_design_tools():
    async def go(s):
        return (await s.call_tool("check_design", {"code": "P5x40S1/2O0[V36@0,0]N20E4"}),
                await s.call_tool("check_design", {"code": "V30@0,0;H30@0,0"}),
                await s.call_tool("check_design", {"code": "R3024@0,0"}),
                await s.call_tool("preview_design", {"code": "(V40@6,0){mx}"}),
                await s.call_tool("list_samples", {}))
    ok, crossing, bad, preview, samples = call(go)
    r = json.loads(text(ok))
    assert r["valid"] and r["geometry_key"] == "7acc55b51e60" and r["counts"]["cuts"] == 50
    r = json.loads(text(crossing))
    assert not r["valid"] and "cross" in r["problems"][0]
    r = json.loads(text(bad))
    assert not r["valid"] and "R<deg>:<length>" in r["problems"][0]
    kinds = [c.type for c in preview.content]
    assert kinds == ["text", "image"] and preview.content[1].mime_type == "image/png"
    assert "alternating" in text(samples)


def test_solver_tools_answer_cleanly():
    """With Docker off they explain why; with it on, a job id comes back."""
    async def go(s):
        start = await s.call_tool("start_simulation", {"code": "V40@0,0", "max_stretch": 0.01,
                                                       "steps": 2, "name": "mcp_test"})
        missing = await s.call_tool("job_status", {"job_id": "00000000"})
        return start, missing
    start, missing = call(go)
    r = json.loads(text(start))
    if r["ok"] and r.get("reused"):                 # an identical run is already stored
        assert r["state"] == "done" and r["completed"] and r["same_settings"]
    elif r["ok"]:
        assert r["state"] in ("queued", "running") and len(r["id"]) == 8
    else:
        assert "Docker" in r["errors"][0]
    assert json.loads(text(missing)) == {"error": "no such job"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
