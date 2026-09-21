import types

import pytest
from langchain_core.messages import ToolMessage

from laya_assistant.policy import PathPolicyMiddleware


def call(tool, path, call_id="c1"):
    return types.SimpleNamespace(tool_call={"id": call_id, "name": tool, "args": {"file_path": path}}, state={})


def run(mw, tool, path):
    ran = []

    def h(req):
        ran.append(req)
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    return mw.wrap_tool_call(call(tool, path), h), ran


@pytest.mark.parametrize("tool", ["write_file", "edit_file", "delete"])
@pytest.mark.parametrize("path", ["/workspace/a.py", "/workspace/sub/dir/b.txt", "/memories/AGENTS.md",
                                  "/memories/sessions/x.md", "/large_tool_results/c1.txt", "/conversation_history/h.md"])
def test_writes_inside_allowed_areas_run(tool, path):
    r, ran = run(PathPolicyMiddleware(), tool, path)
    assert ran and r.content == "ok"


@pytest.mark.parametrize("path", ["/etc/passwd", "/skills/coding-workflow/SKILL.md", "/skills/new/SKILL.md", "/usr/local/bin/x",
                                  "/workspace/../etc/passwd", "workspace/a.py", "/workspaceevil/a.py", "/"])
def test_writes_outside_allowed_areas_are_denied(path):
    r, ran = run(PathPolicyMiddleware(), "write_file", path)
    assert r.status == "error" and "not allowed" in r.content and not ran


def test_reads_are_never_restricted():
    r, ran = run(PathPolicyMiddleware(), "read_file", "/etc/hostname")
    assert ran


def test_read_only_mode_blocks_every_write_but_lets_execute_and_reads_through():
    mw = PathPolicyMiddleware(read_only=True)
    r, ran = run(mw, "write_file", "/workspace/a.py")
    assert r.status == "error" and "read-only" in r.content and not ran
    _, ran = run(mw, "read_file", "/workspace/a.py")
    assert ran
    _, ran = run(mw, "execute", "")
    assert ran
