"""Shared-folder file operations (pure, real filesystem) and the sandbox mount (real Docker)."""
import subprocess
import time
import types

import pytest
from langchain_core.messages import ToolMessage

from laya_assistant import config, policy
from laya_assistant.files import SharedFolder
from laya_assistant.sandbox import DockerSandbox


@pytest.fixture
def shared(tmp_path):
    return SharedFolder(tmp_path / "LayaWorkspace")


def test_the_folder_is_created_and_listed_newest_first_without_hidden_files(shared):
    shared.save_upload("a.txt", b"a")
    time.sleep(0.02)
    shared.save_upload("b.txt", b"bb")
    (shared.root / ".hidden").write_text("x")
    names = [f.rel for f in shared.list()]
    assert names == ["b.txt", "a.txt"] and shared.list()[0].size == 2


def test_uploads_are_sanitised_and_never_overwrite(shared):
    p1 = shared.save_upload("../../etc/passwd", b"1")
    assert p1.parent == shared.root and p1.name == "passwd"
    p2 = shared.save_upload("passwd", b"2")
    assert p2.name != "passwd" and p1.read_bytes() == b"1"  # a second upload gets a new name
    assert shared.save_upload("", b"x").name.startswith("upload")


@pytest.mark.parametrize("rel", ["../outside.txt", "/etc/hosts", "a/../../b"])
def test_paths_outside_the_folder_are_refused(shared, rel):
    with pytest.raises(ValueError):
        shared.resolve(rel)


def test_previews_by_kind(shared):
    shared.save_upload("n.md", b"# hi\nthere")
    shared.save_upload("t.csv", b"a,b\n1,2\n3,4\n")
    shared.save_upload("p.png", b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    shared.save_upload("z.bin", bytes(range(256)))
    assert shared.preview("n.md").kind == "text" and "# hi" in shared.preview("n.md").text
    csv = shared.preview("t.csv")
    assert csv.kind == "csv" and csv.rows == [["a", "b"], ["1", "2"], ["3", "4"]]
    assert shared.preview("p.png").kind == "image"
    assert shared.preview("z.bin").kind == "binary"


def test_large_text_previews_are_truncated(shared):
    shared.save_upload("big.txt", b"x" * 500_000)
    pv = shared.preview("big.txt")
    assert pv.kind == "text" and len(pv.text) <= 20_500 and pv.truncated


def test_rename_and_refuse_to_overwrite(shared):
    shared.save_upload("a.txt", b"a")
    shared.save_upload("b.txt", b"b")
    shared.rename("a.txt", "c.txt")
    assert (shared.root / "c.txt").exists() and not (shared.root / "a.txt").exists()
    with pytest.raises(FileExistsError):
        shared.rename("c.txt", "b.txt")


def test_trash_is_recoverable_and_hidden_from_the_listing(shared):
    shared.save_upload("keep.txt", b"important")
    trashed = shared.trash("keep.txt")
    assert not (shared.root / "keep.txt").exists() and trashed.read_bytes() == b"important"
    assert shared.list() == [] and [t.name for t in shared.list_trash()] == [trashed.name]
    shared.restore(trashed.name)
    assert (shared.root / "keep.txt").read_bytes() == b"important"


def test_open_in_finder_builds_a_reveal_command(shared):
    shared.save_upload("a.txt", b"a")
    calls = []
    shared.open_in_finder("a.txt", runner=lambda cmd, **kw: calls.append(cmd))
    assert calls == [["open", "-R", str(shared.root / "a.txt")]]


# --- the sandbox mount (real Docker) -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mounted(tmp_path_factory):
    root = tmp_path_factory.mktemp("mount") / "LayaWorkspace"
    root.mkdir()
    box = DockerSandbox(shared_dir=root)
    yield box, root
    box.stop()


def test_a_file_the_agent_writes_appears_on_the_mac_and_the_reverse(mounted):
    box, root = mounted
    box.execute("echo from-container > /workspace/shared/made_by_agent.txt")
    assert (root / "made_by_agent.txt").read_text().strip() == "from-container"
    (root / "from_user.txt").write_text("hello agent")
    assert box.execute("cat /workspace/shared/from_user.txt").output.strip() == "hello agent"


def test_the_agent_can_use_the_shared_folder_through_the_file_tools_too(mounted):
    box, root = mounted
    assert box.write("/workspace/shared/tool_made.txt", "via write_file").error is None
    assert (root / "tool_made.txt").read_text() == "via write_file"


def test_the_container_cannot_see_the_rest_of_the_mac(mounted):
    box, _ = mounted
    assert box.execute("ls /Users 2>&1; ls /workspace").output.count("amar") == 0


# --- delete policy: the agent never hard-deletes the user's files ------------------------------------------------


def _delete_call(path):
    return types.SimpleNamespace(tool_call={"id": "d1", "name": "delete", "args": {"file_path": path}}, state={})


def test_agent_delete_in_the_shared_folder_asks_then_moves_to_trash(shared, monkeypatch):
    shared.save_upload("doc.txt", b"precious")
    asked = []
    monkeypatch.setattr(policy, "interrupt", lambda payload: asked.append(payload) or "approve")
    mw = policy.PathPolicyMiddleware(shared=shared)
    ran = []
    r = mw.wrap_tool_call(_delete_call("/workspace/shared/doc.txt"), lambda req: ran.append(1))
    assert asked and "doc.txt" in str(asked[0]) and not ran  # the real delete tool never ran
    assert isinstance(r, ToolMessage) and "trash" in r.content.lower()
    assert not (shared.root / "doc.txt").exists() and len(shared.list_trash()) == 1


def test_a_rejected_delete_keeps_the_file(shared, monkeypatch):
    shared.save_upload("doc.txt", b"precious")
    monkeypatch.setattr(policy, "interrupt", lambda payload: "reject")
    r = policy.PathPolicyMiddleware(shared=shared).wrap_tool_call(_delete_call("/workspace/shared/doc.txt"), lambda req: None)
    assert r.status == "error" and (shared.root / "doc.txt").exists()


def test_deleting_scratch_files_outside_the_shared_folder_needs_no_approval(shared):
    ran = []
    mw = policy.PathPolicyMiddleware(shared=shared)
    mw.wrap_tool_call(_delete_call("/workspace/scratch.txt"), lambda req: ran.append(1) or ToolMessage(content="ok", tool_call_id="d1"))
    assert ran
