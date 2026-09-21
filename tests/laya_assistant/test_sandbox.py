"""Sandbox against REAL Docker. Each test class uses one container (session-scoped fixture) for speed."""
import shlex
import subprocess
import time

import pytest

from laya_assistant.sandbox import IMAGE, DockerSandbox, make_backend


@pytest.fixture(scope="module")
def box():
    b = DockerSandbox()
    yield b
    b.stop()


def test_write_read_edit_through_the_file_api(box):
    assert box.write("/workspace/hello.py", "print('hi')\n").error is None
    assert "print('hi')" in box.read("/workspace/hello.py").file_data["content"]
    assert box.edit("/workspace/hello.py", "hi", "there").error is None
    assert box.execute("python hello.py").output.strip() == "there"


def test_pytest_is_preinstalled_and_workdir_is_workspace(box):
    r = box.execute("python -m pytest --version && pwd")
    assert r.exit_code == 0 and "pytest" in r.output and "/workspace" in r.output


def test_upload_and_download_round_trip(box):
    blob = bytes(range(256)) * 4
    assert box.upload_files([("/workspace/uploads/blob.bin", blob)])[0].error is None
    got = box.download_files(["/workspace/uploads/blob.bin", "/workspace/missing.txt"])
    assert got[0].content == blob and got[1].error == "file_not_found"


def test_files_stay_inside_the_container(box):
    box.write("/workspace/only_in_container.txt", "x")
    assert not subprocess.run(["ls", "/workspace/only_in_container.txt"], capture_output=True).returncode == 0


def test_execute_latency_is_reported_and_reasonable(box):
    box.execute("true")
    ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        box.execute("true")
        ms.append((time.perf_counter() - t0) * 1000)
    print("docker exec ms:", [round(m) for m in ms])
    assert sorted(ms)[2] < 1500


def test_timeout_returns_a_failed_response_not_an_exception(box):
    r = box.execute("sleep 5", timeout=1)
    assert r.exit_code is None and "timed out" in r.output


def test_stop_removes_the_container():
    b = DockerSandbox()
    cid = b.id
    b.stop()
    time.sleep(1)
    out = subprocess.run(["docker", "ps", "-a", "-q", "--filter", f"id={cid}"], capture_output=True, text=True).stdout
    assert out.strip() == ""


def test_image_exists():
    assert subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


def test_fallback_backend_is_a_confined_host_shell_with_only_a_path(monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_KEY", "leak-me")
    handle = make_backend(prefer_docker=False)
    try:
        assert "host" in handle.label.lower()
        out = handle.backend.execute("echo ${SUPER_SECRET_KEY:-absent}").output
        assert "absent" in out and "leak-me" not in out
    finally:
        handle.stop()
