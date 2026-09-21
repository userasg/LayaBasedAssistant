"""The agent's own filesystem: a per-session Docker container.

Every Deep Agents file tool (ls / read_file / write_file / edit_file / glob / grep) is built by
`BaseSandbox` on top of `execute()`, so implementing `execute`, `upload_files` and `download_files` gives the
agent a complete, isolated `/workspace`. One long-lived container per session (never `docker run` per
call), from an image built once, keeps each tool call at roughly one `docker exec`.

If Docker is unavailable, `make_backend` falls back to a confined host shell with a visible label.
"""
import atexit
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.sandbox import BaseSandbox, ExecuteResponse, FileDownloadResponse, FileUploadResponse

IMAGE = "laya-assistant-sandbox:1"
DOCKERFILE = Path(__file__).with_name("sandbox.Dockerfile")
WORKDIR = "/workspace"


def _docker_ready() -> bool:
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def ensure_docker_running(timeout: int = 90) -> None:
    """Start Docker Desktop if its daemon is down (macOS `open -a Docker`), polling until it answers."""
    if _docker_ready():
        return
    subprocess.run(["open", "-a", "Docker"], check=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _docker_ready():
            return
        time.sleep(2)
    raise RuntimeError(f"Docker daemon did not become ready within {timeout}s")


def ensure_image() -> None:
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0:
        return
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, "-f", str(DOCKERFILE), str(DOCKERFILE.parent)], check=True)


class DockerSandbox(BaseSandbox):
    def __init__(self, image: str = IMAGE, workdir: str = WORKDIR, shared_dir: Path | str | None = None):
        ensure_docker_running()
        ensure_image()
        self._workdir = workdir
        mount = []
        if shared_dir is not None:  # a real Mac folder the agent can read and write; nothing else is exposed
            Path(shared_dir).mkdir(parents=True, exist_ok=True)
            mount = ["-v", f"{Path(shared_dir).resolve()}:{workdir}/shared"]
        self._container_id = subprocess.run(
            ["docker", "run", "-d", "--rm", "--memory", "3g", "--cpus", "4", *mount, "-w", workdir, image, "sleep", "infinity"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        atexit.register(self.stop)

    @property
    def id(self) -> str:
        return self._container_id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        try:
            r = subprocess.run(
                ["docker", "exec", "-w", self._workdir, self._container_id, "sh", "-c", command],
                capture_output=True, text=True, timeout=timeout or 120,
            )
            return ExecuteResponse(output=r.stdout + r.stderr, exit_code=r.returncode)
        except subprocess.TimeoutExpired:
            return ExecuteResponse(output=f"Command timed out after {timeout or 120}s", exit_code=None)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        out = []
        for path in paths:
            r = subprocess.run(["docker", "exec", self._container_id, "cat", path], capture_output=True)
            out.append(FileDownloadResponse(path=path, content=r.stdout) if r.returncode == 0
                       else FileDownloadResponse(path=path, error="file_not_found"))
        return out

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        out = []
        for path, content in files:
            q = shlex.quote(path)
            r = subprocess.run(
                ["docker", "exec", "-i", self._container_id, "sh", "-c", f"mkdir -p $(dirname {q}) && cat > {q}"],
                input=content, capture_output=True,
            )
            out.append(FileUploadResponse(path=path, error=None if r.returncode == 0 else "write_error"))
        return out

    def stop(self) -> None:
        subprocess.run(["docker", "stop", "-t", "1", self._container_id], capture_output=True)


@dataclass
class SandboxHandle:
    backend: BackendProtocol
    label: str
    stop: Callable[[], None]


def make_backend(prefer_docker: bool = True, shared_dir: Path | str | None = None) -> SandboxHandle:
    if prefer_docker:
        try:
            box = DockerSandbox(shared_dir=shared_dir)
            return SandboxHandle(box, f"Docker sandbox {box.id[:12]}", box.stop)
        except Exception as e:  # Docker missing, daemon won't start, image build failed
            reason = f"Docker unavailable ({type(e).__name__})"
    else:
        reason = "Docker disabled"
    root = tempfile.mkdtemp(prefix="laya_assistant_host_")
    if shared_dir is not None:  # same /workspace/shared path in the host-shell fallback
        (Path(root) / "workspace").mkdir(exist_ok=True)
        Path(shared_dir).mkdir(parents=True, exist_ok=True)
        (Path(root) / "workspace" / "shared").symlink_to(Path(shared_dir).resolve())
    # Only a PATH that includes this venv, so `python -m pytest` works and no API keys leak to the shell.
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/local/bin"}
    return SandboxHandle(LocalShellBackend(root_dir=root, env=env), f"host shell (confined, {reason})", lambda: None)
