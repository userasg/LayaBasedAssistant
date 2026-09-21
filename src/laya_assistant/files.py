from __future__ import annotations

"""The shared folder: a real Mac folder the agent can read and write through the sandbox mount.

`~/LayaWorkspace` is bind-mounted at `/workspace/shared`. Files the agent makes appear in Finder; files you drop
there are visible to it. Nothing else on the Mac is exposed. Deleting moves a file to `.trash` (recoverable),
never removes it.
"""
import csv
import io
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config

TEXT_SUFFIXES = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml", ".html", ".css", ".js", ".ts", ".sh", ".log", ".xml", ".ini", ".cfg", ".sql"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
PREVIEW_CHARS = 20_000
CSV_ROWS = 50


@dataclass
class FileInfo:
    rel: str
    size: int
    mtime: float
    is_dir: bool = False


@dataclass
class Preview:
    kind: str  # text | csv | image | binary
    text: str = ""
    rows: list = field(default_factory=list)
    data: bytes = b""
    truncated: bool = False


class SharedFolder:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or config.SHARED_DIR).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.trash_dir = self.root / ".trash"

    def resolve(self, rel: str) -> Path:
        """A path inside the folder, or ValueError. Absolute paths and `..` are refused."""
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or not rel:
            raise ValueError(f"path outside the shared folder: {rel!r}")
        full = (self.root / p).resolve()
        if self.root.resolve() not in full.parents and full != self.root.resolve():
            raise ValueError(f"path outside the shared folder: {rel!r}")
        return full

    def list(self, limit: int = 200) -> list[FileInfo]:
        out = []
        for p in self.root.rglob("*"):
            rel = p.relative_to(self.root)
            if any(part.startswith(".") for part in rel.parts) or p.is_dir():
                continue
            st = p.stat()
            out.append(FileInfo(str(rel), st.st_size, st.st_mtime))
        return sorted(out, key=lambda f: f.mtime, reverse=True)[:limit]

    def save_upload(self, name: str, data: bytes) -> Path:
        base = re.sub(r"[^\w.\- ]", "_", Path(name or "").name).strip(". ") or "upload"
        target = self.root / base
        n = 1
        while target.exists():
            target = self.root / f"{Path(base).stem}-{n}{Path(base).suffix}"
            n += 1
        target.write_bytes(data)
        return target

    def preview(self, rel: str) -> Preview:
        p = self.resolve(rel)
        suffix = p.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            return Preview("image", data=p.read_bytes())
        raw = p.read_bytes()[: PREVIEW_CHARS * 4 + 1]
        if suffix in TEXT_SUFFIXES or (suffix == ".csv"):
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return Preview("binary")
            truncated = p.stat().st_size > len(raw) or len(text) > PREVIEW_CHARS
            if suffix == ".csv":
                rows = list(csv.reader(io.StringIO(text)))[:CSV_ROWS]
                return Preview("csv", rows=rows, truncated=truncated)
            return Preview("text", text=text[:PREVIEW_CHARS], truncated=truncated)
        return Preview("binary")

    def rename(self, rel: str, new_name: str) -> Path:
        src = self.resolve(rel)
        dst = self.resolve(str(Path(rel).with_name(Path(new_name).name)))
        if dst.exists():
            raise FileExistsError(dst.name)
        return src.rename(dst)

    def trash(self, rel: str) -> Path:
        src = self.resolve(rel)
        self.trash_dir.mkdir(exist_ok=True)
        dst = self.trash_dir / f"{time.strftime('%Y%m%d-%H%M%S')}__{rel.replace('/', '__')}"
        shutil.move(str(src), str(dst))
        return dst

    def list_trash(self) -> list[Path]:
        return sorted(self.trash_dir.glob("*")) if self.trash_dir.exists() else []

    def restore(self, trash_name: str) -> Path:
        src = self.trash_dir / trash_name
        rel = trash_name.split("__", 1)[1].replace("__", "/")
        dst = self.resolve(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = dst.with_name(f"{dst.stem}-restored{dst.suffix}")
        shutil.move(str(src), str(dst))
        return dst

    def open_in_finder(self, rel: str, runner=subprocess.run) -> None:
        runner(["open", "-R", str(self.resolve(rel))], check=False)
