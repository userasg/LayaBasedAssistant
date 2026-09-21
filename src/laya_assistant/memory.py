"""Long-term memory: what the agent remembers about you across sessions.

Two layers, per the Deep Agents docs:
  * `AGENTS.md`  - ALWAYS loaded into the system prompt via `memory=`, so it stays short: your preferences,
                   conventions and standing instructions.
  * `/memories/` - a durable directory the agent reads on demand (session notes, research findings).
Both live on disk under ~/.laya_assistant/memories (routed by CompositeBackend), so they survive restarts.
Procedures belong in skills, not here.
"""
from pathlib import Path

from deepagents.backends import FilesystemBackend

from . import config

MEMORY_PATHS = ["/memories/AGENTS.md"]

SEED = """# What I know about the user
(Nothing yet. Add short, durable facts here: preferences, conventions, standing instructions.)
"""

MEMORY_PROMPT = """
## Long-term memory
Files under /memories/ persist across sessions:
- /memories/AGENTS.md        your always-loaded notes about the user. Keep it SHORT (one fact per line).
                             When the user states a preference or a standing instruction, add it here.
- /memories/sessions/<date>.md   what was done in a session and where artifacts are. Write one at the end of a big task.
- /memories/research/<topic>.md  findings worth keeping between sessions.
Never store secrets or passwords. Procedures and how-tos belong in skills, not memory.
"""


def memory_root() -> Path:
    return config.HOME / "memories"


def memory_backend() -> FilesystemBackend:
    root = memory_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "sessions").mkdir(exist_ok=True)
    (root / "research").mkdir(exist_ok=True)
    seed = root / "AGENTS.md"
    if not seed.exists():
        seed.write_text(SEED)
    return FilesystemBackend(root_dir=root, virtual_mode=True)
