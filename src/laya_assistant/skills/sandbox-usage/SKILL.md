---
name: sandbox-usage
description: "Facts about the agent's private Docker sandbox: paths, installed tools, what persists, and how to hand files to the user. Use before running commands or saving files."
---

# sandbox-usage

- Your workspace is `/workspace` inside a private container. Use relative paths; the shell starts there.
- Python 3.13 and `pytest` are installed. `pip install` works but is slow: prefer the standard library.
- Files you save in `/workspace` are downloadable by the user; uploads appear in `/workspace/uploads/`.
- `/workspace/shared` is the user's real Mac folder (`~/LayaWorkspace`): save deliverables there so they show up in Finder, and read files they drop there. To delete, use the `delete` tool (it asks first and moves to a recoverable trash); shell `rm` there is blocked.
- Large tool results are saved to `/workspace/.large_tool_results/<id>.txt`: use `read_file` or `grep` on them.
- The container is deleted when the session ends. Anything worth keeping between sessions goes in `/memories/`.
- Commands that delete, overwrite or upload data are checked by a safety layer and may be blocked: pick a safer approach or ask the user.
