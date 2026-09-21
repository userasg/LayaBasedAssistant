"""Path policy for the agent's file tools.

Deep Agents' `permissions=` is not supported with backends that can run commands (a shell could bypass
any path rule), and the sandbox needs `execute`. So the same rules are enforced here, in code, for the
write-type file tools. The shell is confined by the container instead: `/skills` and `/memories` are
separate backends that are not mounted inside it.
"""
from pathlib import PurePosixPath

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from . import review

WRITE_TOOLS = {"write_file", "edit_file", "delete"}
WRITE_ALLOW = ("/workspace/", "/memories/", "/large_tool_results/", "/conversation_history/")
WRITE_DENY = ("/skills/",)


def write_allowed(path: str) -> bool:
    if not path.startswith("/") or ".." in PurePosixPath(path).parts:
        return False
    if any(path.startswith(d) for d in WRITE_DENY):
        return False
    return any(path.startswith(a) for a in WRITE_ALLOW)


SHARED_PREFIX = "/workspace/shared/"


class PathPolicyMiddleware(AgentMiddleware):
    def __init__(self, read_only: bool = False, shared=None):
        self.read_only = read_only
        self.shared = shared  # files.SharedFolder: the user's real folder, where deletes are never final

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] not in WRITE_TOOLS:
            return handler(request)
        path = str(call["args"].get("file_path", ""))
        if call["name"] == "delete" and path.startswith(SHARED_PREFIX) and self.shared is not None and not self.read_only:
            return self._trash_with_approval(call, path)
        if self.read_only:
            why = "this agent is read-only"
        elif not write_allowed(path):
            why = f"writing to {path!r} is not allowed; write under /workspace (files) or /memories (notes)"
        else:
            return handler(request)
        return ToolMessage(content=f"Denied: {why}.", tool_call_id=call["id"], status="error")

    def _trash_with_approval(self, call, path: str) -> ToolMessage:
        """These are the user's real files: ask first, and move to .trash instead of deleting."""
        rel = path[len(SHARED_PREFIX):]
        answer = review.decision(interrupt(review.request("delete", f"Move {rel} to the trash (you can restore it)", {"file": path, "path": rel})))
        refusal = review.declined(answer, f"moving {rel} to the trash")
        if refusal:
            return ToolMessage(content=f"{refusal} {rel} was not touched.", tool_call_id=call["id"], status="error")
        try:
            dest = self.shared.trash(rel)
        except (ValueError, FileNotFoundError) as e:
            return ToolMessage(content=f"Could not trash {rel}: {e}", tool_call_id=call["id"], status="error")
        return ToolMessage(content=f"Moved {rel} to the trash ({dest.name}); the user can restore it.", tool_call_id=call["id"])
