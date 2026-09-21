"""Getting the chat window out of the way while an app is being driven.

Measured in real use: the chat lives in a browser window that stays in front (or on the Space the app needs), so the app being driven
sits behind it and clicks and window lookups stall. So for the length of a computer-use task the window that holds the chat is HIDDEN
(the app's own "Hide", reversible, nothing closes) and shown again the moment the task ends, whatever way it ends.
"""
from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager

NEVER_HIDE = {"Finder", "loginwindow", "Dock", "SystemUIServer", "Window Server", "cua-driver", "CuaDriver"}


def _osascript(script: str, *argv: str, runner=subprocess.run) -> str:
    p = runner(["osascript", "-e", "on run argv", *sum([["-e", line] for line in script.splitlines()], []), "-e", "end run", *argv],
               capture_output=True, text=True, timeout=8)
    return (p.stdout or "").strip()


def frontmost_app(runner=subprocess.run) -> str:
    try:
        return _osascript('tell application "System Events" to return name of first application process whose frontmost is true', runner=runner)
    except Exception:
        return ""


def set_hidden(app: str, hidden: bool, runner=subprocess.run) -> None:
    """Hide or show an app's windows. The name travels as an argument, never inside the script."""
    try:
        _osascript('tell application "System Events" to set visible of application process (item 1 of argv) to ' + ("false" if hidden else "true"), app, runner=runner)
        if not hidden:
            _osascript("tell application (item 1 of argv) to activate", app, runner=runner)
    except Exception:
        pass  # never let window tidying fail a task


@contextmanager
def chat_out_of_the_way(target_app: str = "", runner=subprocess.run):
    """Hide whatever app is in front (normally the browser showing the chat) for the duration, then bring it back. Inert when host actions
    are off, when the front app is the one being driven, or when it is a system process."""
    from .apps import host_actions_enabled

    front = frontmost_app(runner) if host_actions_enabled() else ""
    hide = bool(front) and front not in NEVER_HIDE and front.lower() != (target_app or "").lower()
    if hide:
        set_hidden(front, True, runner)
    try:
        yield front if hide else ""
    finally:
        if hide:
            set_hidden(front, False, runner)
