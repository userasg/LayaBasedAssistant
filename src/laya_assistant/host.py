"""Doing things on the Mac itself with plain scripts: open an app, open a URL, make a note, run a shell command.

Scripts are the simplest and fastest way to do most of what a person asks a computer to do, and they need no accessibility
permission. The sandbox cannot do this (it is a container walled off from the Mac on purpose), so these are host tools with
their own safety rules:
  * `mac_notes_create` passes the title and body as ARGUMENTS to a fixed AppleScript, never spliced into it (no injection);
  * `mac_run` runs a shell command on the Mac: obviously-safe commands (open, a curl GET, date, say, a short ls) run at once;
    anything else asks the user first; the hard-deny list (sudo, rm -rf /, pipe-to-shell, uploads) is refused outright;
    the shell gets a minimal environment (no API keys) and a timeout.
"""
from __future__ import annotations

import os
import re
import subprocess
import time

from langchain_core.tools import tool

from . import app_actions, approvals, config

HOST_SWITCH = "LAYA_HOST_ACTIONS"  # "off": the tools run nothing (test runs, demos). A run must never touch the real Notes app or open windows.
SWITCHED_OFF = f"SKIPPED: host actions are switched off in this run ({HOST_SWITCH}=off); nothing was run on the Mac."
SCRIPT_REFUSAL_S = 300  # after this long a script for an app that kept failing is allowed again
DUPLICATE_WINDOW_S = 600  # the same note (title and text) is created once per ten minutes: a re-planning agent must not litter Notes

NOTES_SCRIPT = '''on run argv
  tell application "Notes"
    make new note with properties {name:(item 1 of argv), body:(item 2 of argv)}
  end tell
  return "ok"
end run'''

# Commands that need no question: they open things, read public web pages, or print. Everything else asks.
_SAFE = [
    re.compile(r"^open -a ['\"]?[\w .+-]+['\"]?$"),  # no & ; | ` $ : an app name, never a second command
    re.compile(r"^open https?://[^\s;|&<>`$]+$"),
    re.compile(r"^open -R [~/][\w ./~-]*$"),
    re.compile(r"^(?:code|cursor|subl|zed)(?: -[rnw])? ['\"]?[~/][\w ./~-]*['\"]?$"),  # open a folder or file in an editor (it changes nothing)
    re.compile(r"^curl -[sSLfI]+ https?://[^\s;|&<>`$]+$"),
    re.compile(r"^(date|pwd|whoami|uptime)$"),
    re.compile(r"^say [\w ,.'!?-]{1,200}$"),
    re.compile(r"^ls( -[a-zA-Z]+)?( [\w ./~-]+)?$"),
]


def is_safe_host_command(command: str) -> bool:
    return any(rx.match(command.strip()) for rx in _SAFE)


def _env() -> dict:
    """A minimal environment: the shell must not inherit API keys or tokens from this process."""
    return {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin", "HOME": str(os.path.expanduser("~")), "LANG": "en_US.UTF-8"}


def notes_argv(title: str, body: str) -> list[str]:
    html = "".join(f"<div>{re.sub('<', '&lt;', line) or '<br>'}</div>" for line in body.splitlines()) or "<div><br></div>"
    return ["osascript", "-e", NOTES_SCRIPT, title[:200], html]


_CATALOG = None


def _catalog():
    global _CATALOG
    if _CATALOG is None:
        from .cua.apps import AppCatalog

        _CATALOG = AppCatalog()
    return _CATALOG


def make_host_tools(runner=None, guard=None, methods=None):
    """`runner` is `subprocess.run` unless a test passes a recorder. With no runner and LAYA_HOST_ACTIONS=off the tools do nothing.
    `guard()` returns a reason when `mac_run` must not act right now (a computer-use task is stalled and waiting for the user), else None.
    `methods` (cua/methods.py) is told which scripts worked, so the next request about that app starts with the way in that works."""
    off = runner is None and os.getenv(HOST_SWITCH, "on").strip().lower() == "off"
    runner = runner or subprocess.run
    made: dict[tuple[str, str], float] = {}  # (title, text) -> when it was created
    script_fails: dict[str, int] = {}  # app -> osascript failures in a row
    failed_at: dict[str, float] = {}  # app -> when the last one failed (a refusal lapses: the user may have fixed a permission)

    @tool
    def mac_notes_create(title: str, body: str) -> str:
        """Create a note in the Notes app on the user's Mac. Only when the user asked for a note: never as a side effect of another task,
        and never twice. Title and body are plain text (newlines for lines or list items). Write only facts you got from the user or read
        in a tool result; never put an unsourced number in a note."""
        if off:
            return SWITCHED_OFF
        key = (" ".join(title.split()).lower(), " ".join(body.split()).lower())
        seen = made.get(key)
        if seen is not None and time.monotonic() - seen < DUPLICATE_WINDOW_S:
            return f"Not created: a note titled {title!r} with exactly this text was already created {time.monotonic() - seen:.0f} s ago. Do not create it again."
        try:
            p = runner(notes_argv(title, body), capture_output=True, text=True, timeout=30, env=_env())
        except Exception as e:
            return f"FAILED: {type(e).__name__}: {e}"
        if p.returncode != 0:
            return f"FAILED: {p.stderr.strip()[:200]}"
        made[key] = time.monotonic()
        return f"Created the note {title!r} in Notes."

    @tool
    def mac_run(command: str, reason: str = "") -> str:
        """Run a shell command on the user's Mac (NOT the sandbox). Use it for things a script does best: `open -a Notes`,
        `open https://example.com`, `curl -sL https://...` to read a page, `osascript -e '...'`. Obviously safe commands run at once;
        anything else asks the user first (they answer in the text box); dangerous ones are refused. Say why in `reason`."""
        if off:
            return SWITCHED_OFF
        if guard is not None and (why := guard()):
            return why
        if "osascript" in command and (app := app_actions.osascript_app(command)) and script_fails.get(app, 0) >= app_actions.SWITCH_AFTER \
                and time.monotonic() - failed_at.get(app, 0) < SCRIPT_REFUSAL_S:
            # the fall-through is enforced here, not left to the model: a 14B that is told to switch keeps writing scripts
            return (f"NOT RUN: scripts for {app} failed {script_fails[app]} times just now. Do not write another: work in its window with "
                    f"computer_use(target='desktop', app='{app}').")
        try:
            p = runner(["/bin/zsh", "-c", command], capture_output=True, text=True, timeout=30, env=_env(), cwd=str(config.HOME))
        except subprocess.TimeoutExpired:
            return "FAILED: the command took longer than 30 s and was stopped."
        out = ((p.stdout or "") + (("\n[stderr] " + p.stderr) if p.stderr.strip() else ""))[:4000]
        advice = ""
        if "osascript" in command and (app := app_actions.osascript_app(command)):
            if methods is not None:
                methods.record(app, "script", p.returncode == 0)
            if p.returncode == 0:
                script_fails.pop(app, None)
            else:
                script_fails[app] = script_fails.get(app, 0) + 1
                failed_at[app] = time.monotonic()
                advice = app_actions.osascript_advice(p.stderr)
                if script_fails[app] >= app_actions.SWITCH_AFTER:  # stop paying a model turn per failed script: the UI is the way in
                    advice += f" Scripting {app} has failed {script_fails[app]} times: stop, and use computer_use(target='desktop', app='{app}')."
        return f"exit {p.returncode}\n{out}{chr(10) + '[hint] ' + advice.strip() if advice else ''}".strip()

    @tool(description=(
        "Do a common thing in a scriptable Mac app with a verified script (no AppleScript to write, and it reads the result back). "
        "Available: " + app_actions.describe() + ". `arg` is the playlist or song name for the play_* actions. Prefer this to mac_run + osascript "
        "and to computer_use whenever the app and action are listed; if it answers NO_TEMPLATE, use the other tools."))
    def mac_app_action(app: str, action: str, arg: str = "") -> str:
        if off:
            return SWITCHED_OFF
        if approvals.current.host_commands == "always_ask":
            return "NOT RUN: the user wants every command on their Mac approved first. Use mac_run so they are asked."
        resolved = _catalog().resolve(app)
        name = resolved.name if resolved else app
        if (name, action) not in app_actions.TEMPLATES:
            return f"NO_TEMPLATE: nothing ready for {action!r} in {name!r}. Ready-made: {app_actions.describe()}."
        r = app_actions.run(name, action, arg, runner=runner, env=_env())
        return r.text if r.ok else f"FAILED: {r.text}"

    @tool
    def mac_open_app(name: str) -> str:
        """Open any app on the user's Mac and bring it to the front, by the name people use ("Music", "Apple Music", "VS Code", "chrome").
        Use this first for "open <app>": it finds the installed app, shows its window (also when it was on another desktop) and tells you what
        it found. It does not click or type inside the app. For a scriptable app (the result says so) the most reliable way to DO things in it
        is AppleScript via mac_run (osascript), which works across desktops; otherwise use computer_use with target 'desktop' and app=<name>."""
        if off:
            return SWITCHED_OFF
        from .cua.apps import AppCatalog, bring_up
        from .cua.desktop import CuaCli

        catalog = _catalog()
        found = catalog.resolve(name)
        if found is None:
            return f"UNAVAILABLE: no installed app matches {name!r}. Do not guess another name; tell the user it is not installed."
        cli = CuaCli()
        try:
            presence = bring_up(found.name, cli)
        except RuntimeError:  # the driver is not running: still open it the plain way
            from .cua.apps import reopen_and_activate

            reopen_and_activate(found.name)
            return f"Opened {found.name} (window state not checked: the desktop driver is not running)."
        script = ("\n" + app_actions.method_note(found.name, catalog, methods)) if found.name not in app_actions._BROWSERS else ""
        if presence.ok:
            return f"Opened {found.name}; its window is on screen.{script}"
        return f"Opened {found.name}, but: {presence.reason}{script}"

    return [mac_notes_create, mac_app_action, mac_run, mac_open_app]
