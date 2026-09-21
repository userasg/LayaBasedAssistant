"""Finding a Mac app by what a person calls it, and bringing it up so it can be driven.

Two things went wrong before, measured on the real Music app:
  * Cua Driver's `launch_app` deliberately does NOT activate the app (`self_activation_suppressed`), so the process runs but its window is
    reported off screen, and "Music has no visible window" was all the agent ever heard.
  * A window that sits on another Space cannot be read (`ax_unresolved`, empty element tree). What works is what a person does: open the
    app, tell it to reopen and activate (macOS then switches to its Space), and wait until the window is on screen. After that the driver
    read 239 controls.
So `bring_up` does exactly that, and says WHY when it cannot, so the agent stops guessing app names.
"""
from __future__ import annotations

import os
import plistlib
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

APP_DIRS = ("/Applications", "/System/Applications", "/System/Applications/Utilities", "/Applications/Utilities",
            "/System/Library/CoreServices", str(Path.home() / "Applications"))

# What people call things, mapped to the real app name. Anything else is matched against the installed apps.
ALIASES = {
    "apple music": "Music", "itunes": "Music", "music app": "Music", "settings": "System Settings", "system preferences": "System Settings",
    "preferences": "System Settings", "chrome": "Google Chrome", "vscode": "Visual Studio Code", "vs code": "Visual Studio Code", "code": "Visual Studio Code",
    "word": "Microsoft Word", "excel": "Microsoft Excel", "powerpoint": "Microsoft PowerPoint", "outlook": "Microsoft Outlook",
    "messages": "Messages", "imessage": "Messages", "text messages": "Messages", "mail": "Mail", "email": "Mail", "calendar": "Calendar",
    "photos": "Photos", "notes": "Notes", "reminders": "Reminders", "maps": "Maps", "tv": "TV", "apple tv": "TV", "podcasts": "Podcasts",
    "facetime": "FaceTime", "terminal": "Terminal", "calculator": "Calculator", "calc": "Calculator", "preview": "Preview",
    "textedit": "TextEdit", "text edit": "TextEdit", "safari": "Safari", "finder": "Finder", "app store": "App Store", "keynote": "Keynote",
    "pages": "Pages", "numbers": "Numbers", "spotify": "Spotify", "slack": "Slack", "zoom": "zoom.us", "whatsapp": "WhatsApp", "notion": "Notion",
}
_FILLER = re.compile(r"\b(the|my|apple|app|application|please|program)\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", s.lower())).strip()


@dataclass
class Resolved:
    name: str
    how: str  # exact | alias | fuzzy
    score: float = 1.0
    alternatives: list[str] = field(default_factory=list)


class AppCatalog:
    """The apps installed on this Mac, from the folders apps live in (no model, no driver: ~ms)."""

    def __init__(self, dirs=APP_DIRS):
        self.paths: dict[str, str] = {}  # real name -> .app path
        for d in dirs:
            if os.path.isdir(d):
                for entry in os.listdir(d):
                    if entry.endswith(".app"):
                        self.paths.setdefault(entry[:-4], os.path.join(d, entry))
        self._low = {_norm(n): n for n in self.paths}

    def resolve(self, spoken: str) -> Resolved | None:
        raw = _norm(spoken)
        if not raw:
            return None
        if raw in self._low:
            return Resolved(self._low[raw], "exact")
        if raw in ALIASES and (ALIASES[raw] in self.paths or not self.paths):
            return Resolved(ALIASES[raw], "alias")
        core = _norm(_FILLER.sub(" ", raw))
        if core in self._low:
            return Resolved(self._low[core], "exact")
        if core in ALIASES and ALIASES[core] in self.paths:
            return Resolved(ALIASES[core], "alias")
        scored = sorted(((self._score(core, low), name) for low, name in self._low.items()), reverse=True)
        if scored and scored[0][0] >= 0.75:
            return Resolved(scored[0][1], "fuzzy", scored[0][0], [n for s, n in scored[1:4] if s >= 0.5])
        return None

    @staticmethod
    def _score(spoken: str, installed: str) -> float:
        a, b = set(spoken.split()), set(installed.split())
        if not a or not b:
            return 0.0
        overlap = len(a & b) / len(a | b)
        contains = 0.85 if (spoken in installed or installed in spoken) and min(len(spoken), len(installed)) >= 4 else 0.0
        return max(overlap, contains)

    def scripting(self, name: str) -> bool:
        """Does the app ship a scripting dictionary? Then AppleScript is the most reliable way to drive it (it works across Spaces)."""
        path = self.paths.get(name)
        if not path:
            return False
        if list(Path(path, "Contents/Resources").glob("*.sdef")):
            return True
        try:
            info = plistlib.loads(Path(path, "Contents/Info.plist").read_bytes())
            return bool(info.get("NSAppleScriptEnabled"))
        except Exception:
            return False


@dataclass
class Presence:
    ok: bool
    app: str
    pid: int | None = None
    window_id: int | None = None
    reason: str = ""


def host_actions_enabled() -> bool:
    return os.getenv("LAYA_HOST_ACTIONS", "on").strip().lower() != "off"


def reopen_and_activate(app: str, runner=subprocess.run, foreground: bool = True) -> None:
    """What a person does by hand: open it, then tell it to reopen and come to the front (macOS switches to its Space). The name travels as
    an argument, never inside the script text. A slow app (or a permission prompt) must not fail the turn: what matters is whether a
    window appears, which the caller checks. With foreground=False it is only launched (`open -g`): nothing is raised or switched to."""
    cmds = [["open", "-g", "-a", app]] if not foreground else [
        ["open", "-a", app],
        ["osascript", "-e", "on run argv", "-e", "tell application (item 1 of argv) to reopen", "-e", "tell application (item 1 of argv) to activate",
         "-e", "end run", app]]
    for cmd in cmds:
        try:
            runner(cmd, capture_output=True, timeout=15)
        except subprocess.TimeoutExpired:
            pass


def _candidates(cli, app: str) -> list[dict]:
    """The app's real windows, biggest and titled first (apps also own tiny helper and menu-bar surfaces)."""
    wins = [w for w in cli.call("list_windows", {"on_screen_only": False}).get("windows", [])
            if str(w.get("app_name", "")).lower() == app.lower() and (w.get("bounds") or {}).get("width", 0) > 100
            and (w.get("bounds") or {}).get("height", 0) > 60]
    return sorted(wins, key=lambda w: (bool(w.get("title")), w["bounds"]["width"] * w["bounds"]["height"], w.get("z_index") or 0), reverse=True)


def bring_up(app: str, cli, wait: float = 12.0, launcher=reopen_and_activate, foreground: bool = True) -> Presence:
    """Open the app and wait for a window whose controls the driver can actually read.

    Measured: after macOS switches Space the window is "on screen" a couple of seconds BEFORE its accessibility tree resolves (the driver
    returns an empty tree, `ax_window_unresolved`, for ~5 s), and an app may own several windows of which only one resolves. So this polls
    until some candidate window returns controls, and only then says it is ready."""
    if not host_actions_enabled():
        return Presence(False, app, reason="host actions are switched off in this run")
    try:
        launcher(app) if foreground else launcher(app, foreground=False)
    except subprocess.TimeoutExpired:
        pass  # the app was slow to answer AppleScript (or is showing a permission prompt): what matters is whether a window appears
    deadline = time.time() + wait
    seen_hidden = seen_any = seen_shown = False
    while True:
        wins = _candidates(cli, app)
        seen_any = seen_any or bool(wins)
        for w in wins:
            if not w.get("is_on_screen"):
                seen_hidden = seen_hidden or w.get("on_current_space") is False
                continue
            seen_shown = True
            st = cli.call("get_window_state", {"pid": w["pid"], "window_id": w["window_id"], "include_screenshot": False})
            if st.get("elements"):
                return Presence(True, app, int(w["pid"]), int(w["window_id"]))
        if time.time() > deadline:
            break
        time.sleep(0.5)
    if not foreground and not seen_shown:
        return Presence(False, app, reason=(f"{app} is running in the background but its window is not on this desktop, and 'show app windows' is "
                                            "off, so it was not brought forward. Turn that setting on (Autonomy) or switch to the app yourself."))
    if seen_shown:
        reason = (f"{app}'s window is showing but its controls could not be read (an app with a custom-drawn or empty accessibility tree). "
                  "Use AppleScript (mac_run osascript) if it is scriptable, or keyboard shortcuts.")
    elif seen_hidden:
        reason = (f"{app} is running but its window is on another desktop (Space) and would not come forward. Ask the user to switch to it, "
                  "or drive it with AppleScript (mac_run osascript) which works across Spaces.")
    elif seen_any:
        reason = f"{app} has a window but it stayed hidden or minimised. Try AppleScript (mac_run osascript) or ask the user to unhide it."
    else:
        reason = (f"{app} started but opened no window (it may be a menu-bar app, or still loading). Try again in a moment, or use AppleScript "
                  "(mac_run osascript) if it is scriptable.")
    return Presence(False, app, reason=reason)
