"""Which way of working in an app has actually worked on this Mac: a script, or its own screen. The router (`app_actions.method_note`) starts from the
user's ladder (script for a scriptable app, command line where an app has one, the on-screen loop otherwise) and this memory moves an app to the rung
that works, so a bad first choice is paid for once, not on every request.

A script that a 14B model writes freehand fails in ways the app's UI does not: measured with Messages ("Can't get participant ... whose name contains",
-1728, then syntax errors, then asking the user for a phone number), while the same task through the window was one search, one click, one line of text.
`PRIOR_UI` is that measurement as a starting point; real outcomes override it in either direction.
"""
from __future__ import annotations

import json
from pathlib import Path

PRIOR_UI = {"Messages"}  # scriptable, but sending to a contact by script failed on this Mac; the window works. A recorded script success overrides this.
MIN_FAILS = 2  # a rung is written off after this many failures with more failures than successes


class MethodMemory:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.stats: dict[str, dict[str, dict[str, int]]] = {}
        if path and path.exists():
            try:
                self.stats = json.loads(path.read_text())
            except (OSError, ValueError):
                self.stats = {}

    def _s(self, app: str, rung: str) -> dict[str, int]:
        return self.stats.get(app, {}).get(rung, {"ok": 0, "fail": 0})

    def record(self, app: str, rung: str, ok: bool) -> None:
        row = self.stats.setdefault(app, {}).setdefault(rung, {"ok": 0, "fail": 0})
        row["ok" if ok else "fail"] += 1
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.stats))
            except OSError:
                pass  # an optimisation: never fail a request over it

    def _bad(self, app: str, rung: str) -> bool:
        s = self._s(app, rung)
        return s["fail"] >= MIN_FAILS and s["fail"] > s["ok"]

    def prefers_ui(self, app: str) -> bool:
        """True when this app should be worked through its window first: scripts have failed here (or are known to), and the window has not."""
        if self._bad(app, "ui"):
            return False
        script = self._s(app, "script")
        return self._bad(app, "script") or (app in PRIOR_UI and script["ok"] == 0)
