"""Acting while you are still talking.

The live transcript changes as you speak, so a clause is only acted on once it is (a) followed by a boundary ("and",
"then", a comma, a full stop) and (b) identical in two consecutive transcript updates (the standard streaming-ASR
"local agreement" test). Only a small whitelist of REVERSIBLE commands runs early: open an app, go to a site you have
approved, scroll. Laya checks each clause for risk first. Anything else, including everything after the first clause
that is not on the whitelist, waits until you stop, then goes through the normal plan-and-approve path. Clauses commit
strictly in order, so nothing runs out of sequence.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .cua.menu import lexical
from .cua.policy import DomainPolicy

_BOUNDARY = re.compile(r"\s*(?:,|;|\.|\?|!|\band then\b|\bthen\b|\bafter that\b|\band\b)\s*", re.I)


def split_clauses(text: str) -> tuple[list[str], bool]:
    """(clauses, ends_with_boundary). The last clause is only trustworthy when the text ends with a boundary."""
    text = (text or "").strip()
    if not text:
        return [], False
    parts = [p.strip() for p in _BOUNDARY.split(text)]
    clauses = [p for p in parts if p]
    ends = bool(_BOUNDARY.search(text[-12:])) and bool(re.search(r"(,|;|\.|\?|!|\band then\b|\bthen\b|\bafter that\b|\band)\s*$", text, re.I))
    return clauses, ends


class ClauseStreamer:
    """Feed successive partial transcripts; get back newly committed clauses, in order."""

    def __init__(self):
        self.committed: list[str] = []
        self._prev: list[str] = []

    def feed(self, partial: str, stop_at_first_unrecognised=None) -> list[str]:
        clauses, ends = split_clauses(partial)
        norm = [c.lower() for c in clauses]
        stable_upto = 0
        for i, c in enumerate(norm):
            if i < len(self._prev) and self._prev[i] == c and (i < len(norm) - 1 or ends):
                stable_upto = i + 1
            else:
                break
        self._prev = norm
        new = []
        for c in clauses[len(self.committed):stable_upto]:
            if self.committed and self.committed[-1] is None:
                break
            if stop_at_first_unrecognised is not None and not stop_at_first_unrecognised(c):
                self.committed.append(None)  # a clause we will not act early on: nothing after it may run early either
                break
            self.committed.append(c)
            new.append(c)
        return new

    def acted(self) -> list[str]:
        return [c for c in self.committed if c]

    def remaining(self, final_text: str) -> tuple[str, list[str]]:
        """What is left to do once you stop: (remaining text, clauses that already ran). If the final transcript no longer
        starts with what was executed (the recogniser revised itself), everything is treated as remaining."""
        done = self.acted()
        clauses, _ = split_clauses(final_text)
        if done and [c.lower() for c in clauses[: len(done)]] == [d.lower() for d in done]:
            rest = clauses[len(done):]
            return " and ".join(rest), done
        return final_text, []


# -- what may run early -------------------------------------------------------------------------------------------

APP_DIRS = ("/Applications", "/System/Applications", "/System/Applications/Utilities", "/Applications/Utilities",
            "/System/Library/CoreServices", str(Path.home() / "Applications"))
# Opening these is harmless in itself, but they are the tools that make damage easy: never opened by an early command.
EARLY_DENY_APPS = {"terminal", "iterm", "iterm2", "script editor", "automator", "keychain access", "system settings", "system preferences",
                   "activity monitor", "disk utility", "console", "shortcuts"}
_OPEN = re.compile(r"^(?:please\s+)?(?:can you\s+)?(?:open|launch|start|switch to|bring up|show me)\s+(?:the\s+|my\s+)?(?P<what>.+?)(?:\s+(?:app|application))?$", re.I)
_GOTO = re.compile(r"^(?:please\s+)?(?:go to|open|visit|navigate to|browse to)\s+(?P<url>(?:https?://)?[\w.-]+\.[a-z]{2,}(?:/\S*)?)$", re.I)
_SCROLL = re.compile(r"^(?:please\s+)?scroll\s+(?P<dir>up|down)(?:\s+a\s+(?:bit|little))?$", re.I)


@dataclass
class QuickCommand:
    kind: str  # open_app | navigate | scroll
    value: str
    clause: str


class AppIndex:
    """Installed apps by what people call them (aliases, then fuzzy). See cua/apps.py."""

    def __init__(self, dirs=APP_DIRS):
        from .cua.apps import AppCatalog

        self.catalog = AppCatalog(dirs)
        self.names = {n.lower(): n for n in self.catalog.paths}

    def resolve(self, spoken: str) -> str | None:
        r = self.catalog.resolve(spoken)
        return r.name if r else None


def parse_quick(clause: str, apps: AppIndex) -> QuickCommand | None:
    c = clause.strip()
    if m := _SCROLL.match(c):
        return QuickCommand("scroll", m.group("dir").lower(), clause)
    if m := _GOTO.match(c):
        url = m.group("url")
        return QuickCommand("navigate", url if url.startswith("http") else "https://" + url, clause)
    if m := _OPEN.match(c):
        app = apps.resolve(m.group("what"))
        if app:
            return QuickCommand("open_app", app, clause)
    return None


@dataclass
class QuickResult:
    ok: bool
    text: str
    clause: str
    kind: str = ""
    ms: float = 0.0


def default_launcher(app: str) -> None:
    """Open it and bring it forward (a bare `open -a` leaves some apps windowless or on another Space). Inert when host actions are off."""
    import os

    from .cua.apps import reopen_and_activate

    if os.getenv("LAYA_HOST_ACTIONS", "on").strip().lower() == "off":
        return
    reopen_and_activate(app)


class QuickExecutor:
    """Runs whitelisted reversible commands. `predictor` (Laya) gates each clause; `browser` is an optional BridgeDriver."""

    def __init__(self, predictor=None, apps: AppIndex | None = None, domain: DomainPolicy | None = None, launcher=default_launcher,
                 browser=None, risky_at: float = 0.8):
        self.predictor, self.apps = predictor, apps or AppIndex()
        self.domain, self.launcher, self.browser, self.risky_at = domain or DomainPolicy(), launcher, browser, risky_at

    def recognises(self, clause: str) -> bool:
        cmd = parse_quick(clause, self.apps)
        return cmd is not None and self._allowed(cmd) is None

    def _allowed(self, cmd: QuickCommand) -> str | None:
        """None if it may run early, else why not."""
        if cmd.kind == "navigate":
            v = self.domain.check(cmd.value)
            if v.action != "allow":
                return "needs approval" if v.action == "ask" else v.reason  # unknown sites are asked about after you stop
            if not self.browser:
                return "no Chrome tab is connected"
        if cmd.kind == "scroll" and not self.browser:
            return "no Chrome tab is connected"
        if cmd.kind == "open_app" and cmd.value.lower() in EARLY_DENY_APPS:
            return f"{cmd.value} is not opened by early commands"
        # Laya's risk score is noisy on short commands (it rates "open the calculator" 0.65), so it can only veto when very sure.
        if self.predictor is not None:
            from .intake import classify

            if classify(self.predictor, cmd.clause).risky >= self.risky_at:
                return "Laya rates it risky"
        return None

    def execute(self, clause: str) -> QuickResult | None:
        import time

        cmd = parse_quick(clause, self.apps)
        if cmd is None or self._allowed(cmd) is not None:
            return None
        t0 = time.perf_counter()
        try:
            if cmd.kind == "open_app":
                self.launcher(cmd.value)
                text = f"opened {cmd.value}"
            elif cmd.kind == "navigate":
                from .cua.types import Action

                self.browser.act(Action("navigate", url=cmd.value))
                text = f"went to {cmd.value}"
            else:
                from .cua.types import Action

                self.browser.act(Action("scroll", dy=-500 if cmd.value == "up" else 500))
                text = f"scrolled {cmd.value}"
            return QuickResult(True, text, clause, cmd.kind, (time.perf_counter() - t0) * 1000)
        except Exception as e:  # a failed early action must not break the stream
            return QuickResult(False, f"could not {cmd.kind.replace('_', ' ')} {cmd.value}: {e}", clause, cmd.kind, (time.perf_counter() - t0) * 1000)
