"""A real Chromium driven through Playwright: used for tests and for a headless, disposable browsing mode.
It runs the SAME observe.js as the Chrome extension, so what is tested here is what runs in your Chrome."""
from __future__ import annotations

import time
from pathlib import Path

from .driver import observation_from_js
from .types import Action, Observation

OBSERVE_JS = (Path(__file__).parent / "extension" / "observe.js").read_text()


class PlaywrightDriver:
    name = "playwright"

    def __init__(self, page):
        self.page = page
        self.page.add_init_script(OBSERVE_JS)  # survives navigations

    def _ensure(self):
        if not self.page.evaluate("typeof window.__laya !== 'undefined'"):
            self.page.evaluate(OBSERVE_JS)

    def observe(self) -> Observation:
        t0 = time.perf_counter()
        self.page.wait_for_load_state("domcontentloaded")
        self._ensure()
        raw = self.page.evaluate("window.__laya.observe()")
        return observation_from_js(raw, (time.perf_counter() - t0) * 1000)

    def act(self, action: Action) -> dict:
        if action.kind == "navigate":
            self.page.goto(action.url)
            return {"ok": True}
        if action.kind == "wait":
            self.page.wait_for_timeout(int((action.dy or 500)))
            return {"ok": True}
        self._ensure()
        payload = {"kind": action.kind, "id": action.id, "value": action.value, "key": action.key, "dy": action.dy}
        try:
            res = self.page.evaluate("(a) => window.__laya.act(a)", payload)
        except Exception as e:  # navigation during the click destroys the JS context: the click itself worked
            if "context was destroyed" in str(e) or "Execution context" in str(e):
                res = {"ok": True}
            else:
                raise
        self.page.wait_for_timeout(60)  # let handlers and client-side rendering settle
        return res

    def screenshot(self) -> bytes:
        return self.page.screenshot()

    def close(self) -> None:
        pass
