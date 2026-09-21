"""A browser the agent owns: a visible Chrome window with its own saved profile (~/.laya_assistant/browser).

Why this is the default: nothing to install or switch on. Chrome only exposes a page's contents to the macOS accessibility
tree when accessibility is enabled inside Chrome (measured: not exposed, even after 12 s), so driving your everyday Chrome
that way fails out of the box. A browser the app launches itself needs no extension, no accessibility setting and no macOS
permission, sees the real DOM, and keeps its cookies between sessions (log in once, it stays logged in). Your own Chrome
remains available through the optional extension.

Playwright's sync API is bound to the thread that created it, but the agent calls tools from arbitrary worker threads, so one
dedicated thread owns the browser and everything else hands it closures.
"""
from __future__ import annotations

import concurrent.futures
import os
import queue
import threading
from pathlib import Path

from .. import config
from .playwright_driver import PlaywrightDriver
from .types import Action, Observation


class BrowserWorker:
    def __init__(self, profile: Path | str | None = None, headless: bool | None = None):
        self.profile = Path(profile or config.HOME / "browser")
        self.headless = headless if headless is not None else os.getenv("LAYA_BROWSER_HEADLESS", "0") == "1"
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()
            self._error = None
            self._q = queue.Queue()
            self._thread = threading.Thread(target=self._run, name="own-browser", daemon=True)
            self._thread.start()
        self._ready.wait(60)
        if self._error:
            raise RuntimeError(f"could not start the browser: {self._error}")

    def _run(self) -> None:
        from playwright.sync_api import sync_playwright

        try:
            with sync_playwright() as p:
                self.profile.mkdir(parents=True, exist_ok=True)
                kw = dict(headless=self.headless, args=["--window-size=1280,900", "--no-first-run", "--no-default-browser-check"],
                          viewport={"width": 1280, "height": 900} if self.headless else None)
                try:
                    ctx = p.chromium.launch_persistent_context(str(self.profile), channel="chrome", **kw)  # your installed Chrome, separate profile
                except Exception:
                    ctx = p.chromium.launch_persistent_context(str(self.profile), **kw)  # bundled Chromium
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                driver = PlaywrightDriver(page)
                self._ready.set()
                while True:
                    item = self._q.get()
                    if item is None:
                        break
                    fn, fut = item
                    try:
                        fut.set_result(fn(driver, ctx))
                    except BaseException as e:
                        fut.set_exception(e)
                ctx.close()
        except Exception as e:
            self._error = e
            self._ready.set()

    def call(self, fn, timeout: float = 90.0):
        """Run fn(driver, context) on the browser thread. Restarts the browser once if you closed its window."""
        for attempt in (0, 1):
            self.start()
            fut: concurrent.futures.Future = concurrent.futures.Future()
            self._q.put((fn, fut))
            try:
                return fut.result(timeout)
            except Exception as e:
                if attempt == 0 and ("closed" in str(e).lower() or "Target page" in str(e)):
                    self.stop()  # the window was closed: open a fresh one
                    continue
                raise

    def stop(self) -> None:
        with self._lock:
            t = self._thread
        if t is not None and t.is_alive():
            self._q.put(None)
            t.join(15)
        self._thread = None


class OwnBrowserDriver:
    name = "browser"
    own = True  # an isolated browser with no logins of yours: the first-visit approval is not needed (blocked sites still are)

    def __init__(self, worker: BrowserWorker):
        self.w = worker

    def observe(self) -> Observation:
        return self.w.call(lambda d, c: d.observe())

    def act(self, action: Action) -> dict:
        return self.w.call(lambda d, c: d.act(action))

    def screenshot(self) -> bytes:
        return self.w.call(lambda d, c: d.screenshot())

    def close(self) -> None:
        pass
