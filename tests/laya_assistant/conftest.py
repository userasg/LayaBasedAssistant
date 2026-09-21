import pytest

from laya_assistant.engine import load_laya


@pytest.fixture(scope="session")
def laya_model():
    """The real Laya checkpoint, loaded once per test session (about 20 s)."""
    return load_laya()


# --- a real browser and a local site for the computer-use tests ---------------------------------------------------

import functools
import http.server
import threading
from pathlib import Path

SITE = Path(__file__).parent / "site"


@pytest.fixture(scope="session")
def site_url():
    """The fixture pages, served on localhost."""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(SITE)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(scope="session")
def pw():
    """ONE Playwright for the whole run. Its sync API keeps an asyncio loop alive in the main thread, so it cannot be started
    twice, and any other test that needs an event loop must run it in its own thread (see run_async)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(pw):
    b = pw.chromium.launch(headless=True)
    yield b
    b.close()


def run_async(coro):
    """Run a coroutine to completion in a fresh thread with its own event loop (safe next to Playwright's loop)."""
    import asyncio
    import threading

    box = {}

    def work():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as e:  # re-raised in the caller
            box["error"] = e

    t = threading.Thread(target=work)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1100, "height": 800})
    pg = ctx.new_page()
    yield pg
    ctx.close()


@pytest.fixture(scope="session")
def ranker():
    from laya_assistant.cua.menu import Ranker

    r = Ranker()
    r.warm()
    return r


@pytest.fixture(scope="session", autouse=True)
def _headless_own_browser():
    """Tests never pop windows on the screen."""
    import os

    os.environ["LAYA_BROWSER_HEADLESS"] = "1"
    yield


@pytest.fixture(scope="session", autouse=True)
def _no_real_host_actions():
    """End-to-end tests let the real 14B agent choose its own tools. It once picked mac_notes_create and filled the user's real Notes app
    with test notes, so the host tools do nothing in test runs. A test that needs the real shell passes `runner=subprocess.run` itself."""
    import os

    os.environ["LAYA_HOST_ACTIONS"] = "off"
    yield


@pytest.fixture(autouse=True)
def _default_autonomy():
    """Every test starts from the shipped defaults (ask as little as possible) and cannot leak a switch to the next one."""
    from laya_assistant import approvals

    approvals.current.plan, approvals.current.host_commands, approvals.current.new_sites, approvals.current.windows = False, "risky", False, True
    yield
    approvals.current.plan, approvals.current.host_commands, approvals.current.new_sites, approvals.current.windows = False, "risky", False, True


@pytest.fixture
def plan_approval_on():
    from laya_assistant import approvals

    approvals.current.plan = True
    yield
