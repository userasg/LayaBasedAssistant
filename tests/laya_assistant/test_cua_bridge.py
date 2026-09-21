"""The Chrome extension and its local bridge, tested by loading the REAL extension into a REAL Chromium."""
import tempfile
import time
from pathlib import Path

import pytest

from laya_assistant.cua.bridge import BridgeDriver, BridgeServer
from laya_assistant.cua.loop import Loop
from laya_assistant.cua.policy import DomainPolicy
from laya_assistant.cua.types import Action, StepPlan

EXT = Path(__file__).parents[2] / "src" / "laya_assistant" / "cua" / "extension"
TOKEN = "test-token-123"
PORT = 8797


@pytest.fixture(scope="module")
def chrome(site_url, pw):
    server = BridgeServer(port=PORT, token=TOKEN)
    server.start()
    ctx = pw.chromium.launch_persistent_context(
        tempfile.mkdtemp(prefix="laya_ext_"), headless=False,
        args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}", "--headless=new"])
    sw = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker", timeout=15000)
    sw.evaluate(f"laya.setToken('{TOKEN}', {PORT})")
    assert server.wait_connected(15), "the extension did not connect to the bridge"
    yield server, ctx, sw
    ctx.close()
    server.stop()


_n = [0]


def open_tab(chrome, url):
    """A fresh tab, found by a unique URL so an earlier test's tab is never picked up by mistake."""
    server, ctx, sw = chrome
    _n[0] += 1
    url = f"{url}?tab={_n[0]}"
    page = ctx.new_page()
    page.goto(url)
    tab_id = sw.evaluate("async (u) => { const [t] = await chrome.tabs.query({url: u}); return t.id; }", url)
    return page, tab_id


def test_a_wrong_token_is_refused(chrome):
    import json

    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    with connect(f"ws://127.0.0.1:{PORT}/bridge") as ws:
        ws.send(json.dumps({"type": "hello", "token": "nope"}))
        with pytest.raises(ConnectionClosed) as e:
            ws.recv(timeout=3)
    assert e.value.rcvd.code == 4401


def test_an_unclaimed_tab_cannot_be_observed_or_acted_on(chrome, site_url):
    server, ctx, sw = chrome
    page, tab_id = open_tab(chrome, site_url + "/index.html")
    with pytest.raises(RuntimeError, match="not one you handed"):
        server.request("observe", tab_id)
    with pytest.raises(RuntimeError, match="not one you handed"):
        server.request("act", tab_id, {"kind": "click", "id": "e1"})


def test_claiming_a_tab_shows_a_banner_and_makes_it_controllable(chrome, site_url):
    server, ctx, sw = chrome
    page, tab_id = open_tab(chrome, site_url + "/index.html")
    sw.evaluate(f"laya.claim({tab_id})")
    time.sleep(0.6)
    assert page.locator("#__laya_banner").count() == 1 and "controlling this tab" in page.inner_text("#__laya_banner")
    assert any(t["id"] == tab_id and t["claimed"] for t in server.tabs)
    obs = BridgeDriver(server, tab_id).observe()
    assert obs.title == "Acme Shop - Home" and any(e.label == "Search" for e in obs.elements)


def test_acting_in_a_claimed_tab_types_clicks_and_navigates(chrome, site_url):
    server, ctx, sw = chrome
    page, tab_id = open_tab(chrome, site_url + "/index.html")
    sw.evaluate(f"laya.claim({tab_id})")
    drv = BridgeDriver(server, tab_id)
    obs = drv.observe()
    box = next(e for e in obs.elements if e.label == "Search products")
    assert drv.act(Action("type", id=box.id, value="sony"))["ok"]
    drv.act(Action("click", id=next(e for e in drv.observe().elements if e.label == "Search").id))
    after = drv.observe()
    assert "results.html" in after.url
    drv.act(Action("navigate", url=site_url + "/form.html"))
    assert drv.observe().title == "Contact us"


def test_the_extension_refuses_to_type_into_password_fields(chrome, site_url):
    server, ctx, sw = chrome
    page, tab_id = open_tab(chrome, site_url + "/login.html")
    sw.evaluate(f"laya.claim({tab_id})")
    drv = BridgeDriver(server, tab_id)
    pw = next(e for e in drv.observe().elements if e.type == "password")
    assert drv.act(Action("type", id=pw.id, value="x")) == {"ok": False, "error": "password_field"}


def test_release_all_removes_control_and_the_banner(chrome, site_url):
    server, ctx, sw = chrome
    page, tab_id = open_tab(chrome, site_url + "/index.html")
    sw.evaluate(f"laya.claim({tab_id})")
    server.release_all()
    time.sleep(0.6)
    assert page.locator("#__laya_banner").count() == 0
    with pytest.raises(RuntimeError, match="not one you handed"):
        server.request("observe", tab_id)


def test_the_screenshot_of_a_claimed_tab_is_a_png(chrome, site_url):
    server, ctx, sw = chrome
    page, tab_id = open_tab(chrome, site_url + "/index.html")
    page.bring_to_front()
    sw.evaluate(f"laya.claim({tab_id})")
    png = BridgeDriver(server, tab_id).screenshot()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_full_loop_runs_through_the_real_extension(chrome, site_url, laya_model, ranker, tmp_path):
    server, ctx, sw = chrome
    page, tab_id = open_tab(chrome, site_url + "/index.html")
    sw.evaluate(f"laya.claim({tab_id})")
    loop = Loop(BridgeDriver(server, tab_id), laya_model, ranker, DomainPolicy(tmp_path / "p.json"))
    r = loop.run([StepPlan("type", "search box", "sony"), StepPlan("click", "Search button")])
    assert r.status == "done", (r.status, r.reason)
    print("through the extension: per-step ms", [round(x.ms) for x in r.results])
    page.wait_for_url("**q=sony**", timeout=5000)  # Playwright learns of the extension-triggered navigation a moment later
    assert "q=sony" in page.url, page.url


def test_there_is_no_bridge_connection_error_is_actionable():
    s = BridgeServer(port=8798, token="x")
    with pytest.raises(RuntimeError, match="Laya Bridge extension"):
        s.request("observe", 1)


@pytest.mark.slow
def test_the_deep_agent_operates_the_browser_through_the_extension(chrome, site_url, laya_model, ranker, tmp_path):
    """Everything real: Laya intake -> 14B agent -> computer_use tool -> real planner -> fast loop -> extension -> Chromium tab."""
    from laya_assistant.cua.manager import ComputerUse
    from laya_assistant.cua.planner import make_planner
    from laya_assistant.session import AssistantSession

    server, ctx, sw = chrome
    page, tab_id = open_tab(chrome, site_url + "/index.html")
    sw.evaluate(f"laya.claim({tab_id})")
    cu = ComputerUse(laya_model, ranker, lambda t: make_planner(), lambda t, app="": BridgeDriver(server, tab_id), DomainPolicy(tmp_path / "p.json"))
    s = AssistantSession.create(laya_model, warm=False, computer=cu, shared_dir=tmp_path / "shared")
    try:
        s.submit("Use my browser: type sony into the shop's search box and press the Search button.")
        for _ in range(12):
            while s.busy:
                time.sleep(0.2)
            if not s.pending:
                break
            # approve the PLAN only; any other approval (a first visit to a new site, an irreversible step) is refused
            reqs = s.pending["action_requests"]
            s.submit_resume([{"type": "approve"} if r["name"] == "write_todos" else {"type": "reject"} for r in reqs])
        page.wait_for_url(lambda u: "q=sony" in u.lower(), timeout=20000)
        kinds = [e.kind for e in s.feed()]
        print("feed:", kinds[-6:], "| steps:", cu.step_log)
        assert "q=sony" in page.url.lower() and cu.step_log and all(x["status"] == "ok" for x in cu.step_log)
    finally:
        s.close()


def test_a_web_page_cannot_connect_to_the_bridge_even_with_a_guessed_token(chrome):
    from websockets.exceptions import InvalidStatus
    from websockets.sync.client import connect

    with pytest.raises(InvalidStatus) as e:
        connect(f"ws://127.0.0.1:{PORT}/bridge", origin="https://evil.example")
    assert e.value.response.status_code == 403
