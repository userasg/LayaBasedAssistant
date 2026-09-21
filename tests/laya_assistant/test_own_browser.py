"""The agent-owned browser: real Chrome/Chromium, a saved profile, used from several threads like the agent's tools do."""
import threading

import pytest

from laya_assistant.cua.loop import Loop
from laya_assistant.cua.own_browser import BrowserWorker, OwnBrowserDriver
from laya_assistant.cua.policy import DomainPolicy
from laya_assistant.cua.types import Action, StepPlan


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    w = BrowserWorker(tmp_path_factory.mktemp("profile"), headless=True)
    yield w
    w.stop()


def test_it_starts_with_no_setup_and_navigates(worker, site_url):
    d = OwnBrowserDriver(worker)
    assert d.act(Action("navigate", url=site_url + "/index.html"))["ok"]
    obs = d.observe()
    assert obs.title == "Acme Shop - Home" and any(e.label == "Search" for e in obs.elements)


def test_it_can_be_driven_from_other_threads_the_way_the_agents_tools_are(worker, site_url):
    d = OwnBrowserDriver(worker)
    d.act(Action("navigate", url=site_url + "/wizard.html"))
    results = {}

    def work(name):  # Playwright objects are thread-bound: this would raise without the dedicated browser thread
        results[name] = d.observe().title

    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert set(results.values()) == {"Get started - step 1"}


def test_the_whole_loop_runs_on_it_from_a_worker_thread(worker, site_url, laya_model, ranker, tmp_path):
    out = {}

    def run():
        d = OwnBrowserDriver(worker)
        d.act(Action("navigate", url=site_url + "/index.html"))
        loop = Loop(d, laya_model, ranker, DomainPolicy(tmp_path / "p.json"))
        out["r"] = loop.run([StepPlan("type", "search box", "sony"), StepPlan("click", "Search button")])
        out["url"] = d.observe().url

    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert out["r"].status == "done", (out["r"].status, out["r"].reason) and "q=sony" in out["url"]


def test_logins_persist_in_the_profile_between_runs(tmp_path, site_url):
    prof = tmp_path / "profile"
    w1 = BrowserWorker(prof, headless=True)
    w1.call(lambda d, ctx: (d.page.goto(site_url + "/index.html"), ctx.add_cookies([{"name": "session", "value": "abc", "url": site_url, "expires": 4102444800}])))  # a login cookie has an expiry
    w1.stop()
    w2 = BrowserWorker(prof, headless=True)
    try:
        cookies = w2.call(lambda d, ctx: ctx.cookies())
        assert any(c["name"] == "session" and c["value"] == "abc" for c in cookies)
    finally:
        w2.stop()


def test_closing_the_window_does_not_break_the_agent_it_opens_a_new_one(tmp_path, site_url):
    w = BrowserWorker(tmp_path / "p", headless=True)
    try:
        d = OwnBrowserDriver(w)
        d.act(Action("navigate", url=site_url + "/index.html"))
        w.call(lambda drv, ctx: ctx.close())  # the user closes the browser
        assert d.observe().source == "browser"  # transparently reopened
    finally:
        w.stop()


@pytest.mark.slow
def test_the_real_agent_starts_from_a_blank_browser_it_owns_and_gets_it_done(site_url, laya_model, ranker, tmp_path):
    """Real 14B agent, real planner, real Laya, real Chrome window (headless in tests) that the app opened itself: no setup."""
    from laya_assistant.cua.wiring import make_computer_use
    from laya_assistant.session import AssistantSession

    w = BrowserWorker(tmp_path / "profile", headless=True)
    cu = make_computer_use(laya_model, ranker, bridge=None, worker=w)
    s = AssistantSession.create(laya_model, warm=False, computer=cu, shared_dir=tmp_path / "shared")
    try:
        s.submit(f"Open {site_url}/index.html in the browser and search the shop for sony.")
        for _ in range(8):
            while s.busy:
                import time as _t
                _t.sleep(0.2)
            if not s.pending:
                break
            s.answer_pending("yes")  # answered in words, like the text box
        url = w.call(lambda d, c: d.page.url)
        print("browser ended at:", url, "| steps:", [x["step"][:40] for x in cu.step_log])
        assert "q=sony" in url.lower()
    finally:
        s.close()
        w.stop()
