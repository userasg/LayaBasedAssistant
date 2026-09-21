"""Planner parsing (pure) and the goal runner: real Chromium + real Laya + a scripted planner, plus one slow test with the real 14B."""
import pytest

from laya_assistant.cua.manager import ComputerUse
from laya_assistant.cua.planner import make_planner, parse_plan
from laya_assistant.cua.playwright_driver import PlaywrightDriver
from laya_assistant.cua.policy import DomainPolicy
from laya_assistant.cua.tools import make_tools
from laya_assistant.cua.types import StepPlan


def test_parse_plan_extracts_steps_from_noisy_model_output():
    text = 'Sure! Here is the plan:\n```json\n[{"do":"type","target":"search box","value":"sony"},{"do":"click","target":"Search"},{"do":"done"}]\n```'
    steps = parse_plan(text)
    assert [(s.do, s.target, s.value) for s in steps] == [("type", "search box", "sony"), ("click", "Search", None), ("done", "", None)]


@pytest.mark.parametrize("bad", ["no json here", "[]", '[{"do":"teleport","target":"x"}]', '[{"do":'])
def test_parse_plan_rejects_unusable_output(bad):
    with pytest.raises(ValueError):
        parse_plan(bad)


def test_parse_plan_caps_length_and_normalises_submit():
    raw = "[" + ",".join('{"do":"click","target":"a%d"}' % i for i in range(20)) + "]"
    assert len(parse_plan(raw, max_steps=5)) == 5
    assert parse_plan('[{"do":"submit","target":"query"}]')[0].do == "press_enter"


class Scripted:
    """A planner that returns fixed plans in order (stands in for the LLM only; Laya and the loop are real)."""

    def __init__(self, *plans):
        self.plans, self.calls = list(plans), []

    def __call__(self, goal, obs, done=None, failure=None, max_steps=8):
        self.calls.append((goal, failure, list(done or [])))
        return self.plans.pop(0)


@pytest.fixture
def cu_env(page, site_url, laya_model, ranker, tmp_path):
    def make(*plans):
        planner = Scripted(*plans)
        cu = ComputerUse(laya_model, ranker, lambda target: planner, lambda target, app="": PlaywrightDriver(page), DomainPolicy(tmp_path / "p.json"))
        return cu, planner
    return make, page, site_url


def S(do, target="", value=None):
    return StepPlan(do, target, value)


def test_a_goal_runs_to_done_with_one_plan_and_no_replanning(cu_env):
    make, page, url = cu_env
    cu, planner = make([S("type", "search box", "sony"), S("click", "Search button")])
    page.goto(url + "/index.html")
    out = cu.run("browser", "find sony headphones")
    assert out.startswith("DONE") and "Sony WH-1000XM5" in out
    assert len(planner.calls) == 1 and len(cu.step_log) == 2


def test_a_failed_step_hands_back_to_the_planner_with_the_reason_and_recovers(cu_env):
    make, page, url = cu_env
    cu, planner = make([S("click", "the shopping cart icon")],  # nothing matches
                       [S("type", "search box", "sony"), S("click", "Search button")])
    page.goto(url + "/index.html")
    out = cu.run("browser", "find sony headphones")
    assert out.startswith("DONE")
    assert len(planner.calls) == 2 and planner.calls[1][1]  # the second plan was told why the first failed
    assert "could not tell" in planner.calls[1][1] or "no candidate" in planner.calls[1][1]


def test_replanning_is_capped(cu_env):
    make, page, url = cu_env
    bad = [S("click", "the shopping cart icon")]
    cu, planner = make(bad, bad, bad, bad)
    page.goto(url + "/index.html")
    assert cu.run("browser", "x").startswith("FAILED") and len(planner.calls) == 3  # first plan + 2 replans


def test_needs_approval_parks_the_exact_action_and_confirm_performs_it(cu_env):
    make, page, url = cu_env
    cu, _ = make([S("type", "Your name", "Ada"), S("type", "Email", "ada@example.org"), S("click", "Send message")])
    page.goto(url + "/form.html")
    out = cu.run("browser", "send a message")
    assert out.startswith("NEEDS_APPROVAL") and page.title() == "Contact us"
    pid = out.split("[")[1].split("]")[0]
    done = cu.confirm(pid)
    assert done.startswith("DONE") and page.title() == "Message sent"
    assert cu.confirm(pid).startswith("FAILED")  # a pending action can run once


def test_cancel_drops_the_pending_action(cu_env):
    make, page, url = cu_env
    cu, _ = make([S("click", "Delete account")])
    page.goto(url + "/danger.html")
    pid = cu.run("browser", "delete").split("[")[1].split("]")[0]
    assert cu.cancel(pid) == "CANCELLED." and page.title() == "Account settings"


def test_needs_user_for_passwords_is_reported_as_a_handover(cu_env):
    make, page, url = cu_env
    cu, _ = make([S("type", "Email", "a@b.co"), S("type", "Password", "x")])
    page.goto(url + "/login.html")
    out = cu.run("browser", "log in")
    assert out.startswith("NEEDS_USER") and "password" in out.lower()


def test_the_tools_wrap_the_manager_and_report_unavailable_surfaces(laya_model, ranker):
    def no_surface(target, app=""):
        raise RuntimeError("no Chrome tab is connected")

    cu = ComputerUse(laya_model, ranker, lambda t: None, no_surface)
    use, confirm, cancel = make_tools(cu)
    assert use.invoke({"target": "browser", "goal": "x"}).startswith("UNAVAILABLE: no Chrome tab")
    assert cancel.invoke({"pending_id": "zzz"}).startswith("FAILED")


@pytest.mark.slow
def test_the_real_planner_plans_and_the_fast_loop_executes_a_search(page, site_url, laya_model, ranker, tmp_path):
    cu = ComputerUse(laya_model, ranker, lambda t: make_planner(), lambda t, app="": PlaywrightDriver(page), DomainPolicy(tmp_path / "p.json"))
    page.goto(site_url + "/index.html")
    out = cu.run("browser", "search the shop for sony headphones")
    print(out, "| steps:", cu.step_log)
    assert out.startswith("DONE") and "q=sony" in page.url.lower()  # it typed the query and submitted it


def test_an_unavailable_target_is_not_retried_and_tells_the_agent_to_switch_method(laya_model, ranker):
    calls = []

    def factory(target, app=""):
        calls.append(target)
        raise RuntimeError("Google Chrome exposes almost no accessibility information")

    cu = ComputerUse(laya_model, ranker, lambda t: None, factory)
    use = make_tools(cu)[0]
    first = use.invoke({"target": "my_chrome", "goal": "x"})
    second = use.invoke({"target": "my_chrome", "goal": "x"})
    assert first.startswith("UNAVAILABLE: Google Chrome")
    assert "do NOT call computer_use" in second and "mac_run" in second
    assert calls == ["my_chrome"]  # the second call never even tried: no more retry loops
    assert use.invoke({"target": "desktop", "goal": "x"}).startswith("UNAVAILABLE")  # a different target is tried on its own
    assert calls == ["my_chrome", "desktop"]


def test_the_default_browser_target_needs_no_setup(laya_model, ranker, tmp_path):
    from laya_assistant.cua.own_browser import BrowserWorker, OwnBrowserDriver
    from laya_assistant.cua.wiring import make_computer_use

    cu = make_computer_use(laya_model, ranker, bridge=None, worker=BrowserWorker(tmp_path / "prof", headless=True))
    try:
        d = cu.driver_factory("browser", "")
        assert isinstance(d, OwnBrowserDriver)
        assert cu._loop(d).domain is cu.own_domain  # no first-visit prompts in the isolated browser
        from laya_assistant.cua.desktop import DesktopDriver

        try:
            mine = cu.driver_factory("my_chrome", "")  # your own Chrome needs the extension or the Cua Driver permission...
            assert isinstance(mine, DesktopDriver)  # ...and if the permission is granted on this machine, it gets the desktop driver
        except RuntimeError as e:
            assert "target='browser' works without" in str(e)  # otherwise the message says what is missing and what works now
    finally:
        cu.worker.stop()
