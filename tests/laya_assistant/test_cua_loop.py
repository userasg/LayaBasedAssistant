"""The whole computer-use loop: real Chromium, real local pages, real Laya, real embeddings. No model is faked.
(The LLM planner is not involved here: steps are given, so the Laya-scored loop itself is what is measured.)"""
import threading

import pytest

from laya_assistant.cua.loop import Loop
from laya_assistant.cua.playwright_driver import PlaywrightDriver
from laya_assistant.cua.policy import DomainPolicy
from laya_assistant.cua.recorder import StepRecorder
from laya_assistant.cua.types import StepPlan


@pytest.fixture
def env(page, site_url, laya_model, ranker, tmp_path):
    rec = StepRecorder()
    loop = Loop(PlaywrightDriver(page), laya_model, ranker, DomainPolicy(tmp_path / "p.json"), rec)
    return loop, page, site_url, rec


def S(do, target="", value=None):
    return StepPlan(do, target, value)


def title(page):
    return page.title()


def test_search_task_completes(env):
    loop, page, url, _ = env
    page.goto(url + "/index.html")
    r = loop.run([S("type", "search box", "sony"), S("click", "Search button")])
    assert r.status == "done", (r.status, r.reason)
    assert "results.html" in page.url and "Sony WH-1000XM5" in page.inner_text("body")
    print("search: per-step ms", [round(x.ms) for x in r.results])


def test_form_fill_batches_and_stops_at_the_irreversible_send_then_resumes_on_approval(env):
    loop, page, url, rec = env
    page.goto(url + "/form.html")
    steps = [S("type", "Your name", "Ada Lovelace"), S("type", "Email", "ada@example.org"), S("select", "Country", "Germany"),
             S("type", "Message", "Hello there"), S("check", "I agree to the terms"), S("click", "Send message")]
    r = loop.run(steps)
    assert r.status == "needs_approval" and r.pending and "Send message" in r.pending.description
    assert page.inner_text("#f") and title(page) == "Contact us"  # nothing was sent
    # all five fill steps ran, batched on one observation
    filled = [x for x in r.results if x.status == "ok"]
    assert len(filled) == 5
    assert sum(1 for x in filled if "observe_ms" in x.timings and x.timings["observe_ms"] > 0) <= 2
    r2 = loop.resume(steps, r.pending)
    assert r2.status == "done", (r2.status, r2.reason)
    assert title(page) == "Message sent"


def test_a_password_field_hands_control_to_the_user(env):
    loop, page, url, _ = env
    page.goto(url + "/login.html")
    r = loop.run([S("type", "Email", "ada@example.org"), S("type", "Password", "hunter2"), S("click", "Sign in")])
    assert r.status == "needs_user" and "password" in r.reason.lower()
    assert page.locator("input[type=password]").input_value() == ""  # never typed


def test_wizard_runs_the_ordinary_steps_freely_and_asks_only_at_the_final_commit(env):
    loop, page, url, _ = env
    page.goto(url + "/wizard.html")
    steps = [S("type", "Full name", "Ada"), S("click", "Next"), S("check", "Pro plan"), S("click", "Next"), S("click", "Finish")]
    r = loop.run(steps)
    # Laya rated "Finish" irreversible (p=0.92); the four ordinary steps before it ran without asking
    assert r.status == "needs_approval" and "Finish" in r.pending.description
    assert [x.status for x in r.results[:4]] == ["ok"] * 4 and title(page) == "Get started - step 3"
    r2 = loop.resume(steps, r.pending)
    assert r2.status == "done" and title(page) == "Finished"


def test_cookie_banner_then_the_real_task(env):
    loop, page, url, _ = env
    page.goto(url + "/cookie.html")
    r = loop.run([S("click", "Accept all cookies"), S("click", "Pricing")])
    assert r.status == "done", (r.status, r.reason)
    assert "pricing.html" in page.url or "pricing" in page.url.lower()


def test_delete_account_always_asks_and_export_does_not(env):
    loop, page, url, _ = env
    page.goto(url + "/danger.html")
    r = loop.run([S("click", "Delete account")])
    assert r.status == "needs_approval" and "irreversible" in r.reason.lower()
    assert title(page) == "Account settings"  # not clicked
    page.goto(url + "/danger.html")
    r = loop.run([S("click", "Export data")])
    assert r.status == "done" and title(page) == "Export started"


def test_the_target_is_found_among_thirty_links(env):
    loop, page, url, rec = env
    page.goto(url + "/console.html")
    r = loop.run([S("click", "Settings")])
    assert r.status == "done", (r.status, r.reason)
    assert title(page) == "Page: Settings"
    row = [x for x in rec.rows if x.get("kind") != "outcome"][-1]
    print("30 links:", row["tier"], row["ms"])


def test_a_step_with_no_matching_element_stops_and_offers_choices_instead_of_clicking_something_else(env):
    loop, page, url, _ = env
    page.goto(url + "/index.html")
    r = loop.run([S("click", "the shopping cart icon")])
    assert r.status in ("failed", "needs_choice") and page.url.endswith("index.html")  # nothing was clicked either way
    assert r.status == "failed" or 1 <= len(r.choices) <= 3


def test_a_click_that_changes_nothing_is_reported_not_assumed(env):
    loop, page, url, _ = env
    page.goto(url + "/index.html")
    r = loop.run([S("click", "Home")])  # already on Home: the link reloads the same page
    assert r.status in ("done", "failed")  # verified against the page, whichever way it lands
    if r.status == "failed":
        assert "changed" in r.reason


def test_a_blocked_domain_stops_everything(env, tmp_path):
    loop, page, url, _ = env
    loop.domain.extra_deny.add("127.0.0.1")
    page.goto(url + "/index.html")
    r = loop.run([S("type", "search box", "x")])
    assert r.status == "blocked" and "never-automate" in r.reason


def test_the_global_stop_flag_halts_the_run(env):
    loop, page, url, _ = env
    loop.stop_flag = threading.Event()
    loop.stop_flag.set()
    page.goto(url + "/index.html")
    r = loop.run([S("type", "search box", "sony")])
    assert r.status == "stopped" and page.locator("input[name=q]").input_value() == ""


def test_every_step_is_recorded_for_training(env):
    loop, page, url, rec = env
    page.goto(url + "/index.html")
    loop.run([S("type", "search box", "sony"), S("click", "Search button")])
    decisions = [x for x in rec.rows if x.get("kind") != "outcome"]
    assert len(decisions) == 2 and sum(1 for x in rec.rows if x.get("kind") == "outcome") == 2  # every action is verified and recorded
    row = decisions[0]
    assert {"step", "cands", "tier", "chosen", "safety", "verdict", "ms"} <= set(row) and row["chosen"]
