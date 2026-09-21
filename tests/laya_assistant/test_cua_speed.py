"""The point of the design, measured: the fast loop vs an agent that asks the 14B LLM to choose every target.
Same pages, same steps, same menu; only who decides differs. Real Chromium, real Laya, real 14B."""
import time

import pytest

from laya_assistant import config
from laya_assistant.cua import menu as menu_mod
from laya_assistant.cua.loop import Loop, to_action
from laya_assistant.cua.playwright_driver import PlaywrightDriver
from laya_assistant.cua.policy import DomainPolicy
from laya_assistant.cua.types import StepPlan as S
from laya_assistant.cua.wiring import make_tiebreak
from laya_assistant.engine import make_llm

pytestmark = pytest.mark.slow

TASKS = [
    ("/index.html", [S("type", "search box", "sony"), S("click", "Search button")]),
    ("/wizard.html", [S("type", "Full name", "Ada"), S("click", "Next"), S("check", "Pro plan"), S("click", "Next")]),
    ("/form.html", [S("type", "Your name", "Ada"), S("type", "Email", "ada@example.org"), S("select", "Country", "Germany"),
                    S("type", "Message", "Hello"), S("check", "I agree to the terms")]),
]
N_STEPS = sum(len(s) for _, s in TASKS)


def run_fast(page, url, laya_model, ranker, tmp_path):
    total, tiers = 0.0, []
    for path, steps in TASKS:
        page.goto(url + path)
        loop = Loop(PlaywrightDriver(page), laya_model, ranker, DomainPolicy(tmp_path / "p.json"))
        t0 = time.perf_counter()
        r = loop.run(steps)
        total += (time.perf_counter() - t0) * 1000
        assert r.status == "done", (path, r.status, r.reason)
        tiers += [x.decision.tier for x in r.results]
    return total, tiers


def run_llm_every_step(page, url, ranker, pick):
    total = 0.0
    for path, steps in TASKS:
        page.goto(url + path)
        drv = PlaywrightDriver(page)
        t0 = time.perf_counter()
        for st in steps:
            obs = drv.observe()
            cands = menu_mod.build(st, obs, ranker)[:4]
            chosen = pick(st, obs, cands) or cands[0]  # the LLM decides EVERY target
            assert drv.act(to_action(chosen, st))["ok"]
        drv.observe()  # closing look
        total += (time.perf_counter() - t0) * 1000
    return total


def make_full_page_picker(llm):
    """The conventional agent: show the LLM the whole page and ask it to name the element (no pre-narrowed menu)."""
    import re

    from laya_assistant.cua.types import Candidate
    from laya_assistant.engine import llm_lock

    def pick(step, obs):
        prompt = f"{obs.summary(40)}\n\nStep: {step.text()}\nWhich element id performs this step? Answer with just the id, like e7."
        with llm_lock():
            text = str(llm.invoke(prompt).content)
        m = re.search(r"e\d+", text)
        el = obs.by_id(m.group()) if m else None
        return Candidate(el.id, "click" if step.do == "click" else step.do, el, step.value, 1.0) if el else None

    return pick


def run_llm_full_page(page, url, ranker, pick_full):
    total = 0.0
    for path, steps in TASKS:
        page.goto(url + path)
        drv = PlaywrightDriver(page)
        t0 = time.perf_counter()
        for st in steps:
            obs = drv.observe()
            chosen = pick_full(st, obs) or menu_mod.build(st, obs, ranker)[0]
            assert drv.act(to_action(chosen, st))["ok"]
        drv.observe()
        total += (time.perf_counter() - t0) * 1000
    return total


def test_the_fast_loop_beats_an_llm_that_decides_every_step(page, site_url, laya_model, ranker, tmp_path):
    pick = make_tiebreak(make_llm(config.EXECUTOR_MODEL, temperature=0.0))
    pick(S("click", "Search"), type("O", (), {"title": "warm"})(), menu_mod.build(S("click", "x"), type("Ob", (), {"elements": []})(), ranker) or [])  # no-op warm
    from laya_assistant.cua.types import Candidate, Element
    pick(S("click", "Next"), type("O", (), {"title": "warm"})(), [Candidate("e1", "click", Element("e1", "button", "Next"), None, 0.9)])  # loads the 14B
    full = make_full_page_picker(make_llm(config.EXECUTOR_MODEL, temperature=0.0))
    full(S("click", "Next"), type("O", (), {"summary": lambda self, n=0: "page", "by_id": lambda self, i: None})())  # warm
    fast, tiers = run_fast(page, site_url, laya_model, ranker, tmp_path)
    kind = run_llm_every_step(page, site_url, ranker, pick)
    realistic = run_llm_full_page(page, site_url, ranker, full)
    per_fast, per_kind, per_real = fast / N_STEPS, kind / N_STEPS, realistic / N_STEPS
    print(f"\nSPEED over {N_STEPS} steps: fast loop {per_fast:.0f} ms/step (tiers: {tiers.count('code')} code, {tiers.count('laya')} Laya, {tiers.count('llm')} LLM)"
          f"\n  LLM picks from the same 4-candidate menu (kindest baseline): {per_kind:.0f} ms/step = {per_kind / per_fast:.1f}x slower"
          f"\n  LLM shown the whole page (conventional agent):               {per_real:.0f} ms/step = {per_real / per_fast:.1f}x slower")
    assert per_fast < 400
    # measured over several runs on these small pages: kindest baseline 1.8x-3.4x (it varies with the model's cache state),
    # whole-page baseline 4.0x. The margins below are under every measurement so this fails only on a real regression.
    assert per_kind / per_fast >= 1.3
    assert per_real / per_fast >= 2.5
