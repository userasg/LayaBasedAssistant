"""The bounded failure ladder, the ask-with-choices step, the stall guard and the decision cache.

These need no browser and no model: a fake screen, a fake ranker (every element equally, weakly similar, so nothing can be chosen with confidence) and a
predictor that fails the test if it is asked anything, because none of these paths may need one.
"""
import types

import numpy as np
import pytest

from laya_assistant import host
from laya_assistant.cua.cache import DecisionCache
from laya_assistant.cua.loop import Loop
from laya_assistant.cua.manager import ComputerUse, parse_choice
from laya_assistant.cua.planner import make_planner
from laya_assistant.cua.policy import DomainPolicy
from laya_assistant.cua.recorder import StepRecorder
from laya_assistant.cua.types import Element, Observation, StepPlan
from laya_assistant.session import AssistantSession


class WeakRanker:
    """Every element looks the same and only weakly like the step: below the match floor, above the 'plausible' floor."""

    def score(self, query, texts):
        return np.full(len(texts), 0.5)


class NoPredictor:
    def predict(self, *a, **k):
        raise AssertionError("no model may be asked on these paths")


class FakeScreen:
    name = "fake"
    source = "desktop"

    def __init__(self, *labels, works=True):
        self.els = [Element(f"e{i}", "textbox", lab) for i, lab in enumerate(labels)]
        self.works, self.acts = works, []

    def observe(self):
        els = [Element(e.id, e.role, e.label, e.value) for e in self.els]
        return Observation("desktop", "Editor", None, els, app="Editor", content=" ".join(e.value for e in self.els))

    def act(self, action):
        self.acts.append(action)
        if action.kind == "type" and self.works:
            next(e for e in self.els if e.id == action.id).value = action.value
        return {"ok": True}


def S(do, target="", value=None):
    return StepPlan(do, target, value)


class Scripted:
    def __init__(self, *plans):
        self.plans, self.calls = list(plans), []

    def __call__(self, goal, obs, done=None, failure=None, max_steps=8):
        self.calls.append((failure, list(done or [])))
        return self.plans.pop(0)


def make_cu(screen, *plans, cache=None, tmp_path=None):
    planner = Scripted(*plans)
    cu = ComputerUse(NoPredictor(), WeakRanker(), lambda t: planner, lambda t, app="": screen, DomainPolicy(tmp_path / "p.json"), StepRecorder(), cache=cache)
    return cu, planner


# -- the ladder ----------------------------------------------------------------------------------------------------------------------------

def test_an_unsettled_step_is_replanned_once_then_the_user_is_asked_with_options_and_nothing_else_may_act(tmp_path):
    screen = FakeScreen("Terminal input", "File name")
    ambiguous = [S("type", "the thing", "hello.py")]
    cu, planner = make_cu(screen, ambiguous, ambiguous, tmp_path=tmp_path)
    out = cu.run("desktop", "make a file", app="Editor")
    assert out.startswith("NEEDS_CHOICE [c1]") and "1. textbox: Terminal input" in out and "2. textbox: File name" in out and "0. none" in out
    assert len(planner.calls) == 2  # the first plan and ONE re-plan, never more
    assert screen.acts == []  # nothing was guessed
    assert cu.blocked_message() and "waiting for the user" in cu.blocked_message()


def test_the_users_pick_is_resolved_in_code_and_completes_the_task_and_is_remembered(tmp_path):
    screen, cache = FakeScreen("Terminal input", "File name"), DecisionCache()
    step = [S("type", "the thing", "hello.py")]
    cu, planner = make_cu(screen, step, step, cache=cache, tmp_path=tmp_path)
    cu.run("desktop", "make a file", app="Editor")
    out = cu.answer("the second one")
    assert out.startswith("DONE"), out
    assert screen.els[1].value == "hello.py" and screen.els[0].value == ""  # the chosen field, not the other
    assert len(planner.calls) == 2  # answering cost no model call
    assert cu.blocked_message() is None and not cu.pending_choices()
    assert any(r.get("kind") == "human_choice" and r["chosen_index"] == 2 for r in cu.rec.rows)
    assert cache.entries and next(iter(cache.entries.values()))["label"] == "File name"


def test_a_remembered_choice_is_replayed_with_no_question_and_no_model(tmp_path):
    screen, cache = FakeScreen("Terminal input", "File name"), DecisionCache()
    cache.put(S("type", "the thing"), screen.observe(), types.SimpleNamespace(element=screen.els[1]), "human")
    cu, planner = make_cu(screen, [S("type", "the thing", "again.py")], cache=cache, tmp_path=tmp_path)
    out = cu.run("desktop", "make a file", app="Editor")
    assert out.startswith("DONE") and screen.els[1].value == "again.py"
    assert cu.step_log[0]["tier"] == "cache" and len(planner.calls) == 1


def test_a_remembered_choice_that_stops_working_is_forgotten(tmp_path):
    screen, cache = FakeScreen("Terminal input", "File name", works=False), DecisionCache()
    step = S("type", "the thing", "x.py")
    cache.put(step, screen.observe(), types.SimpleNamespace(element=screen.els[1]), "human")
    loop = Loop(screen, NoPredictor(), WeakRanker(), DomainPolicy(tmp_path / "p.json"), StepRecorder(), cache=cache)
    r = loop.run([step])
    assert r.status == "failed" and not cache.entries  # the typed text never appeared: the entry is evicted, so it cannot fail the same way twice


def test_a_task_with_nothing_plausible_on_screen_stalls_with_a_report_and_refuses_a_repeat(tmp_path):
    screen = FakeScreen()  # an empty window: nothing to offer
    cu, planner = make_cu(screen, [S("click", "New File")], [S("click", "New File")], tmp_path=tmp_path)
    out = cu.run("desktop", "make a file", app="Editor")
    assert out.startswith("STALLED") and "Do NOT retry" in out and "mac_run" in out
    again = cu.run("desktop", "make a file", app="Editor")
    assert "already stopped" in again and len(planner.calls) == 2  # no third plan, no driver work
    assert cu.blocked_message()
    cu.new_request()  # the user says something new: the task may be tried again
    assert cu.blocked_message() is None


def test_a_repeated_identical_failure_skips_the_replan(tmp_path):
    screen = FakeScreen("Terminal input", "File name")
    step = [S("type", "the thing", "x")]
    cu, planner = make_cu(screen, step, step, step, tmp_path=tmp_path)
    cu.run("desktop", "g", app="Editor")
    cu.new_request()
    assert len(planner.calls) == 2  # plan + one replan; the identical second failure went straight to the question, not a third plan


@pytest.mark.parametrize("reply,want", [("2", 2), ("b", 2), ("the second one", 2), ("option 1", 1), ("first", 1), ("file name", 2), ("none", 0), ("Cancel.", 0),
                                        ("4", None), ("write me a poem", None), ("yes", None), ("term", 1)])
def test_replies_that_answer_the_question(reply, want):
    assert parse_choice(reply, ["textbox: Terminal input", "textbox: File name"]) == want


def test_a_reply_that_is_not_an_answer_is_left_for_the_agent(tmp_path):
    screen = FakeScreen("Terminal input", "File name")
    step = [S("type", "the thing", "x")]
    cu, _ = make_cu(screen, step, step, tmp_path=tmp_path)
    cu.run("desktop", "g", app="Editor")
    assert cu.answer("actually, open Safari") is None and cu.pending_choices()  # still waiting; the session drops it via new_request()


def test_none_of_these_cancels_without_touching_the_screen(tmp_path):
    screen = FakeScreen("Terminal input", "File name")
    step = [S("type", "the thing", "x")]
    cu, _ = make_cu(screen, step, step, tmp_path=tmp_path)
    cu.run("desktop", "g", app="Editor")
    assert cu.answer("none").startswith("CANCELLED") and screen.acts == []


# -- the other guards ---------------------------------------------------------------------------------------------------------------------------

def test_mac_run_is_refused_while_a_task_waits_for_the_user():
    ran = []
    tools = {t.name: t for t in host.make_host_tools(runner=lambda cmd, **kw: ran.append(cmd), guard=lambda: "NOT RUN: waiting for the user")}
    assert tools["mac_run"].invoke({"command": "osascript -e 'x'"}).startswith("NOT RUN") and ran == []


def test_the_planner_sees_only_the_last_few_finished_steps():
    seen = []

    class LLM:
        def invoke(self, prompt):
            seen.append(prompt)
            return types.SimpleNamespace(content='[{"do":"done"}]')

    plan = make_planner(LLM())
    plan("goal", Observation("desktop", "W", None, []), [f"step{i}" for i in range(10)], None)
    p = seen[0]
    assert "4 earlier steps" in p and "step9" in p and "step4" in p and "step3" not in p


def test_a_question_or_stall_is_shown_to_the_user_without_the_instructions_meant_for_the_agent():
    q = "NEEDS_CHOICE [c1]: I could not tell which control the step 'x' means. Screen: s.\nOptions:\n1. a\n2. b\n0. none of these\nAsk the user exactly this question"
    assert "Ask the user" not in AssistantSession._present(q) and "1. a" in AssistantSession._present(q)
    s = "STALLED: it broke. Done so far: nothing. Do NOT retry this task and do not work around it"
    assert AssistantSession._present(s).endswith("How would you like to proceed?") and "Do NOT" not in AssistantSession._present(s)


def test_the_session_answers_a_pending_question_itself_and_only_when_the_reply_is_an_answer():
    calls = []

    class Computer:
        def pending_choices(self):
            return [("c1", object())]

        def answer(self, text):
            calls.append(text)
            return "DONE (1 steps in 0.1s). Now showing x" if text == "2" else None

    s = AssistantSession.__new__(AssistantSession)
    s.computer, s.log = Computer(), types.SimpleNamespace(add=lambda *a, **k: None)
    assert s._answer_computer_question("2").startswith("DONE") and s._answer_computer_question("open Safari") is None and calls == ["2", "open Safari"]
    s.computer = None
    assert s._answer_computer_question("2") is None
