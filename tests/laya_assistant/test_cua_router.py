"""What the Messages run exposed, as tests: verification that survives a UI re-index, plans that stay on the surface, planning in chunks, and the
script -> command line -> window router. Fake screens and fake planners: no model, no browser, no real Mac.
"""
import json
import types

import numpy as np
import pytest

from laya_assistant import app_actions as aa
from laya_assistant import host
from laya_assistant.cua import apps
from laya_assistant.cua.loop import Loop, Pending, find_element, verify
from laya_assistant.cua.manager import ComputerUse
from laya_assistant.cua.methods import MethodMemory
from laya_assistant.cua.planner import make_planner, parse_plan
from laya_assistant.cua.policy import DomainPolicy, validate
from laya_assistant.cua.recorder import StepRecorder
from laya_assistant.cua.tools import make_tools
from laya_assistant.cua.types import Candidate, Element, Observation, StepPlan


def S(do, target="", value=None):
    return StepPlan(do, target, value)


def obs(*els, content="", app="Messages"):
    return Observation("desktop", "W", None, list(els), app=app, content=content)


class FlatRanker:
    def score(self, query, texts):
        return np.full(len(texts), 0.5)


class TokenRanker:
    """Similarity as the share of the step's words found in the element's text: close enough to the real embeddings for an exact-label match to win."""

    def score(self, query, texts):
        q = set(query.lower().split())
        return np.array([len(q & set(t.lower().replace(":", " ").split())) / max(len(q), 1) for t in texts])


class NoPredictor:
    def predict(self, *a, **k):
        raise AssertionError("no model may be asked on these paths")


class CalmPredictor:
    """Laya's safety questions (irreversible / needs the user / blocked), all answered 'no': what it says about an ordinary click."""

    def predict(self, state, questions):
        return {"answers": {q: {"noul": 0.02} for q in questions}}


# -- B: verification that survives the UI changing ------------------------------------------------------------------------------------------

def test_a_typed_field_is_found_again_after_the_tree_is_re_indexed():
    """The Messages case: typing in Search brought up a results list, every id moved, and e2 was no longer the search field."""
    before = obs(Element("e2", "textbox", "Search"), Element("e17", "textbox", "Message"))
    cand = Candidate("e2", "type", before.elements[0], "Manjit Gahir", 0.9)
    after = obs(Element("e2", "button", "Manjit Gahir"), Element("e9", "textbox", "Search", "Manjit Gahir"), Element("e40", "textbox", "Message"))
    assert find_element(cand, after).id == "e9"
    assert verify(cand, before, after, {"ok": True})[0] is True


def test_a_field_whose_label_became_what_was_typed_is_still_that_field():
    """Measured on the real Messages window: after typing, the search field reads `textbox 'Manjit Gahir'` (its label was 'Search')."""
    before = obs(Element("e2", "textbox", "Search", "Search"), Element("e15", "textbox", "Message", "iMessage"))
    cand = Candidate("e2", "type", before.elements[0], "Manjit Gahir", 0.9)
    after = obs(Element("e2", "textbox", "Manjit Gahir", "Manjit Gahir"), Element("e15", "textbox", "Message", "iMessage"))
    assert find_element(cand, after).id == "e2" and verify(cand, before, after, {"ok": True})[0] is True
    # but another field is not mistaken for it just because it is a textbox
    assert find_element(Candidate("e15", "type", before.elements[1], "hello", 0.9), after).id == "e15"


def test_a_typed_value_is_accepted_from_the_window_text_when_the_field_cannot_be_found_again():
    before = obs(Element("e2", "textbox", "Search"))
    cand = Candidate("e2", "type", before.elements[0], "Manjit Gahir", 0.9)
    assert verify(cand, before, obs(Element("e5", "button", "Go"), content="Search Manjit Gahir Contacts"), {"ok": True})[0] is True
    ok, why = verify(cand, before, obs(Element("e5", "button", "Go"), content="nothing typed here"), {"ok": True})
    assert not ok and "not on screen" in why


def test_a_field_that_holds_the_wrong_text_still_fails():
    before = obs(Element("e2", "textbox", "Search"))
    cand = Candidate("e2", "type", before.elements[0], "Manjit Gahir", 0.9)
    ok, why = verify(cand, before, obs(Element("e2", "textbox", "Search", "Someone Else"), content="Manjit Gahir"), {"ok": True})
    assert not ok and "holds" in why  # the field itself is authoritative when it is found


def test_of_several_controls_with_one_label_the_nearest_to_where_it_was_is_used():
    cand = Candidate("e10", "click", Element("e10", "button", "Close"), None, 0.9)
    after = obs(Element("e3", "button", "Close"), Element("e12", "button", "Close"), Element("e30", "button", "Close"))
    assert find_element(cand, after).id == "e12"


class ShiftingScreen:
    """After the first look every id moves by one (a new row appeared at the top), as a real accessibility tree does."""
    name = "fake"

    def __init__(self):
        self.looks, self.acts = 0, []

    def observe(self):
        self.looks += 1
        shift = 1 if self.looks > 1 else 0
        els = ([Element("e0", "button", "New banner")] if shift else []) + [Element(f"e{shift}", "button", "Send"), Element(f"e{shift + 1}", "textbox", "Message")]
        els = [Element(f"e{i}", e.role, e.label) for i, e in enumerate(els)]
        return Observation("desktop", "W", None, els, app="Editor", content="x" * self.looks)

    def act(self, action):
        self.acts.append(action)
        return {"ok": True}


def test_an_approved_action_is_performed_on_the_control_even_if_its_id_moved_while_the_user_decided(tmp_path):
    screen = ShiftingScreen()
    first = screen.observe()  # what the loop saw when it parked the step
    cand = Candidate("e0", "click", first.elements[0], None, 0.9)
    loop = Loop(screen, NoPredictor(), FlatRanker(), DomainPolicy(tmp_path / "p.json"), StepRecorder())
    r = loop.resume([S("click", "Send"), S("done")], Pending("p1", 0, S("click", "Send"), cand, "irreversible", "click Send"))
    assert screen.acts[0].id == "e1"  # the Send button is e1 now, not e0 (the banner)
    assert r.status == "done"


# -- C: plans stay on the surface -----------------------------------------------------------------------------------------------------------

def test_a_desktop_plan_that_navigates_to_a_website_is_refused_and_replanned():
    seen = []
    replies = iter(['[{"do":"navigate","value":"https://duckduckgo.com/?q=Manjit"}]',
                    '[{"do":"type","target":"Search","value":"Manjit Gahir"},{"do":"more"}]'])

    class LLM:
        def invoke(self, prompt):
            seen.append(prompt)
            return types.SimpleNamespace(content=next(replies))

    steps = make_planner(LLM(), surface="a Mac app", desktop=True)("message Manjit hello", obs(), None, None)
    assert [s.do for s in steps] == ["type", "more"] and len(seen) == 2
    assert "not available on this surface" in seen[1]
    assert "NEVER use \"navigate\"" in seen[0] and "navigate|" not in seen[0] and "|navigate" not in seen[0]  # not even offered as a kind


def test_a_browser_plan_may_still_navigate():
    assert parse_plan('[{"do":"navigate","value":"https://example.com"},{"do":"done"}]')[0].do == "navigate"
    with pytest.raises(ValueError, match="not available"):
        from laya_assistant.cua.types import DESKTOP_KINDS

        parse_plan('[{"do":"navigate","value":"https://example.com"}]', allowed=DESKTOP_KINDS)


def test_the_planner_is_told_to_plan_only_what_it_can_see_and_to_say_more():
    seen = []

    class LLM:
        def invoke(self, prompt):
            seen.append(prompt)
            return types.SimpleNamespace(content='[{"do":"done"}]')

    make_planner(LLM())("g", obs(), None, None)
    assert '"more"' in seen[0] and "CURRENT screen" in seen[0]


# -- D: planning in chunks ------------------------------------------------------------------------------------------------------------------

class ChatScreen:
    """Search box; typing shows a result row; clicking the row opens the conversation (which has a message box)."""
    name = "fake"

    def __init__(self):
        self.query, self.opened, self.typed, self.acts = "", False, "", []

    def observe(self):
        els = [Element("e0", "textbox", "Search", self.query)]
        if self.query and not self.opened:
            els.append(Element("e1", "button", "Manjit Gahir conversation"))
        if self.opened:
            els.append(Element("e2", "textbox", "iMessage", self.typed))
        return Observation("desktop", "Messages", None, els, app="Messages", content=f"{self.query} {self.opened} {self.typed}")

    def act(self, action):
        self.acts.append(action)
        if action.kind == "type" and action.id == "e0":
            self.query = action.value
        elif action.kind == "type":
            self.typed = action.value
        elif action.kind == "click":
            self.opened = True
        return {"ok": True}


class Scripted:
    def __init__(self, *plans):
        self.plans, self.calls = list(plans), []

    def __call__(self, goal, obs, done=None, failure=None, max_steps=8):
        self.calls.append((failure, list(done or []), obs.summary(30)))
        return self.plans.pop(0)


def make_cu(screen, *plans, tmp_path, methods=None):
    planner = Scripted(*plans)
    cu = ComputerUse(CalmPredictor(), TokenRanker(), lambda t: planner, lambda t, app="": screen, DomainPolicy(tmp_path / "p.json"), StepRecorder(), methods=methods)
    return cu, planner


def test_a_goal_that_needs_the_next_screen_is_planned_in_chunks_and_finishes_only_when_the_planner_says_done(tmp_path):
    screen = ChatScreen()
    cu, planner = make_cu(screen,
                          [S("type", "Search", "Manjit Gahir"), S("more")],
                          [S("click", "Manjit Gahir conversation"), S("type", "iMessage", "hello"), S("done")],
                          tmp_path=tmp_path)
    out = cu.run("desktop", "message Manjit Gahir hello", app="Messages")
    assert out.startswith("DONE") and "3 steps" in out, out
    assert screen.typed == "hello" and screen.opened
    assert len(planner.calls) == 2
    assert planner.calls[1][1] == ["type Search = 'Manjit Gahir'"] and "Manjit Gahir conversation" in planner.calls[1][2]  # the second plan SAW the results


def test_chunking_stops_when_a_chunk_changes_nothing_on_screen(tmp_path):
    screen = ChatScreen()
    cu, planner = make_cu(screen, [S("wait"), S("more")], [S("wait"), S("more")], tmp_path=tmp_path)
    out = cu.run("desktop", "g", app="Messages")
    assert out.startswith("STALLED") and "did not change" in out and len(planner.calls) == 1


def test_chunking_is_bounded_even_when_every_chunk_changes_something(tmp_path):
    screen = ChatScreen()
    plans = [[S("type", "Search", f"v{i}"), S("more")] for i in range(10)]
    cu, planner = make_cu(screen, *plans, tmp_path=tmp_path)
    out = cu.run("desktop", "g", app="Messages")
    assert out.startswith("STALLED") and "step budget" in out and len(planner.calls) == 6


# -- A: the router --------------------------------------------------------------------------------------------------------------------------

class Cat:
    """Installed-app facts: Notes and Messages are scriptable, Visual Studio Code has a `code` command, Slack has neither."""
    def resolve(self, s):
        s = s.lower()
        for name in ("Messages", "Notes", "Visual Studio Code", "Slack"):
            if s == name.lower() or s in name.lower().split():
                return types.SimpleNamespace(name=name)
        return None

    def scripting(self, name):
        return name in ("Notes", "Messages")

    def dictionary(self, name):
        return ""

    def cli(self, name):
        return "code" if name == "Visual Studio Code" else None


def test_the_ladder_script_then_command_line_then_window():
    assert "AppleScript via mac_run" in aa.method_note("Notes", Cat())
    cli = aa.method_note("Visual Studio Code", Cat())
    assert "command line (code)" in cli and "mac_run" in cli and "computer_use(target='desktop', app='Visual Studio Code')" in cli
    assert "no scripting dictionary, so AppleScript cannot drive it" in aa.method_note("Slack", Cat())


def test_an_app_where_scripts_fail_is_worked_through_its_window_first(tmp_path):
    m = MethodMemory(tmp_path / "m.json")
    assert m.prefers_ui("Messages")  # the measured starting point
    assert "Do not write AppleScript" in aa.method_note("Messages", Cat(), m)
    assert not m.prefers_ui("Notes") and "AppleScript via mac_run" in aa.method_note("Notes", Cat(), m)
    m.record("Notes", "script", False)
    assert not m.prefers_ui("Notes")  # one failure is not a pattern
    m.record("Notes", "script", False)
    assert m.prefers_ui("Notes")
    m.record("Messages", "script", True)
    assert not m.prefers_ui("Messages")  # a script that worked overrides the starting point
    assert MethodMemory(tmp_path / "m.json").prefers_ui("Notes")  # and it is remembered across sessions


def test_a_window_that_keeps_failing_does_not_push_an_app_back_to_scripts_forever(tmp_path):
    m = MethodMemory()
    for _ in range(3):
        m.record("Messages", "ui", False)
    assert not m.prefers_ui("Messages")


def test_ui_outcomes_are_recorded_per_app(tmp_path):
    m = MethodMemory()
    screen = ChatScreen()
    cu, _ = make_cu(screen, [S("type", "Search", "x"), S("done")], tmp_path=tmp_path, methods=m)
    cu.run("desktop", "g", app="Messages")
    assert m.stats["Messages"]["ui"] == {"ok": 1, "fail": 0}


def test_mac_run_refuses_a_third_script_for_an_app_that_failed_twice_and_names_the_way_out():
    ran = []
    runner = lambda cmd, **kw: ran.append(cmd) or types.SimpleNamespace(returncode=1, stdout="", stderr="execution error: (-1728)")
    m = MethodMemory()
    mac_run = {t.name: t for t in host.make_host_tools(runner=runner, methods=m)}["mac_run"]
    cmd = "osascript -e 'tell application \"Messages\" to get participants'"
    mac_run.invoke({"command": cmd})
    mac_run.invoke({"command": cmd})
    third = mac_run.invoke({"command": cmd})
    assert third.startswith("NOT RUN") and "computer_use(target='desktop', app='Messages')" in third and len(ran) == 2
    assert m.stats["Messages"]["script"]["fail"] == 2
    assert not mac_run.invoke({"command": "osascript -e 'tell application \"Notes\" to get name'"}).startswith("NOT RUN")  # another app is unaffected


@pytest.mark.parametrize("cmd,safe", [("code ~/Developer/cua-self-tests", True), ("code -r /Users/me/x/hello.py", True), ("code ~/x; rm -rf ~", False),
                                      ("code $(whoami)", False), ("cursor ~/proj", True)])
def test_opening_something_in_an_editor_is_safe_anything_chained_is_not(cmd, safe):
    assert host.is_safe_host_command(cmd) is safe


def test_a_command_line_is_found_on_path_or_inside_the_app_bundle(tmp_path, monkeypatch):
    app = tmp_path / "Visual Studio Code.app"
    (app / "Contents/Resources/app/bin").mkdir(parents=True)
    exe = app / "Contents/Resources/app/bin/code"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    cat = apps.AppCatalog(dirs=(str(tmp_path),))
    monkeypatch.setattr(apps, "SHELL_PATH", str(tmp_path / "nowhere"))
    assert cat.cli("Visual Studio Code") == "'" + str(exe) + "'"  # not on PATH: the bundle's own copy, quoted for the shell
    assert cat.cli("Slack") is None


# -- app inference and the send rule ------------------------------------------------------------------------------------------------------

def test_a_desktop_goal_without_an_app_takes_it_from_the_goal(tmp_path, monkeypatch):
    monkeypatch.setattr(apps, "AppCatalog", Cat)
    got = {}
    screen = ChatScreen()
    planner = Scripted([S("done")])
    cu = ComputerUse(NoPredictor(), FlatRanker(), lambda t: planner, lambda t, app="": got.setdefault("app", app) and screen or screen,
                     DomainPolicy(tmp_path / "p.json"), StepRecorder())
    cu.run("desktop", "open my messages app and message Manjit Gahir hello")
    assert got["app"] == "Messages"


def test_a_desktop_goal_that_names_no_app_is_refused_instead_of_looking_at_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(apps, "AppCatalog", Cat)
    cu, _ = make_cu(ChatScreen(), [S("done")], tmp_path=tmp_path)
    assert make_tools(cu)[0].invoke({"target": "desktop", "goal": "do the thing"}).startswith("UNAVAILABLE")


def test_finding_the_app_ignores_filler_words():
    assert aa.find_app("can you open my messages app and message Manjit Gahir hello", Cat()).name == "Messages"


def test_pressing_enter_in_a_messaging_app_asks_first_but_typing_does_not(tmp_path):
    box = Element("e2", "textbox", "iMessage")
    o = obs(box, app="Messages")
    dom = DomainPolicy(tmp_path / "p.json")
    assert validate("press_enter", Candidate("e2", "press_enter", box, None, 0.9), o, dom).action == "ask"
    assert validate("type", Candidate("e2", "type", box, "hello", 0.9), o, dom).action == "allow"
    assert validate("press_enter", Candidate("e2", "press_enter", box, None, 0.9), obs(box, app="Notes"), dom).action == "allow"


def test_wiring_keeps_the_decision_cache_in_memory_unless_the_app_gives_it_somewhere_to_persist(tmp_path):
    """A test run once wrote a fixture-site entry into the real ~/.laya_assistant/decision_cache.json."""
    from laya_assistant.cua.wiring import make_computer_use

    assert make_computer_use(NoPredictor(), FlatRanker(), bridge=None, worker=object()).cache.path is None
    assert make_computer_use(NoPredictor(), FlatRanker(), bridge=None, worker=object(), recorder_path=tmp_path / "cua_steps.jsonl").cache.path == tmp_path / "decision_cache.json"
