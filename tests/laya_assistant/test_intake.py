"""Intake against the REAL Laya checkpoint and the labelled probe set (no fake model)."""
import pytest

from laya_assistant import config, intake
from laya_assistant.decisions import DecisionLog

from .intake_probe import PROBE


@pytest.fixture(scope="module")
def results(laya_model):
    return [(t, intent, fast_ok, risky, intake.classify(laya_model, t)) for t, intent, fast_ok, risky in PROBE]


def test_no_hard_or_risky_request_is_routed_to_the_fast_path(results):
    wrong = [t for t, _, fast_ok, risky, i in results if i.route == "fast" and (not fast_ok or risky)]
    assert wrong == []


def test_most_simple_turns_take_the_fast_path(results):
    fast_ok = [(t, i) for t, _, ok, _, i in results if ok]
    routed = sum(i.route == "fast" for _, i in fast_ok)
    assert routed / len(fast_ok) >= 0.6, [(t, i.route) for t, i in fast_ok if i.route != "fast"]


def test_destructive_phrasing_is_flagged_risky(results):
    by_text = {t: i for t, _, _, _, i in results}
    for t in ["wipe the workspace and start over from scratch", "run rm -rf on the build directory",
              "cancel my subscription and delete my account"]:
        assert by_text[t].risky >= 0.6, t


def test_confident_intents_are_mostly_right(results):
    confident = [(intent, i) for _, intent, _, _, i in results if i.band == "act"]
    assert sum(i.intent == intent for intent, i in confident) / len(confident) >= 0.75


def test_classify_is_fast_when_warm(laya_model):
    intake.classify(laya_model, "warm")
    ms = [intake.classify(laya_model, "write a python function to reverse a string").ms for _ in range(5)]
    print("intake ms:", [round(m) for m in ms])
    assert sorted(ms)[2] < 150


def test_classify_logs_a_laya_row(laya_model):
    log = DecisionLog()
    i = intake.classify(laya_model, "hi", log=log)
    assert log.rows[0]["system"] == "laya" and log.rows[0]["step"] == "intake"
    assert i.route in log.rows[0]["result"]


def test_defer_band_always_goes_to_the_executor():
    assert intake.decide_route("defer", simple=0.99, factual=0.99, risky=0.0, needs_plan=0.0) == "executor"
    assert intake.decide_route("act", simple=0.9, factual=0.0, risky=0.0, needs_plan=0.0) == "fast"
    assert intake.decide_route("act", simple=0.9, factual=0.0, risky=0.9, needs_plan=0.0) == "executor"
    assert intake.decide_route("act", simple=0.9, factual=0.0, risky=0.0, needs_plan=0.9) == "executor"


def test_note_is_appended_text_and_defer_note_asks_to_clarify():
    fake = intake.Intake("other", 0.1, 0.1, 0.1, 0.1, 0.30, "defer", "executor", 5.0)
    note = intake.intake_note(fake)
    assert note.startswith("[Laya note:") and "clarifying question" in note
    act = intake.Intake("write_code", 0.9, 0.0, 0.0, 0.0, 0.95, "act", "executor", 5.0)
    assert "write_todos" in intake.intake_note(act)


# --- the work-marker veto -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Plan first, then create a file greeting.md in /workspace/shared containing one friendly line, and read it back to me.",
    "create a file called notes.txt with hello in it",
    "open Notes and write a shopping list: milk, eggs, bread",
    "save this to ~/Documents/list.txt",
    "go to https://example.org and read it",
    "search the web for the best headphones",
    "delete the old folder",
    "run this script for me",
    "look up the weather in Paris",
])
def test_requests_that_ask_for_work_are_never_answered_by_the_toolless_fast_model(text):
    assert intake.looks_like_work(text)


@pytest.mark.parametrize("text", [
    "hi there", "hello!", "thanks a lot", "good morning", "what is the capital of France?", "who wrote Hamlet?", "how are you today?",
    "tell me a joke", "what is 12 times 12?", "what does ubiquitous mean?", "explain photosynthesis briefly", "send my regards to the team",
    "I made a greeting card yesterday",
])
def test_small_talk_and_trivia_are_not_mistaken_for_work(text):
    assert not intake.looks_like_work(text)


def test_the_real_miss_now_goes_to_the_agent_and_says_why_in_the_log(laya_model):
    """Measured before the veto: simple=0.63 ('greeting'), intent chat, route fast: the 3B model answered without any tools."""
    text = "Plan first, then create a file greeting.md in /workspace/shared containing one friendly line, and read it back to me."
    log = DecisionLog()
    i = intake.classify(laya_model, text, log=log)
    assert i.route == "executor" and "work marker" in log.rows[0]["result"]
    assert intake.classify(laya_model, "hi there").route == "fast"  # and a greeting still takes the fast path
