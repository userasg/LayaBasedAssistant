"""Streamlit AppTest against the real stack (slow: loads Laya, warms Ollama, starts Docker).

Turns run on a worker thread and the page polls, so tests re-run the script until the turn has finished."""
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.slow


def wait_done(app, timeout=180):
    s = app.session_state["session"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.run()
        if not s.busy and not s.turn_open:
            return
        time.sleep(0.3)
    raise AssertionError("turn did not finish")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(Path(__file__).parents[2] / "src" / "laya_assistant" / "app.py"), default_timeout=240).run()
    yield at
    if "session" in at.session_state:
        at.session_state["session"].close()


def test_app_renders_without_exceptions_and_starts_in_review_mode(app):
    assert not app.exception, [e.value for e in app.exception]
    assert app.title[0].value == "Laya Assistant"
    assert app.session_state["session"].handle.label.startswith("Docker sandbox")
    assert app.toggle(key="auto_send").value is False  # voice defaults to review-before-send


def test_the_first_screen_offers_examples_and_clicking_one_runs_it_as_one_answer_with_the_work_folded(app):
    from laya_assistant.ui.chat import EXAMPLES

    chips = [b for b in app.button if str(b.key).startswith("example_")]
    assert len(chips) == len(EXAMPLES) == 4 and not app.chat_message  # nothing said yet: just the examples
    files_prompt = EXAMPLES[2][1]
    next(b for b in chips if b.key == "example_2").click().run()
    session = app.session_state["session"]
    assert session.running_prompt == files_prompt  # exactly the words on the button were sent
    wait_done(app)
    assert not app.exception, [e.value for e in app.exception]
    assert not [b for b in app.button if str(b.key).startswith("example_")]  # the examples go away once there is a conversation
    assistant = [m for m in app.chat_message if m.name == "assistant"]
    assert len(assistant) == 1 and len(assistant[0].markdown) >= 1  # one bubble for the whole request, however many steps it took
    assert session.meta.route == "executor" and session.meta.elapsed > 0
    from laya_assistant.transcript import build_turns

    (turn,) = build_turns(session.agent.get_state(session.config).values["messages"], session.metas)
    folded = [e for e in app.expander if str(e.label).startswith("Show")]
    assert len(folded) == (1 if (turn.steps or turn.plan) else 0)  # the tool calls and the plan are folded under the answer, not drawn as bubbles


def test_typed_greeting_is_answered_by_the_fast_route_without_blocking_the_page(app):
    t0 = time.time()
    app.chat_input[0].set_value("hi there").run()
    assert time.time() - t0 < 5  # the submit returned at once; the turn runs in the background
    wait_done(app)
    assert not app.exception, [e.value for e in app.exception]
    msgs = [(m.name, str(m.markdown[0].value)) for m in app.chat_message]
    assert msgs[-2][0] == "user" and "hi there" in msgs[-2][1]
    assert msgs[-1][0] == "assistant" and len(msgs[-1][1]) > 2


def test_dictated_text_is_reviewed_edited_and_sent(app):
    app.session_state["dictation"] = "what is the capital of France"
    app.run()
    box = next(t for t in app.text_area if t.key == "dictation")
    box.set_value("what is the capital of Italy").run()  # the user fixes a mis-heard word
    app.button[[b.label for b in app.button].index("Send")].click().run()
    # the script cleared the box the moment it sent (AppTest re-applies values assigned through at.session_state on later
    # runs, so the clearing is asserted here, right after the click, not after the polling loop)
    assert app.session_state["dictation"] == ""
    session = app.session_state["session"]
    while session.busy:
        time.sleep(0.1)
    sent = [str(m.content) for m in session.agent.get_state(session.config).values["messages"] if m.type == "human"]
    assert any("capital of Italy" in t for t in sent)
    assert not any("France" in t for t in sent)  # the edited text was sent, not the mis-heard one
    session.turn_open = False  # the harness keeps re-applying the dictation value; stop the polling loop from re-running it


def test_a_long_conversation_keeps_the_page_small(app):
    from laya_assistant.ui.chat import VISIBLE_TURNS

    session = app.session_state["session"]
    for i in range(14):
        session.submit(f"hello number {i}")
        while session.busy:
            time.sleep(0.05)
    wait_done(app)
    assert not app.exception, [e.value for e in app.exception]
    total = len([m for m in session.agent.get_state(session.config).values["messages"] if m.type == "human"])
    fold = next(e for e in app.expander if "earlier turns" in str(e.label))
    assert total > VISIBLE_TURNS and int(str(fold.label).split()[0]) == total - VISIBLE_TURNS  # everything older sits in one fold
    assert len([m for m in app.chat_message if m.name == "user"]) == VISIBLE_TURNS + len(
        [m for m in fold.chat_message if m.name == "user"])  # the fold holds the rest; only the latest turns are drawn outside it


def test_the_text_box_stays_usable_while_an_approval_is_pending_and_routes_the_answer(app):
    session = app.session_state["session"]
    while session.busy:
        time.sleep(0.1)
    session.turn_open = False
    answers = []
    session.pending = {"action_requests": [{"name": "write_todos", "args": {"todos": [{"content": "a", "status": "pending"}]}}],
                       "review_configs": []}
    session.answer_pending = lambda text: answers.append(text) or True  # the routing is under test here, not the graph
    app.run()
    box = app.chat_input[0]
    assert not box.disabled and "yes / no" in (box.placeholder or "")
    box.set_value("yes, but skip step 2").run()
    assert answers == ["yes, but skip step 2"]
    session.pending = None


def test_the_autonomy_panel_defaults_to_asking_as_little_as_possible_and_switches_apply_live(app):
    from laya_assistant import approvals

    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert approvals.current.plan is False and approvals.current.host_commands == "risky" and approvals.current.new_sites is False
    plans = lambda: next(t for t in app.toggle if "plans" in str(t.label))  # fresh element each time: a rerun replaces them
    plans().set_value(True).run()
    assert approvals.current.plan is True
    plans().set_value(False).run()
    assert approvals.current.plan is False


@pytest.mark.parametrize("name,args,extra,shown", [
    ("mac_run", {"command": "mv ~/a.txt ~/b.txt"}, {"laya_risk": 0.74}, "mv ~/a.txt ~/b.txt"),
    ("delete", {"file": "/workspace/shared/report.csv", "path": "report.csv"}, {}, "report.csv"),
])
def test_the_approval_card_draws_a_command_or_a_delete_and_returns_the_decision(app, name, args, extra, shown):
    from laya_assistant import review

    session = app.session_state["session"]
    while session.busy:
        time.sleep(0.1)
    session.turn_open = False
    sent = []
    session.submit_resume = lambda decisions: sent.append(decisions) or True  # the card's decision is under test, not the graph
    session.pending = review.request(name, f"Do {name}", args, **extra)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    card_text = " ".join(str(x.value) for x in [*app.markdown, *app.code, *app.caption])
    assert shown in card_text
    next(b for b in app.button if b.label == "Approve").click().run()
    assert sent == [[{"type": "approve"}]]
    session.pending = None
    del session.submit_resume  # back to the real method for the tests that follow


def test_rejecting_on_the_card_rejects_every_request(app):
    from laya_assistant import review

    session = app.session_state["session"]
    session.turn_open = False
    sent = []
    session.submit_resume = lambda decisions: sent.append(decisions) or True
    session.pending = review.request("mac_run", "Run", {"command": "rm ~/x"})
    app.run()
    next(b for b in app.button if b.label == "Reject").click().run()
    assert sent == [[{"type": "reject"}]]
    session.pending = None
    del session.submit_resume


def test_a_spoken_answer_to_a_pending_question_goes_through_the_review_card(app):
    from laya_assistant import review

    session = app.session_state["session"]
    while session.busy:
        time.sleep(0.1)
    session.turn_open = False
    answers = []
    session.answer_pending = lambda text: answers.append(text) or True
    session.pending = review.request("mac_run", "Run", {"command": "mv ~/a ~/b"})
    app.session_state["dictation"] = "yes"
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert answers == []  # parked in the card for review, not sent
    assert any("answer the question" in str(c.value) for c in app.caption)
    next(b for b in app.button if b.label == "Send").click().run()
    assert answers == ["yes"]
    session.pending = None
    del session.answer_pending
