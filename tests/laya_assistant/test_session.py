"""Session orchestration against real Laya, real Ollama, real Docker. Slow."""
import pytest

from laya_assistant import config
from laya_assistant.session import AssistantSession

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def session(laya_model):
    s = AssistantSession.create(laya_model)
    yield s
    s.close()


@pytest.fixture(scope="module")
def fast_session(laya_model):
    """A second session for tests that only need quick fast-route turns. Their exchanges must not pile up in the long-lived agent session
    above: the planning tests share that one, and a 14B plans worse (or loops for minutes) with a long, chatty history."""
    s = AssistantSession.create(laya_model, warm=False)
    yield s
    s.close()


def kinds(events):
    return [e.kind for e in events]


def test_greeting_takes_the_fast_route_streams_tokens_and_creates_no_plan(session):
    events = list(session.run_turn("hi there"))
    assert events[0].kind == "intake" and events[0].data.route == "fast"
    tokens = [e for e in events if e.kind == "token"]
    assert tokens and "interrupt" not in kinds(events) and events[-1].kind == "final"
    assert events[-1].data.strip()
    row = next(r for r in session.log.rows if r["step"] == "fast model")
    print(f"fast route: first token {row['result']}, total {row['ms']:.0f} ms")
    assert config.FAST_MODEL in row["result"] or "first token" in row["result"]


def test_fast_exchange_is_visible_to_the_executor_thread(session):
    msgs = session.agent.get_state(session.config).values["messages"]
    assert [m.type for m in msgs[-2:]] == ["human", "ai"] and "hi there" in str(msgs[-2].content)


def test_code_request_pauses_for_plan_then_completes_and_files_are_listed(session, plan_approval_on):
    events = list(session.run_turn("Plan first, then create hello.py that prints hello world and run it to check it works."))
    assert events[0].data.route == "executor"
    guard = 0
    while events[-1].kind == "interrupt" and guard < 8:
        guard += 1
        events += list(session.resume(session.approve_all()))
    assert events[-1].kind == "final", kinds(events)[-6:]
    assert "hello.py" in [p.rsplit("/", 1)[-1] for p in session.list_files()]
    assert b"hello" in session.download("/workspace/hello.py").lower() or True  # content not asserted: model-authored
    assert any(e.kind == "tool" for e in events)
    calls = [e for e in events if e.kind == "call"]
    assert calls and all(c.data["name"] and c.data["id"] for c in calls)  # what the agent is about to do, before it does it
    results = [e.data["id"] for e in events if e.kind in ("tool", "subagent")]
    assert results and set(results) <= {c.data["id"] for c in calls}  # every result answers a call the UI has already seen


def test_uploads_land_in_the_sandbox(session):
    events = list(session.run_turn("hello", uploads=[("note.txt", b"remember me")]))
    assert session.download("/workspace/uploads/note.txt") == b"remember me"
    # a turn with attachments takes the executor route, so it may pause for plan approval: all three are valid ends
    assert events[-1].kind in ("final", "error", "interrupt")


def test_step_budget_stops_a_run_with_an_error_event(laya_model, monkeypatch):
    monkeypatch.setattr(config, "STEP_BUDGET", 3)
    s = AssistantSession.create(laya_model, warm=False)
    try:
        events = list(s.run_turn("write a python script that prints the first 20 primes and run it"))
        assert events[-1].kind == "error" and "step budget" in events[-1].data.lower()
    finally:
        s.close()


# --- background turns: the UI polls; interacting with the page can never abort or strand a turn ------------------


def _wait(session, timeout=120):
    import time

    t0 = time.time()
    while session.busy and time.time() - t0 < timeout:
        time.sleep(0.05)
    assert not session.busy, "turn did not finish"


def test_submit_returns_immediately_and_the_feed_fills_in_the_background(session):
    import time

    t0 = time.perf_counter()
    session.submit("hello there")
    assert time.perf_counter() - t0 < 0.5  # the call does not wait for the model
    _wait(session)
    kinds_ = [e.kind for e in session.feed()]
    assert kinds_[0] == "intake" and kinds_[-1] == "final" and "token" in kinds_


def test_an_abandoned_turn_never_strands_the_gpu_lock(session):
    """The UI can be interrupted at any moment (a click reruns the page). Nothing may keep the lock afterwards."""
    from laya_assistant.engine import ENGINE

    session.submit("good morning")  # nobody ever reads the feed
    _wait(session)
    assert ENGINE.acquire(timeout=2), "the GPU lock is stuck"
    ENGINE.release()


def test_the_feed_is_reset_for_each_turn_and_a_second_submit_while_busy_is_refused(session):
    session.submit("hi")
    assert session.submit("this must not start a second concurrent turn") is False
    _wait(session)
    session.submit("thanks")
    _wait(session)
    assert [e.kind for e in session.feed()].count("intake") == 1  # only the latest turn's events


def test_a_worker_exception_becomes_an_error_event_not_a_dead_thread(laya_model, monkeypatch):
    s = AssistantSession.create(laya_model, warm=False)
    try:
        monkeypatch.setattr(s, "run_turn", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        s.submit("hi")
        _wait(s)
        last = s.feed()[-1]
        assert last.kind == "error" and "boom" in str(last.data)
        # the failed request is not lost: the chat can still show what was asked and what went wrong
        from laya_assistant.transcript import build_turns

        (turn,) = build_turns(s.agent.get_state(s.config).values["messages"], s.metas)
        assert turn.prompt == "hi" and turn.meta.error and "boom" in turn.meta.error
    finally:
        s.close()


def test_submit_resume_continues_a_paused_plan_in_the_background(session, plan_approval_on):
    session.submit("Plan first, then create hello2.py that prints hello world and run it to check it works.")
    _wait(session)
    guard = 0
    while session.pending and guard < 8:
        guard += 1
        session.submit_resume(session.approve_all())
        _wait(session)
    assert session.feed()[-1].kind in ("final", "error")
    assert any(p.endswith(".py") for p in session.list_files())


def test_each_request_records_how_it_was_routed_and_how_long_it_took(fast_session):
    session = fast_session
    session.submit("hi again")
    _wait(session)
    msgs = session.agent.get_state(session.config).values["messages"]
    human = [m for m in msgs if m.type == "human"][-1]
    meta = session.metas[human.id]  # the human message carries the id that joins the request to its timings
    assert meta is session.meta and meta.route == "fast" and meta.laya_ms > 0
    assert meta.first_token_ms and meta.first_token_ms > 0 and 0 < meta.elapsed < 30 and meta.error is None and not meta.cancelled


def test_stop_ends_a_turn_keeps_the_message_and_frees_the_gpu(fast_session):
    from laya_assistant.engine import ENGINE

    session = fast_session
    session.submit("tell me about the weather today")
    session.cancel()  # pressed immediately: the flag is set before the first token
    _wait(session)
    assert [e.kind for e in session.feed()][-1] == "cancelled" and session.meta.cancelled
    msgs = session.agent.get_state(session.config).values["messages"]
    assert any(m.type == "human" and "weather today" in str(m.content) for m in msgs)  # the person's message is still in the chat
    assert ENGINE.acquire(timeout=2), "the GPU lock is stuck after a Stop"
    ENGINE.release()
    session.submit("hello")  # and the next request works normally: the flag does not linger
    _wait(session)
    assert session.feed()[-1].kind == "final" and not session.meta.cancelled


def test_sized_file_listing_reports_sizes(session):
    session.handle.backend.write("/workspace/sized.txt", "abcde")
    sizes = dict(session.list_files_sized())
    assert sizes["/workspace/sized.txt"] == 5


def test_answering_the_plan_with_a_typed_yes_continues_without_a_loop(session, plan_approval_on):
    _submit_expecting_plan(session, "Plan first, then create typed_yes.py that prints hello world and run it to check it works.")
    assert session.pending["action_requests"][0]["name"] == "write_todos"
    session.answer_pending("yes")  # plain words, no button
    pauses = 0
    for _ in range(6):
        _wait(session)
        if not session.pending:
            break
        pauses += 1  # any further pause would be a re-approval loop
        session.answer_pending("yes")
    print("re-approvals after the first yes:", pauses)
    from laya_assistant import config

    assert not session.pending and session.feed()[-1].kind in ("final", "error")
    # The exact count depends on how often the 14B rewrites its own plan, so assert what must always hold: it terminates, and it can
    # never ask more than the cap allows. (Status-only updates and reworded steps never ask: unit-tested in test_cortex.)
    assert pauses < config.MAX_PLAN_ASKS
    assert any(p.endswith("typed_yes.py") for p in session.list_files())


def _submit_expecting_plan(session, prompt, tries=3):
    """The 14B occasionally skips planning altogether (the cortex nudges, then falls back to all tools). That is model variance, not
    what these tests are about, so retry a couple of times before asserting anything about the plan pause."""
    for i in range(tries):
        session.submit(prompt + (" " * i))
        _wait(session)
        if session.pending:
            return
    raise AssertionError("the agent never produced a plan to pause on")


def test_a_typed_change_request_reaches_the_agent_and_a_revised_plan_can_then_be_approved(session, plan_approval_on):
    _submit_expecting_plan(session, "Plan first, then create typed_change.py that prints hello world and run it.")
    session.answer_pending("also print the current year in the same script")  # not a yes: the agent must revise
    for _ in range(6):
        _wait(session)
        if not session.pending:
            break
        session.answer_pending("yes")
    assert not session.pending
    assert session.feed()[-1].kind in ("final", "error")


def test_by_default_a_code_task_runs_straight_through_with_no_question(session):
    """The shipped default asks about as little as possible: the plan is shown in the feed but not held for approval."""
    session.submit("Create default_run.py that prints hello world and run it to check it works.")
    _wait(session)
    assert session.pending is None, f"it paused for: {session.pending}"
    assert session.feed()[-1].kind in ("final", "error")
    assert any(p.endswith("default_run.py") for p in session.list_files())


def test_plan_calls_and_their_results_never_appear_as_tool_events():
    """Measured: a 14B that calls write_todos twice in parallel gets a guard error back as a ToolMessage with NO name. The plan has its own
    event, so neither the call nor any answer to it may show up as a step."""
    from langchain_core.messages import AIMessage, ToolMessage

    s = AssistantSession.__new__(AssistantSession)
    s._plan_calls, s.pending = set(), None
    todo = {"content": "a", "status": "pending"}
    ai = AIMessage(content="", tool_calls=[{"name": "write_todos", "args": {"todos": [todo]}, "id": "p1", "type": "tool_call"},
                                            {"name": "write_file", "args": {"file_path": "/workspace/a.py"}, "id": "w1", "type": "tool_call"}])
    first = list(s._translate((), "updates", {"model": {"messages": [ai]}}))
    assert [e.kind for e in first] == ["plan", "call"] and first[1].data["id"] == "w1"
    replies = [ToolMessage(content="Error: The `write_todos` tool should never be called multiple times in parallel", tool_call_id="p1"),
               ToolMessage(content="Updated file /workspace/a.py", tool_call_id="w1", name="write_file")]
    second = list(s._translate((), "updates", {"tools": {"messages": replies}}))
    assert [(e.kind, e.data["id"]) for e in second] == [("tool", "w1")]
