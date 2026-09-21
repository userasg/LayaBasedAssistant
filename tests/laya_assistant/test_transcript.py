"""How a conversation is folded for the chat window. Real LangChain messages and real session Events; no model is involved."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from laya_assistant import transcript as tr
from laya_assistant.session import Event


def call(name, args, id):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def agent_turn():
    """A realistic agent turn: plan, narration next to a tool call, two tools (one fails), then the answer."""
    return [
        HumanMessage(content="find the best headphones and note them", id="h1"),
        AIMessage(content="", tool_calls=[call("write_todos", {"todos": [{"content": "search", "status": "completed"}, {"content": "note", "status": "in_progress"}]}, "t0")]),
        ToolMessage(content="Updated todo list", tool_call_id="t0", name="write_todos"),
        AIMessage(content="I'll search the web first.", tool_calls=[call("internet_search", {"query": "best headphones 2026"}, "t1")]),
        ToolMessage(content="Sony WH-1000XM6 (https://x.example): great ANC\nBose QC Ultra", tool_call_id="t1", name="internet_search"),
        AIMessage(content="", tool_calls=[call("mac_notes_create", {"title": "Headphones", "body": "Sony"}, "t2")]),
        ToolMessage(content="FAILED: Notes got an error", tool_call_id="t2", name="mac_notes_create"),
        AIMessage(content="The Sony WH-1000XM6 came out on top; the note could not be saved."),
    ]


def test_an_agent_turn_becomes_one_answer_with_the_work_folded_away():
    (turn,) = tr.build_turns(agent_turn())
    assert turn.prompt == "find the best headphones and note them"
    assert turn.answer == "The Sony WH-1000XM6 came out on top; the note could not be saved."  # only the last plain message
    assert [s.kind for s in turn.steps] == ["say", "tool", "tool"]  # the narration stays with the work, not in the answer
    assert turn.steps[0].done.startswith("I'll search")
    search, note = turn.steps[1], turn.steps[2]
    assert (search.done, search.detail) == ("Searched the web", "best headphones 2026")
    assert "Sony WH-1000XM6" in search.result and not search.failed
    assert note.done == "Made a note" and note.detail == "Headphones" and note.failed  # a failure is flagged, not hidden
    assert [t["content"] for t in turn.plan] == ["search", "note"] and turn.plan[1]["status"] == "in_progress"  # the plan is kept, but is not a step


def test_a_conversation_is_split_at_each_human_message_and_timings_are_joined_by_id():
    msgs = agent_turn() + [HumanMessage(content="thanks", id="h2"), AIMessage(content="Any time!")]
    meta = tr.TurnMeta(route="fast", laya_ms=54, first_token_ms=43, elapsed=0.4)
    turns = tr.build_turns(msgs, {"h2": meta})
    assert [t.prompt for t in turns] == ["find the best headphones and note them", "thanks"]
    assert turns[0].meta is None and turns[1].meta is meta and turns[1].answer == "Any time!" and turns[1].steps == []


def test_a_turn_paused_for_approval_has_no_answer_yet():
    msgs = agent_turn()[:5] + [AIMessage(content="", tool_calls=[call("computer_use", {"target": "browser", "goal": "buy it"}, "t9")])]
    (turn,) = tr.build_turns(msgs)
    assert turn.answer == "" and turn.steps[-1].doing == "Working in the browser"


def test_markers_the_app_adds_are_not_shown_as_what_the_person_said():
    text = "summarise these\n\n[Attached files: /workspace/uploads/a.txt, /workspace/uploads/b b.png]\n\n[Already done while the user was speaking: open Notes; scroll down. Do not repeat it.]"
    assert tr.clean_prompt(text) == ("summarise these", ["a.txt", "b b.png"], ["open Notes", "scroll down"])
    link = "see [this page](http://x.example) please\n\nand [that one](http://y.example)"
    assert tr.clean_prompt(link) == (link, [], [])  # a person's own brackets are never cut


def test_tools_are_described_in_words_and_unknown_ones_still_show_their_arguments():
    assert tr.describe("execute", {"command": "python hello.py"}).doing == "Running a command in the sandbox"
    assert tr.describe("task", {"subagent_type": "researcher", "description": "look up prices"}).done == "Asked the researcher"
    assert tr.describe("computer_use", {"target": "my_chrome", "goal": "open gmail"}).doing == "Working in your Chrome"
    odd = tr.describe("frobnicate", {"x": 1})
    assert odd.doing == "Using frobnicate" and "'x': 1" in odd.detail
    assert len(tr.describe("execute", {"command": "x" * 500}).detail) <= 110


def test_failures_are_recognised_from_the_result_text():
    assert tr.looks_failed("FAILED: no such file") and tr.looks_failed("UNAVAILABLE: chrome") and tr.looks_failed("ok", status="error")
    assert tr.looks_failed("exit 2\nboom") and tr.looks_failed("[Command failed with exit code 1]")
    assert not tr.looks_failed("exit 0\nfine") and not tr.looks_failed("hello world\n[Command succeeded with exit code 0]")


def test_the_meta_line_says_how_a_request_was_handled():
    fast = tr.Turn("a", "hi", meta=tr.TurnMeta(route="fast", laya_ms=54.2, first_token_ms=43, elapsed=0.41))
    assert tr.meta_line(fast) == "⚡ fast · Laya 54 ms · first word 43 ms · 0.4 s"
    (agent,) = tr.build_turns(agent_turn())
    agent.meta = tr.TurnMeta(route="executor", laya_ms=61, elapsed=38.2, cancelled=True)
    assert tr.meta_line(agent) == "🧠 agent · Laya 61 ms · 2 steps · 38 s · stopped by you"
    assert tr.meta_line(tr.Turn("c", "x", meta=tr.TurnMeta(error="Stopped: step budget reached"))).startswith("⚠️ Stopped")
    assert tr.meta_line(tr.Turn("d", "x")) == ""
    assert [tr.fmt_seconds(s) for s in (0.04, 0.4, 9.96, 38.2, 125)] == ["40 ms", "0.4 s", "10.0 s", "38 s", "2 min 5 s"]


# --- the live feed ----------------------------------------------------------------------------------------------------------

def ev(kind, data=None, t=0.0):
    return Event(kind, data, t)


def test_the_live_view_shows_what_is_running_now_and_only_the_text_written_since_the_last_tool():
    live = tr.summarize_feed([
        ev("intake", "INTAKE", 1.0),
        ev("token", "I'll search. ", 1.2),
        ev("call", {"name": "internet_search", "args": {"query": "q"}, "id": "a", "sub": None}, 1.5),
        ev("tool", {"name": "internet_search", "id": "a", "sub": None, "preview": "3 results"}, 3.5),
        ev("call", {"name": "execute", "args": {"command": "ls"}, "id": "b", "sub": None}, 3.6),
        ev("token", "Here is ", 3.7), ev("token", "the answer", 3.8),
    ])
    assert live.intake == "INTAKE"
    assert [s.doing for s in live.running] == ["Running a command in the sandbox"]  # the search came back; the command has not
    assert [(s.done, s.result, round(s.seconds, 1)) for s in live.finished] == [("Searched the web", "3 results", 2.0)]
    assert live.text == "Here is the answer"  # the narration before the first tool call is not shown as the answer
    assert live.last_event == 3.8


def test_a_plan_an_error_and_a_stop_show_up_in_the_live_view():
    todos = [{"content": "a", "status": "pending"}]
    live = tr.summarize_feed([ev("plan", todos), ev("error", "boom"), ev("cancelled")])
    assert live.plan == todos and live.error == "boom" and live.cancelled
    assert tr.summarize_feed([]).running == [] and tr.summarize_feed([]).text == ""


def test_a_result_without_a_matching_call_still_appears_as_a_finished_step():
    live = tr.summarize_feed([ev("subagent", {"name": "read_file", "id": "zz", "sub": "coder", "preview": "FAILED: nope"})])
    assert live.running == [] and live.finished[0].sub == "coder" and live.finished[0].failed


def test_the_plan_tool_is_not_listed_as_a_step_in_the_live_view():
    live = tr.summarize_feed([ev("plan", [{"content": "a", "status": "pending"}]),
                              ev("tool", {"name": "write_todos", "id": "p", "sub": None, "preview": "Updated todo list"})])
    assert live.finished == [] and live.plan


def test_the_streamed_answer_stays_visible_when_the_turn_ends():
    live = tr.summarize_feed([ev("intake", "I"), ev("token", "Hello "), ev("token", "there"), ev("final", "Hello there")])
    assert live.text == "Hello there"
