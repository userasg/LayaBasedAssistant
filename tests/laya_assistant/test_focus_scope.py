"""Getting the chat out of the way, the show-windows switch, per-request context scoping and the planner's JSON repair."""
import types

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from laya_assistant import approvals, scope
from laya_assistant.cua import apps, focus
from laya_assistant.cua.planner import parse_plan


class Recorder:
    """Stands in for subprocess.run for osascript/open only: records what would have been done to the Mac."""

    def __init__(self, front="Google Chrome"):
        self.front, self.calls = front, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        out = self.front if any("frontmost is true" in part for part in cmd) else ""
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)

    def visibility(self):
        return [(c[-1], "false" if any("to false" in p for p in c) else "true") for c in self.calls if any("set visible" in p for p in c)]


@pytest.fixture
def host_on(monkeypatch):
    monkeypatch.setenv("LAYA_HOST_ACTIONS", "on")


def test_the_chat_window_is_hidden_during_the_task_and_always_shown_again(host_on):
    r = Recorder("Google Chrome")
    with pytest.raises(RuntimeError):
        with focus.chat_out_of_the_way("Music", runner=r):
            assert r.visibility() == [("Google Chrome", "false")]  # hidden while the task runs
            raise RuntimeError("the task blew up")
    assert r.visibility() == [("Google Chrome", "false"), ("Google Chrome", "true")]  # and back, however it ended
    assert any(c[:2] == ["osascript", "-e"] and c[-1] == "Google Chrome" and any("activate" in p for p in c) for c in r.calls)


@pytest.mark.parametrize("front", ["Music", "Finder", "loginwindow", ""])
def test_it_never_hides_the_app_being_driven_or_a_system_process(host_on, front):
    r = Recorder(front)
    with focus.chat_out_of_the_way("Music", runner=r):
        pass
    assert r.visibility() == []


def test_it_does_nothing_when_host_actions_are_off():
    r = Recorder("Google Chrome")
    with focus.chat_out_of_the_way("Music", runner=r):
        pass
    assert r.calls == []


def test_the_app_name_is_an_argument_never_part_of_the_script(host_on):
    r = Recorder('Chrome" & (do shell script "rm -rf ~") & "')
    with focus.chat_out_of_the_way("Music", runner=r):
        pass
    for cmd in r.calls:
        assert all("rm -rf" not in part for part in cmd[:-1])


def test_background_mode_launches_without_raising_anything(host_on):
    calls = []
    apps.reopen_and_activate("Music", runner=lambda cmd, **kw: calls.append(cmd), foreground=False)
    assert calls == [["open", "-g", "-a", "Music"]]  # no reopen, no activate
    calls.clear()
    apps.reopen_and_activate("Music", runner=lambda cmd, **kw: calls.append(cmd))
    assert calls[0] == ["open", "-a", "Music"] and calls[1][0] == "osascript"


def test_in_background_mode_an_app_on_another_desktop_is_reported_not_dragged_forward(host_on):
    class Cli:
        def call(self, tool, args=None, timeout=None):
            return {"windows": [{"app_name": "Music", "pid": 1, "window_id": 2, "bounds": {"width": 900, "height": 600}, "is_on_screen": False,
                                 "on_current_space": False}]}

    got = []
    p = apps.bring_up("Music", Cli(), wait=0.4, launcher=lambda a, foreground=True: got.append(foreground), foreground=False)
    assert got == [False] and not p.ok and "show app windows" in p.reason


def test_the_show_windows_switch_defaults_on_and_is_a_real_setting():
    assert approvals.current.windows is True


# --- context scoping ---------------------------------------------------------------------------------------------------------

def convo():
    return [
        HumanMessage(content="open apple music and play a song", id="h1"),
        AIMessage(content="", tool_calls=[{"name": "computer_use", "args": {"target": "desktop", "goal": "play"}, "id": "c1", "type": "tool_call"}]),
        ToolMessage(content="DONE (3 steps)", tool_call_id="c1", name="computer_use"),
        AIMessage(content="Playing a song from your Odd Future playlist."),
        HumanMessage(content="hi", id="h2"), AIMessage(content="Hello!"),
        HumanMessage(content="research jev online then write it in notion", id="h3"),
        AIMessage(content="", tool_calls=[{"name": "write_todos", "args": {"todos": []}, "id": "t1", "type": "tool_call"}]),
    ]


def test_the_model_sees_this_request_plus_a_recap_never_the_old_tool_calls():
    out = scope.scoped_messages(convo())
    assert [m.type for m in out] == ["human", "ai"]  # this request and what the agent has done for it so far
    text = out[0].content
    assert text.endswith("research jev online then write it in notion")
    assert "open apple music and play a song -> Playing a song from your Odd Future playlist." in text and "hi -> Hello!" in text
    assert "computer_use" not in text and "DONE (3 steps)" not in text and "Do NOT redo" in text


def test_the_recap_is_limited_and_the_first_request_is_untouched():
    msgs = []
    for i in range(9):
        msgs += [HumanMessage(content=f"task {i}", id=f"h{i}"), AIMessage(content=f"done {i}")]
    msgs.append(HumanMessage(content="now this", id="hx"))
    text = scope.scoped_messages(msgs)[0].content
    assert text.count("\n- ") + text.startswith("- ") <= scope.RECAP_TURNS and "task 8 -> done 8" in text and "task 0" not in text
    first = [HumanMessage(content="only one", id="a"), AIMessage(content="x")]
    assert scope.scoped_messages(first) == first


def test_scoping_does_not_change_the_saved_thread():
    msgs = convo()
    before = [m.content for m in msgs]
    scope.scoped_messages(msgs)
    assert [m.content for m in msgs] == before  # the chat window is drawn from the saved thread


# --- the planner's JSON repair -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reply", [
    '[{"do": "open_app", "value": "Music"}, {"do": "done"},]',  # trailing comma
    "Here you go:\n```json\n[{'do': 'open_app', 'value': 'Music'}, {'do': 'done'}]\n```",  # single quotes
    '[{"do": "open_app", // start\n "value": "Music"}, {"do": "done"}]',  # a comment
    '[{“do”: “open_app”, “value”: “Music”}, {“do”: “done”}]',  # smart quotes
])
def test_the_usual_small_model_json_slips_are_repaired(reply):
    steps = parse_plan(reply)
    assert [s.do for s in steps] == ["open_app", "done"] and steps[0].value == "Music"


def test_unrepairable_output_is_a_clean_value_error():
    with pytest.raises(ValueError):
        parse_plan("[{do: open_app value Music")
