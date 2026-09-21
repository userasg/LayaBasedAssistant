"""The questions the agent asks the person (risky command, delete, plan) share one protocol, and a typed answer really decides.

The graph, the pause/resume, the cortex middleware and Laya are all real. Only the chat model is scripted, because the test has to
force the agent to propose one particular command; it is the one thing a real 14B would not do reliably on demand.
"""
import types

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from laya_assistant import cortex, host, review
from laya_assistant.decisions import DecisionLog
from laya_assistant.session import decisions_from_text


def test_only_a_clear_approval_approves():
    assert review.decision({"decisions": [{"type": "approve"}]})["type"] == "approve"
    assert review.decision("approve")["type"] == "approve" and review.decision("reject")["type"] == "reject"
    reply = review.decision({"decisions": [{"type": "respond", "message": "use ~/tmp instead"}]})
    assert reply == {"type": "respond", "message": "use ~/tmp instead"}
    for junk in (None, {}, {"decisions": []}, {"decisions": [{}]}, {"decisions": ["approve"]}, {"decisions": [{"type": "sure"}]}, "yes", 3, []):
        assert review.decision(junk)["type"] == "reject", junk  # an empty or odd reply must never count as a yes


def test_a_declined_request_tells_the_agent_what_to_do_next():
    assert review.declined({"type": "approve"}, "this") is None
    assert "Rejected by the user" in review.declined({"type": "reject"}, "this command")
    typed = review.declined({"type": "respond", "message": "only the .tmp files"}, "this command")
    assert "only the .tmp files" in typed and "follow their answer" in typed


def test_the_request_has_the_shape_the_approval_card_reads():
    r = review.request("mac_run", "Run on your Mac: rm x", {"command": "rm x"}, laya_risk=0.8)
    (action,) = r["action_requests"]
    assert action == {"name": "mac_run", "args": {"command": "rm x"}, "description": "Run on your Mac: rm x", "laya_risk": 0.8}
    assert r["review_configs"][0]["allowed_decisions"] == ["approve", "reject", "respond"]


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def build_agent(laya_model, command, runner):
    script = iter([AIMessage(content="", tool_calls=[{"name": "mac_run", "args": {"command": command}, "id": "c1", "type": "tool_call"}]),
                   AIMessage(content="All done.")])
    return create_agent(ScriptedModel(messages=script), tools=host.make_host_tools(runner=runner),
                        middleware=[cortex.LayaCortexMiddleware(laya_model, DecisionLog(), host_tools={"mac_run"})], checkpointer=InMemorySaver())


def run_until_asked(laya_model, command="mv ~/a.txt ~/b.txt"):
    ran = []
    runner = lambda cmd, **kw: ran.append(cmd) or types.SimpleNamespace(returncode=0, stdout="moved", stderr="")
    agent = build_agent(laya_model, command, runner)
    cfg = {"configurable": {"thread_id": "t"}}
    out = agent.invoke({"messages": [HumanMessage(content="tidy up my files")]}, cfg)
    assert "__interrupt__" in out, "a risky command should have paused for the person"
    return agent, cfg, out["__interrupt__"][0].value, ran


def test_a_risky_command_pauses_with_a_card_the_ui_can_draw(laya_model):
    _, _, payload, ran = run_until_asked(laya_model)
    (action,) = payload["action_requests"]
    assert action["name"] == "mac_run" and action["args"]["command"] == "mv ~/a.txt ~/b.txt" and "mv ~/a.txt" in action["description"]
    assert ran == []  # nothing ran while it waited


@pytest.mark.parametrize("typed,should_run", [("yes", True), ("Go ahead!", True), ("no", False), ("nope", False)])
def test_typing_yes_or_no_in_the_box_decides(laya_model, typed, should_run):
    agent, cfg, payload, ran = run_until_asked(laya_model)
    n = len(payload["action_requests"])
    out = agent.invoke(Command(resume={"decisions": decisions_from_text(typed, n)}), cfg)  # exactly what the chat box sends
    assert bool(ran) is should_run, f"{typed!r}: ran={ran}"
    assert out["messages"][-1].content == "All done."


def test_a_typed_answer_that_is_not_a_plain_yes_reaches_the_agent_and_runs_nothing(laya_model):
    agent, cfg, payload, ran = run_until_asked(laya_model)
    out = agent.invoke(Command(resume={"decisions": decisions_from_text("yes but copy it instead", 1)}), cfg)
    assert ran == []
    tool_reply = next(m for m in out["messages"] if m.type == "tool")
    assert "copy it instead" in tool_reply.content and tool_reply.status == "error"


def test_deleting_a_shared_file_follows_the_same_protocol(tmp_path):
    from langchain_core.messages import ToolMessage

    from laya_assistant import policy
    from laya_assistant.files import SharedFolder

    shared = SharedFolder(tmp_path)
    for reply, kept in [({"decisions": [{"type": "reject"}]}, True), ({"decisions": [{"type": "respond", "message": "keep it"}]}, True),
                        ({"decisions": [{"type": "approve"}]}, False)]:
        shared.save_upload("doc.txt", b"precious")
        mw = policy.PathPolicyMiddleware(shared=shared)
        call = types.SimpleNamespace(tool_call={"id": "d1", "name": "delete", "args": {"file_path": "/workspace/shared/doc.txt"}}, state={})
        import unittest.mock as mock

        with mock.patch.object(policy, "interrupt", lambda payload: reply):  # the reply exactly as the resume delivers it
            out = mw.wrap_tool_call(call, lambda r: None)
        assert isinstance(out, ToolMessage) and (shared.root / "doc.txt").exists() is kept
        if not kept:
            assert "trash" in out.content.lower()
        else:
            assert out.status == "error" and "doc.txt was not touched" in out.content
            shared.trash("doc.txt")  # reset for the next case
