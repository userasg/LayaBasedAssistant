"""What the model sees for THIS request: a short recap of earlier ones, and only this request's own messages.

Measured in real use: every new request rode on the whole earlier conversation (old tool calls, old plans, old narration), so the 14B
mixed tasks up ("Jev" was searched as a virus), skipped planning because the previous request's todo list was still in state, and slowed
down as the thread grew. The saved thread is untouched (the chat window is drawn from it); only what is SENT to the model is scoped:

  [earlier requests: one line each, request -> result]  +  the current request and everything the agent has done for it so far
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from .transcript import build_turns, clip, text_of

RECAP_TURNS = 4  # how many earlier requests are recapped
RECAP_CHARS = 200


def recap(prior_messages, max_turns: int = RECAP_TURNS) -> str:
    turns = build_turns(prior_messages)[-max_turns:]
    lines = []
    for t in turns:
        result = t.answer or ("(stopped or unfinished)" if t.steps or t.plan else "(no reply)")
        lines.append(f"- {clip(t.prompt, 120)} -> {clip(result, RECAP_CHARS)}")
    return "\n".join(lines)


def scoped_messages(messages: list) -> list:
    """The messages to send: the current request (from its human message on) with a recap of earlier requests folded into that message."""
    humans = [i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"]
    if len(humans) < 2:
        return list(messages)
    last = humans[-1]
    summary = recap(messages[:last])
    current = messages[last]
    if summary:
        c = current.content
        head = ("[Earlier in this chat, for reference only. Do NOT redo or continue these; the request below is the only task now:\n"
                f"{summary}]\n\n")
        current = current.model_copy(update={"content": head + c if isinstance(c, str) else [{"type": "text", "text": head}, *c]})
    return [current, *messages[last + 1:]]
