"""The conversation column: what was said, what is happening now, and the card that asks for your OK.

  history(session)          finished turns: your message, ONE answer, a quiet timing line, and "Show work" folded underneath
  render_live(session)      the running turn: a headline that says what is happening, the plan ticking off, a Stop button
  render_approval(session)  the one place the assistant waits for you (a plan, a command, a delete, an irreversible step)
  render_empty()            the first screen: a few things to try
"""
from __future__ import annotations

import re
import time

import streamlit as st

from ..session import AssistantSession
from ..transcript import Step, Turn, build_turns, clean_prompt, fmt_seconds, meta_line, summarize_feed

VISIBLE_TURNS = 8  # older turns fold into one expander so the page stays small however long the chat gets
DETAIL_TURNS = 3  # only the latest turns keep their "Show work" ready to open
ROUTE = {"fast": "⚡ fast model", "executor": "🧠 Deep Agent"}
PLAN_ICON = {"completed": "✅", "in_progress": "🔄", "pending": "⬜"}

# Things that work end to end on this stack (checked on the real app), one per way the assistant can act.
EXAMPLES = [
    ("🌐", "Search the web for the best budget noise-cancelling headphones and give me the top three"),
    ("📝", "Open Notes and write a shopping list: milk, eggs, bread"),
    ("📁", "What is in my workspace folder?"),
    ("💻", "Write a Python script that prints the first 20 primes, then run it"),
]

_MD = re.compile(r"([\\`*_\[\]<>!#|~])")


def plain(text: str) -> str:
    """Text from the model or a tool, made safe to show inside markdown (no links, images, emphasis or HTML from it)."""
    return _MD.sub(r"\\\1", str(text))


def code(text: str) -> str:
    return "`" + str(text).replace("`", "'") + "`"


def plan_md(todos: list[dict]) -> str:
    rows = []
    for t in todos:
        status = t.get("status", "pending")
        label = plain(t.get("content", ""))
        rows.append(f"{PLAN_ICON.get(status, '⬜')} " + (f"**{label}**" if status == "in_progress" else label))
    return "  \n".join(rows)


def steps_md(steps: list[Step]) -> str:
    out = []
    for s in steps:
        if s.kind == "say":
            out += ["", f"> {plain(s.doing)}", ""]
            continue
        who = f"*{plain(s.sub)}* · " if s.sub else ""
        out.append(f"- {s.icon} {who}**{s.done}**" + (" ⚠️" if s.failed else "") + (f" {code(s.detail)}" if s.detail else ""))
        if s.result:
            out.append(f"    - ↳ {code(s.result)}")
    return "\n".join(out)


# -- finished turns ----------------------------------------------------------------------------------------------------------

def _chips(prompt_files: list[str], early: list[str]) -> None:
    if prompt_files:
        st.caption("📎 " + ", ".join(plain(f) for f in prompt_files))
    if early:
        st.caption("⚡ Already done while you were speaking: " + "; ".join(plain(e) for e in early))


def _work(turn: Turn) -> None:
    tools = sum(s.kind == "tool" for s in turn.steps)
    label = f"Show work · {tools} step{'s' if tools != 1 else ''}" if tools else "Show plan"
    with st.expander(label):
        if turn.plan:
            st.markdown("**Plan**")
            st.markdown(plan_md(turn.plan))
        if turn.steps:
            st.markdown("**What I did**")
            st.markdown(steps_md(turn.steps))


def render_turn(turn: Turn, detail: bool = True, waiting: bool = False) -> None:
    with st.chat_message("user"):
        st.markdown(turn.prompt)
        _chips(turn.attachments, turn.early)
    with st.chat_message("assistant"):
        m = turn.meta
        if turn.answer:
            st.markdown(turn.answer)
        elif waiting:
            st.caption("⏸ Waiting for your OK below.")
        elif m and m.cancelled:
            st.caption("⏹ Stopped before it finished.")
        elif not (m and m.error):
            st.caption("No reply.")
        line = meta_line(turn)
        if line:
            st.caption(line)
        if detail and (turn.steps or turn.plan):
            _work(turn)


def history(session: AssistantSession) -> int:
    """Draw the finished turns; returns how many there are (the running one is drawn by render_live, not here)."""
    messages = session.agent.get_state(session.config).values.get("messages", [])
    turns = build_turns(messages, session.metas)
    if session.busy:
        turns = [t for t in turns if t.id != session.running_id]
    waiting = session.pending is not None and not session.busy
    hidden, shown = turns[:-VISIBLE_TURNS], turns[-VISIBLE_TURNS:]
    if hidden:
        with st.expander(f"{len(hidden)} earlier turns"):
            for t in hidden:
                render_turn(t, detail=False)
    for i, t in enumerate(shown):
        render_turn(t, detail=len(shown) - i <= DETAIL_TURNS, waiting=waiting and i == len(shown) - 1)
    return len(turns)


# -- the running turn ---------------------------------------------------------------------------------------------------------

def _headline(session: AssistantSession, live) -> str:
    if session.stopping:
        return "Stopping…"
    if live.running:
        return f"{live.running[0].doing}…"
    if live.text:
        return "Writing…"  # not "the answer": until the message ends it may be narration before a tool call
    if live.intake is None:
        return "Understanding your request…"
    return "Thinking…"


def render_live(session: AssistantSession) -> None:
    """The running turn, redrawn from the worker's feed every 0.4 s. Your message shows at once; the assistant's card says what it
    is doing right now, ticks the plan off as it goes, and keeps the last few finished steps in view."""
    live = summarize_feed(session.feed())
    now = time.time()
    started = session.turn_started or now
    prompt, files, early = clean_prompt(session.running_prompt or "")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
            _chips(files, early)
    with st.chat_message("assistant"):
        with st.container(key="live"):
            head, stop = st.columns([5, 1], vertical_alignment="center")
            with head.container(key="live_head_stopping" if session.stopping else "live_head"):
                st.markdown(f"**{_headline(session, live)}** · {fmt_seconds(now - started)}")
            if stop.button("Stop", key="stop_turn", icon=":material/stop_circle:", disabled=session.stopping, width="stretch",
                           help="Ends this request. A model call already in progress finishes first."):
                session.cancel()
                st.toast("Stopping…")
            if live.intake is not None:
                i = live.intake
                st.caption(f"🟢 Laya · {i.intent} → {ROUTE.get(i.route, i.route)} · {i.ms:.0f} ms")
            if live.plan:
                st.markdown(plan_md(live.plan))
            for s in live.running:
                took = f" · {fmt_seconds(now - s.started)}" if s.started else ""
                st.markdown(f"⏳ **{s.doing}…**" + (f" {code(s.detail)}" if s.detail else "") + took)
            for s in live.finished[-2:]:
                st.caption(f"✓ {s.done}" + (" — failed" if s.failed else "") + (f" · {fmt_seconds(s.seconds)}" if s.seconds is not None else ""))
            cu = session.computer.step_log[-3:] if session.computer is not None else []
            for row in cu:
                st.caption(f"⚡ {plain(row['step'][:70])} · {row['status']} · {row['ms']} ms · {row['tier']}")
            if live.text:
                st.markdown(live.text + "▌")
            if live.error:
                st.error(live.error)


def render_live_compact(session: AssistantSession) -> None:
    """Shown on the Files view while a request runs, so the assistant is never silently busy."""
    live = summarize_feed(session.feed())
    st.info(f"**{_headline(session, live)}** · {fmt_seconds(time.time() - (session.turn_started or time.time()))} · go back to the chat to watch")


# -- asking for your OK -----------------------------------------------------------------------------------------------------------

TITLES = {
    "write_todos": "📋 Review the plan",
    "mac_run": "✋ Run this on your Mac?",
    "delete": "🗑 Move this file to the trash?",
    "computer_confirm": "✋ This step can't be undone",
}


def _plan_editor(i: int, action: dict) -> list[dict]:
    """Tutorial 11's editable checklist: untick a step to drop it, or retype it."""
    edited = []
    for j, todo in enumerate(action["args"].get("todos", [])):
        c1, c2 = st.columns([1, 12], vertical_alignment="center")
        keep = c1.checkbox("keep", value=True, key=f"keep_{i}_{j}", label_visibility="collapsed")
        text = c2.text_input("step", value=todo["content"], key=f"txt_{i}_{j}", label_visibility="collapsed")
        if keep:
            edited.append({**todo, "content": text})
    return edited


def _describe_action(session: AssistantSession, action: dict) -> None:
    name, args = action["name"], action.get("args", {})
    if name == "mac_run":
        st.code(str(args.get("command", "")), language="bash")
        risk = action.get("laya_risk")
        st.caption("Runs on your Mac, outside the sandbox." + (f" Laya rates it {risk:.2f} likely to change or delete something." if risk is not None else ""))
    elif name == "delete":
        st.markdown(f"Move {code(args.get('path') or args.get('file', ''))} to the trash. You can restore it from 📁 Files.")
    elif name == "computer_confirm":
        job = session.computer.jobs.get(args.get("pending_id")) if session.computer else None
        if job:
            st.markdown(f"**{plain(job.pending.description)}**")
            st.caption(plain(job.pending.reason))
        else:
            st.code(str(args), language="text")
    else:
        st.markdown(f"**{plain(name)}**")
        st.code(str(args), language="text")


def render_approval(session: AssistantSession) -> None:
    reqs = session.pending["action_requests"]
    decisions = []
    with st.container(border=True, key="approval"):
        st.markdown(f"**{TITLES.get(reqs[0]['name'], '✋ Needs your OK')}**")
        with st.form("review", border=False):
            for i, action in enumerate(reqs):
                if action["name"] == "write_todos":
                    decisions.append(("plan", action, _plan_editor(i, action)))
                else:
                    _describe_action(session, action)
                    decisions.append(("ask", action, None))
            c1, c2, _ = st.columns([1.2, 1.2, 4])
            approve = c1.form_submit_button("Approve", type="primary", icon=":material/check:", width="stretch")
            reject = c2.form_submit_button("Reject", icon=":material/close:", width="stretch")
        st.caption("Or just type **yes**, **no**, or what to change in the box below.")
    if approve or reject:
        out = []
        for kind, action, edited in decisions:
            if reject:
                out.append({"type": "reject"})
            elif kind == "plan" and edited != action["args"].get("todos", []):
                out.append({"type": "edit", "edited_action": {"name": "write_todos", "args": {"todos": edited}}})
            else:
                out.append({"type": "approve"})
        session.submit_resume(out)
        st.rerun()


# -- the first screen ---------------------------------------------------------------------------------------------------------------

def render_empty() -> str | None:
    """A greeting and a few things to try. Returns the example the person clicked, if any."""
    st.markdown("#### What can I help with?")
    st.caption("Type below, or press **🎙 Talk** and just speak. Laya decides how each request is handled, and the timings show under every answer.")
    chosen = None
    with st.container(key="examples"):
        cols = st.columns(2)
        for i, (icon, prompt) in enumerate(EXAMPLES):
            if cols[i % 2].button(f"{icon}  {prompt}", key=f"example_{i}", width="stretch"):
                chosen = prompt
    return chosen
