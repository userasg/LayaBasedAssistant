"""What the chat window shows: one clean answer per turn, with the work folded away.

The agent's message history interleaves the person's prompt, the model's narration, tool calls and tool results. Drawn as it
came, that was a wall of text: every "I'll now search..." became its own chat bubble. These are pure functions over LangChain
messages and session Events (no Streamlit, no model), so the folding rules are tested on their own.

  build_turns(messages, metas)  history  -> [Turn]   prompt, the final answer, the steps taken, the plan, timings
  summarize_feed(events)        live feed -> Live    what is happening right now, in words
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def text_of(content) -> str:
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content if isinstance(b, dict))


def clip(text, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def fmt_seconds(s: float) -> str:
    if s < 10:
        return f"{s:.1f} s" if s >= 0.1 else f"{s * 1000:.0f} ms"
    if s < 60:
        return f"{s:.0f} s"
    return f"{int(s // 60)} min {int(s % 60)} s"


# -- describing tool calls in words a person would use -------------------------------------------------------------------

# tool -> (icon, while it runs, once done, the argument that says what it was about)
TOOLS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "execute": ("💻", "Running a command in the sandbox", "Ran a command in the sandbox", ("command",)),
    "write_file": ("📝", "Writing a file", "Wrote a file", ("file_path", "path")),
    "read_file": ("📖", "Reading a file", "Read a file", ("file_path", "path")),
    "edit_file": ("✏️", "Editing a file", "Edited a file", ("file_path", "path")),
    "ls": ("📂", "Looking in a folder", "Looked in a folder", ("path",)),
    "glob": ("🔎", "Finding files", "Found files", ("pattern",)),
    "grep": ("🔎", "Searching the files", "Searched the files", ("pattern",)),
    "delete": ("🗑", "Moving a file to the trash", "Moved a file to the trash", ("path", "file_path")),
    "internet_search": ("🔎", "Searching the web", "Searched the web", ("query",)),
    "computer_confirm": ("✅", "Doing the step you approved", "Did the step you approved", ()),
    "computer_cancel": ("⏹", "Dropping the step", "Dropped the step", ()),
    "mac_notes_create": ("🗒", "Making a note", "Made a note", ("title",)),
    "mac_open_app": ("🚀", "Opening an app", "Opened an app", ("name",)),
    "mac_app_action": ("🎛", "Controlling a Mac app", "Controlled a Mac app", ("app",)),
    "mac_run": ("🖥", "Running on your Mac", "Ran on your Mac", ("command",)),
    "describe_image": ("👁", "Looking at an image", "Looked at an image", ("path",)),
}
_TARGETS = {"browser": "the browser", "my_chrome": "your Chrome", "desktop": "a Mac app"}


@dataclass
class Step:
    """One thing the agent did (or is doing): a tool call, or a line of narration."""
    icon: str
    doing: str  # "Searching the web"
    done: str  # "Searched the web"
    detail: str = ""  # what it was about: the query, the command, the path
    result: str = ""  # the first lines of what came back
    sub: str | None = None  # the subagent it ran inside, if any
    failed: bool = False
    kind: str = "tool"  # "tool" | "say"
    started: float | None = None
    seconds: float | None = None


def describe(name: str, args: dict | None, sub: str | None = None) -> Step:
    args = args or {}
    if name == "task":  # delegating to a subagent
        who = str(args.get("subagent_type") or "helper")
        return Step("🤝", f"Asking the {who}", f"Asked the {who}", clip(args.get("description", ""), 110), sub=sub)
    if name == "computer_use":
        where = _TARGETS.get(str(args.get("target", "browser")), "the browser")
        return Step("🌐", f"Working in {where}", f"Worked in {where}", clip(args.get("goal", ""), 110), sub=sub)
    icon, doing, done, keys = TOOLS.get(name, ("🛠", f"Using {name}", f"Used {name}", ()))
    detail = next((str(args[k]) for k in keys if args.get(k)), "")
    if not detail and name not in TOOLS and args:
        detail = str(args)
    return Step(icon, doing, done, clip(detail, 110), sub=sub)


_FAIL = re.compile(r"^\s*(FAILED|STALLED|UNAVAILABLE|BLOCKED|ERROR|Error|Not created|Refused|Traceback)", re.I)
_EXIT = re.compile(r"exit(?: code)? (\d+)", re.I)


def looks_failed(text: str, status: str | None = None) -> bool:
    if status == "error" or _FAIL.match(text or ""):
        return True
    m = _EXIT.search((text or "")[:80])
    return bool(m and m.group(1) != "0")


def result_line(text: str, n: int = 170) -> str:
    """The first couple of non-empty lines of a tool result, for a one-line preview."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return clip(" · ".join(lines[:2]), n)


# -- the history: prompt -> answer, with the work folded ---------------------------------------------------------------------

@dataclass
class TurnMeta:
    """Timings and routing for one request, kept by the session (the message history does not carry them)."""
    route: str = ""  # "fast" | "executor"
    intent: str = ""
    laya_ms: float = 0.0
    first_token_ms: float | None = None
    elapsed: float = 0.0  # seconds the assistant worked on it (time spent waiting for your OK is not counted)
    error: str | None = None
    cancelled: bool = False


@dataclass
class Turn:
    id: str
    prompt: str
    attachments: list[str] = field(default_factory=list)
    early: list[str] = field(default_factory=list)  # things already done while the person was still speaking
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)
    meta: TurnMeta | None = None


# markers the app appends to a prompt for the agent's benefit; the person never typed them
_MARKER = re.compile(r"\n\n\[(?:Attached files:|Already done while the user was speaking:|Already done: opened|Laya note)")
_ATTACHED = re.compile(r"\[Attached files: ([^\]]*)\]")
_EARLY = re.compile(r"\[Already done while the user was speaking: (.*?)\. Do not repeat it\.\]", re.S)


def clean_prompt(text: str) -> tuple[str, list[str], list[str]]:
    """(what the person said, attached file names, things already done while they spoke)."""
    prompt = _MARKER.split(text, maxsplit=1)[0].strip()
    files = [Path(p.strip()).name for m in _ATTACHED.findall(text) for p in m.split(",") if p.strip()]
    early = [c.strip() for m in _EARLY.findall(text) for c in m.split(";") if c.strip()]
    return prompt, files, early


def build_turns(messages, metas: dict | None = None) -> list[Turn]:
    metas = metas or {}
    turns: list[Turn] = []
    cur: Turn | None = None
    calls: dict[str, Step] = {}
    for m in messages:
        if m.type == "human":
            prompt, files, early = clean_prompt(text_of(m.content))
            cur = Turn(id=m.id or "", prompt=prompt, attachments=files, early=early, meta=metas.get(m.id))
            turns.append(cur)
            calls = {}
        elif cur is None:
            continue  # anything before the first prompt (system messages) is not part of a turn
        elif m.type == "ai":
            said = text_of(m.content).strip()
            tool_calls = getattr(m, "tool_calls", None) or []
            fresh = []
            for c in tool_calls:
                if c["name"] == "write_todos":
                    cur.plan = c["args"].get("todos", [])
                    continue
                step = describe(c["name"], c["args"])
                calls[c.get("id") or f"{len(calls)}"] = step
                fresh.append(step)
            if tool_calls and said:  # narration next to a tool call belongs to the work, not the answer
                cur.steps.append(Step("💬", clip(said, 220), clip(said, 220), kind="say"))
            elif said:
                cur.answer = said
            cur.steps.extend(fresh)
        elif m.type == "tool":
            step = calls.get(getattr(m, "tool_call_id", None))
            if step is not None:
                body = text_of(m.content)
                step.result = result_line(body)
                step.failed = looks_failed(body, getattr(m, "status", None))
    return turns


def meta_line(turn: Turn) -> str:
    """One quiet line under an answer: how it was handled and how long it took."""
    m = turn.meta
    if m is None:
        return ""
    if m.error:
        return f"⚠️ {clip(m.error, 140)}"
    bits = []
    if m.route == "fast":
        bits.append("⚡ fast")
    elif m.route:
        bits.append("🧠 agent")
    if m.laya_ms:
        bits.append(f"Laya {m.laya_ms:.0f} ms")
    if m.first_token_ms:
        bits.append(f"first word {m.first_token_ms:.0f} ms")
    tools = [s for s in turn.steps if s.kind == "tool"]
    if m.route != "fast" and tools:
        bits.append(f"{len(tools)} step{'s' if len(tools) != 1 else ''}")
    if m.elapsed:
        bits.append(fmt_seconds(m.elapsed))
    if m.cancelled:
        bits.append("stopped by you")
    return " · ".join(bits)


# -- the live feed: what is happening right now ------------------------------------------------------------------------------

@dataclass
class Live:
    intake: object | None = None
    plan: list[dict] = field(default_factory=list)
    running: list[Step] = field(default_factory=list)  # tool calls that have not come back yet
    finished: list[Step] = field(default_factory=list)
    text: str = ""  # what the model is writing right now (the answer, once the tools are done)
    error: str | None = None
    cancelled: bool = False
    last_event: float | None = None


def summarize_feed(events) -> Live:
    live = Live()
    open_calls: dict[str, Step] = {}
    tokens: list[str] = []
    for e in events:
        live.last_event = e.t if hasattr(e, "t") else live.last_event
        if e.kind == "token":
            tokens.append(e.data)
            continue
        if e.kind == "intake":
            live.intake = e.data
            continue
        if e.kind in ("final", "interrupt"):
            continue  # the turn is ending: keep showing what was streamed until the history takes over
        tokens.clear()  # anything the model wrote before a tool call was narration; the answer starts after the last one
        if e.kind == "plan":
            live.plan = e.data
        elif e.kind == "call":
            step = describe(e.data["name"], e.data.get("args"), e.data.get("sub"))
            step.started = getattr(e, "t", None)
            open_calls[e.data.get("id") or f"anon{len(open_calls)}"] = step
        elif e.kind in ("tool", "subagent"):
            if e.data.get("name") == "write_todos":
                continue  # the plan is drawn from its own event, not as a step
            step = open_calls.pop(e.data.get("id") or "", None) or describe(e.data["name"], {}, e.data.get("sub"))
            body = str(e.data.get("preview", ""))
            step.result, step.failed = result_line(body), looks_failed(body)
            if step.started is not None and hasattr(e, "t"):
                step.seconds = max(0.0, e.t - step.started)
            live.finished.append(step)
        elif e.kind == "error":
            live.error = str(e.data)
        elif e.kind == "cancelled":
            live.cancelled = True
    live.running = list(open_calls.values())
    live.text = "".join(tokens)
    return live
