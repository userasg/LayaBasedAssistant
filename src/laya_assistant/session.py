"""One chat session, UI-free: routes each turn (fast model vs Deep Agent), streams typed events, handles resume.

`run_turn` / `resume` are generators of `Event`s, so the Streamlit app (or a test) just renders them:
  intake     Intake decision (route, intent, probabilities, ms)
  token      a streamed piece of the assistant's answer (fast route or executor)
  call       a tool is about to run (name, args, id): what the agent is doing right now
  tool       a tool result finished (name, id, short preview)
  subagent   activity inside a subagent (name, id, short preview)
  plan       the todo list changed
  interrupt  paused for a human: the plan for approval (write_todos) or a risky action
  final      the finished answer text
  error      the run stopped (step budget, model failure)
  cancelled  the person pressed Stop; nothing after this event was read
"""
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from . import agents, config, intake
from .cortex import GATE_QUESTIONS
from .decisions import DecisionLog
from .engine import llm_lock, make_llm, warm_ollama
from .files import SharedFolder
from .sandbox import make_backend
from .transcript import TurnMeta, text_of as _text

import re

_YES = re.compile(r"^\s*(y|yes|yep|yeah|yup|ok|okay|sure|go|go ahead|do it|proceed|approve|approved|confirm|looks good|lgtm|fine|please do|that'?s fine)\s*[.!]*\s*$", re.I)
_NO = re.compile(r"^\s*(n|no|nope|nah|cancel|stop|reject|don'?t|do not|abort|never ?mind|not now)\s*[.!]*\s*$", re.I)


def decisions_from_text(text: str, n: int) -> list[dict]:
    """Answer pending approvals in plain words. An exact yes / no decides; anything else (including "yes, but change step 2")
    is passed to the agent as the user's reply, so a nuanced answer is never misread as a plain yes."""
    if _YES.match(text):
        return [{"type": "approve"} for _ in range(n)]
    if _NO.match(text):
        return [{"type": "reject"} for _ in range(n)]
    return [{"type": "respond", "message": text.strip()} for _ in range(n)]


FAST_SYSTEM = "You are a friendly, concise assistant. Answer in one or two short sentences."


@dataclass
class Event:
    kind: str
    data: Any = None
    t: float = field(default_factory=time.time)  # when it happened: the live view shows how long each step took


class AssistantSession:
    def __init__(self, predictor, handle, log, agent, fast_llm, session_dir: Path, shared: SharedFolder | None = None, computer=None):
        self.predictor, self.handle, self.log, self.agent, self.fast_llm = predictor, handle, log, agent, fast_llm
        self.shared, self.computer = shared, computer
        self.session_dir = session_dir
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": config.STEP_BUDGET}
        self.pending: dict | None = None
        self.last_intake: intake.Intake | None = None
        self._feed: list[Event] = []
        self._feed_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self.turn_started: float | None = None
        self.turn_open = False  # a turn ran since the UI last rendered its result
        self.metas: dict[str, TurnMeta] = {}  # routing and timings per request, keyed by the id of its human message
        self.meta: TurnMeta | None = None  # the request being worked on (an approval resumes the same one)
        self.running_id: str | None = None  # id and text of that request, known the instant it is submitted: the chat shows the
        self.running_prompt = ""  # person's message at once, before the graph has written it into the thread
        self._cancel = threading.Event()
        self._plan_calls: set[str] = set()  # ids of write_todos calls: the plan has its own event, so their results are not tool events

    @classmethod
    def create(cls, predictor, prefer_docker: bool = True, warm: bool = True, shared_dir=None, computer=None) -> "AssistantSession":
        session_dir = Path(tempfile.mkdtemp(prefix="laya_session_"))
        log = DecisionLog(path=session_dir / "decisions.jsonl")
        shared = SharedFolder(shared_dir)  # a real Mac folder, mounted in the sandbox at /workspace/shared
        handle = make_backend(prefer_docker, shared_dir=shared.root)
        agent = agents.build_assistant(predictor=predictor, log=log, sandbox_backend=handle.backend,
                                       llm=make_llm(config.EXECUTOR_MODEL), shared=shared, computer=computer)
        fast = make_llm(config.FAST_MODEL, temperature=0.4)
        if warm:
            for m in (config.FAST_MODEL, config.EXECUTOR_MODEL):
                warm_ollama(m)
            # The first safety-gate call costs ~300 ms (measured: 306 ms cold, 21 ms after); pay it here, not on the person's first command.
            predictor.predict({"command": "ls", "user_request": "hello"}, GATE_QUESTIONS)
        return cls(predictor, handle, log, agent, fast, session_dir, shared, computer)

    def close(self) -> None:
        self.stop_computer()
        self.handle.stop()

    def stop_computer(self) -> None:
        """The global Stop: halts a running computer-use goal before its next step."""
        if self.computer is not None:
            self.computer.stop.set()

    @property
    def stopping(self) -> bool:
        return self.busy and self._cancel.is_set()

    def cancel(self) -> None:
        """Stop the running turn. No further events are read from it, and a computer-use goal halts before its next step.
        A model call already in flight finishes first (it cannot be interrupted), so the turn ends a moment later, not at once."""
        self._cancel.set()
        self.stop_computer()

    # -- background turns -----------------------------------------------------------------------------
    # A turn runs on a worker thread that always runs to completion and appends Events to a feed the UI polls.
    # Streamlit reruns the whole script on every click, toggle or mic event; a turn consumed inside the script
    # is aborted by those, and can strand the GPU lock across a `yield`. Here nothing the page does can touch it.

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def feed(self) -> list[Event]:
        with self._feed_lock:
            return list(self._feed)

    def _start(self, gen_factory, new_request: bool = False) -> bool:
        if self.busy:
            return False
        with self._feed_lock:
            self._feed = []
        self._cancel.clear()
        if new_request:
            self.meta = None  # set again once Laya has looked at the request
        self.turn_started, self.turn_open = time.time(), True
        began = time.time()

        def push(ev: Event) -> None:
            with self._feed_lock:
                self._feed.append(ev)

        def work():
            gen = None
            try:
                gen = gen_factory()
                for ev in gen:
                    push(ev)
                    if ev.kind == "error" and self.meta is not None:
                        self.meta.error = str(ev.data)
                    # "intake" is not a stopping point: the request is only written into the thread once the turn proper starts, so
                    # stopping before that would make the person's message vanish from the chat
                    if self._cancel.is_set() and ev.kind not in ("intake", "final", "error", "interrupt"):
                        push(Event("cancelled"))
                        if self.meta is not None:
                            self.meta.cancelled = True
                        break
            except Exception as e:  # a model or tool failure must end the turn visibly, not kill the thread silently
                self.log.add("code", "error", f"{type(e).__name__}: {e}"[:90], None, 0.0)
                self._keep_failed_request(f"{type(e).__name__}: {e}")
                push(Event("error", f"{type(e).__name__}: {e}"))
            finally:
                try:
                    if gen is not None:
                        gen.close()  # a no-op for a generator that finished; after a Stop it unwinds the turn and frees the GPU lock
                except Exception as e:
                    self.log.add("code", "error", f"closing the turn: {type(e).__name__}: {e}"[:90], None, 0.0)
                if self.meta is not None:
                    self.meta.elapsed += time.time() - began

        self._worker = threading.Thread(target=work, name="turn", daemon=True)
        self._worker.start()
        return True

    def _keep_failed_request(self, error: str) -> None:
        """A turn that fails before the graph has written the person's message (Laya or the sandbox down) would otherwise vanish from
        the chat without a word: record the message and the error so the chat shows both."""
        try:
            if self.meta is None:
                self.meta = self.metas[self.running_id or uuid.uuid4().hex] = TurnMeta()
            self.meta.error = error
            have = {m.id for m in self.agent.get_state(self.config).values.get("messages", []) if m.type == "human"}
            if self.running_id and self.running_id not in have:
                self.agent.update_state(self.config, {"messages": [HumanMessage(content=self.running_prompt, id=self.running_id), AIMessage(content="")]})
        except Exception as e:  # never let the bookkeeping kill the worker
            self.log.add("code", "error", f"recording a failed request: {type(e).__name__}: {e}"[:90], None, 0.0)

    def submit(self, text: str, uploads: list[tuple[str, bytes]] | None = None) -> bool:
        if self.busy:
            return False
        turn_id = uuid.uuid4().hex  # the human message carries it, so the UI can join this request to its timings and to the live view
        self.running_id, self.running_prompt = turn_id, text
        return self._start(lambda: self.run_turn(text, uploads, turn_id), new_request=True)

    def submit_resume(self, decisions: list[dict]) -> bool:
        return self._start(lambda: self.resume(decisions))

    # -- files ----------------------------------------------------------------------------------------

    def list_files(self) -> list[str]:
        out = self.handle.backend.execute(
            "find /workspace -type f -not -path '*/.large_tool_results/*' -not -path '*/__pycache__/*' "
            "-not -path '*/.pytest_cache/*' 2>/dev/null | sort").output
        return [l for l in out.splitlines() if l.startswith("/workspace")]

    def list_files_sized(self) -> list[tuple[str, int]]:
        out = self.handle.backend.execute(
            "find /workspace -type f -not -path '*/.large_tool_results/*' -not -path '*/__pycache__/*' "
            "-not -path '*/.pytest_cache/*' -printf '%p\\t%s\\n' 2>/dev/null | sort").output
        rows = []
        for line in out.splitlines():
            path, _, size = line.partition("\t")
            if path.startswith("/workspace") and size.isdigit():
                rows.append((path, int(size)))
        return rows

    def download(self, path: str) -> bytes:
        got = self.handle.backend.download_files([path])[0]
        return got.content or b""

    # -- turns ----------------------------------------------------------------------------------------

    def _history(self, n: int = 6) -> list:
        msgs = self.agent.get_state(self.config).values.get("messages", [])
        keep = [m for m in msgs if m.type in ("human", "ai") and _text(m.content).strip()]
        return keep[-n:]

    def run_turn(self, text: str, uploads: list[tuple[str, bytes]] | None = None, turn_id: str | None = None) -> Iterator[Event]:
        note = ""
        if uploads:
            self.handle.backend.upload_files([(f"/workspace/uploads/{name}", data) for name, data in uploads])
            note = "\n\n[Attached files: " + ", ".join(f"/workspace/uploads/{n}" for n, _ in uploads) + "]"
        i = intake.classify(self.predictor, text, log=self.log)
        self.last_intake = i
        fast = i.route == "fast" and not uploads
        turn_id = turn_id or uuid.uuid4().hex
        self.meta = self.metas[turn_id] = TurnMeta(route="fast" if fast else "executor", intent=i.intent, laya_ms=i.ms)
        yield Event("intake", i)
        opened = self._open_first_clause(text)  # "open Music and play X": Music opens NOW, before any model plans anything
        if opened and opened[1]:
            yield from self._canned_turn(text, f"Opened {opened[0]}.", turn_id)
        elif fast:
            yield from self._fast_turn(text, turn_id)
        else:
            extra = f"\n\n[Already done: opened {opened[0]} and brought it to the front. Do not open it again.]" if opened else ""
            yield from self._executor_turn(text + note + extra, i, turn_id)

    def _open_first_clause(self, text: str):
        """(app, whole_request_was_just_that) if the request starts with "open <app>" for an installed app, and it was opened; else None.
        Exact code (an alias and app index, ~ms): no model is needed to open an app."""
        from .cua.apps import host_actions_enabled, reopen_and_activate
        from .stream_intent import EARLY_DENY_APPS, AppIndex, parse_quick, split_clauses

        if not host_actions_enabled():
            return None
        clauses, _ = split_clauses(text)
        if not clauses:
            return None
        if not hasattr(self, "_app_index"):
            self._app_index = AppIndex()
        cmd = parse_quick(clauses[0], self._app_index)
        if cmd is None or cmd.kind != "open_app" or cmd.value.lower() in EARLY_DENY_APPS:
            return None
        t0 = time.perf_counter()
        try:
            from . import approvals

            reopen_and_activate(cmd.value, foreground=approvals.current.windows)
        except Exception as e:
            self.log.add("code", "open app", f"{cmd.value}: {type(e).__name__}", None, (time.perf_counter() - t0) * 1000)
            return None
        self.log.add("code", "open app", f"opened {cmd.value} before the model", None, (time.perf_counter() - t0) * 1000)
        return cmd.value, len(clauses) == 1

    def _canned_turn(self, text: str, answer: str, turn_id: str | None) -> Iterator[Event]:
        """A turn answered by code alone (nothing left for a model to do)."""
        yield Event("token", answer)
        self.agent.update_state(self.config, {"messages": [HumanMessage(content=text, id=turn_id or uuid.uuid4().hex), AIMessage(content=answer)]})
        if self.meta is not None:
            self.meta.route = "fast"
        yield Event("final", answer)

    def _fast_turn(self, text: str, turn_id: str | None = None) -> Iterator[Event]:
        history = self._history()
        messages = [SystemMessage(content=FAST_SYSTEM), *history, HumanMessage(content=text)]
        out: list[str] = []
        first_ms = None
        total = 0.0
        try:
            with llm_lock():
                t0 = time.perf_counter()
                for chunk in self.fast_llm.stream(messages):
                    piece = _text(chunk.content)
                    if piece:
                        if first_ms is None:
                            first_ms = (time.perf_counter() - t0) * 1000
                        out.append(piece)
                        yield Event("token", piece)
                total = (time.perf_counter() - t0) * 1000
        finally:
            # Even if the turn is stopped or the model fails half way, the exchange goes into the thread: the person's message
            # stays on screen, and the executor sees it if a later turn escalates.
            answer = "".join(out).strip()
            self.agent.update_state(self.config, {"messages": [HumanMessage(content=text, id=turn_id or uuid.uuid4().hex), AIMessage(content=answer)]})
            if self.meta is not None:
                self.meta.first_token_ms = first_ms
        self.log.add("llm", "fast model", f"{config.FAST_MODEL}: first token {first_ms or 0:.0f} ms", None, total)
        yield Event("final", answer)

    def _executor_turn(self, text: str, i: intake.Intake, turn_id: str | None = None) -> Iterator[Event]:
        try:  # the previous request's todo list must not carry over: a stale list made the agent skip planning the new task
            if (self.agent.get_state(self.config).values or {}).get("todos"):
                self.agent.update_state(self.config, {"todos": []})
        except Exception as e:
            self.log.add("code", "error", f"clearing old todos: {type(e).__name__}"[:90], None, 0.0)
        state_in = {"messages": [HumanMessage(content=text, id=turn_id or uuid.uuid4().hex)], "laya": intake.agent_hint(i)}
        rubric = agents.rubric_for(i.intent)
        if rubric:
            state_in["rubric"] = rubric
        yield from self._stream(state_in)

    def approve_all(self) -> list[dict]:
        return [{"type": "approve"} for _ in (self.pending or {}).get("action_requests", [])]

    def answer_pending(self, text: str) -> bool:
        """Use the chat box to answer an approval. Returns False if nothing is pending."""
        if not self.pending:
            return False
        n = len(self.pending.get("action_requests", []))
        return self.submit_resume(decisions_from_text(text, n))

    def reject_all(self) -> list[dict]:
        return [{"type": "reject"} for _ in (self.pending or {}).get("action_requests", [])]

    def resume(self, decisions: list[dict]) -> Iterator[Event]:
        self.pending = None
        yield from self._stream(Command(resume={"decisions": decisions}))

    def _stream(self, first_input) -> Iterator[Event]:
        cfg = {**self.config, "recursion_limit": config.STEP_BUDGET}
        try:
            for ns, mode, data in self.agent.stream(first_input, config=cfg, stream_mode=["messages", "updates"], subgraphs=True):
                yield from self._translate(ns, mode, data)
                if self.pending is not None:
                    yield Event("interrupt", self.pending)
                    return
        except GraphRecursionError:
            self.log.add("code", "step budget", f"stopped after {config.STEP_BUDGET} graph steps", None, 0.0)
            yield Event("error", f"Stopped: step budget reached ({config.STEP_BUDGET} steps).")
            return
        msgs = self.agent.get_state(self.config).values.get("messages", [])
        last = next((m for m in reversed(msgs) if m.type == "ai" and _text(m.content).strip()), None)
        yield Event("final", _text(last.content) if last else "")

    def _translate(self, ns, mode, data) -> Iterator[Event]:
        sub = ns[0].split(":")[0] if ns else None
        if mode == "messages":
            chunk, meta = data
            piece = _text(chunk.content)
            if piece and not sub and getattr(chunk, "type", "") in ("AIMessageChunk", "ai") and not getattr(chunk, "tool_call_chunks", None):
                yield Event("token", piece)
            return
        # mode == "updates"
        if "__interrupt__" in data:
            self.pending = data["__interrupt__"][0].value
            return
        for node, upd in data.items():
            msgs = (upd or {}).get("messages", []) if isinstance(upd, dict) else []
            for m in msgs if isinstance(msgs, list) else []:
                if m.type == "tool":
                    if getattr(m, "tool_call_id", None) in self._plan_calls:
                        continue  # the answer to a write_todos call (including the guard's "never call it in parallel" error, which has no name)
                    yield Event("subagent" if sub else "tool", {"name": getattr(m, "name", "") or "tool", "sub": sub, "id": getattr(m, "tool_call_id", None),
                                                                "preview": _text(m.content)[:160]})
                elif m.type == "ai" and getattr(m, "tool_calls", None):
                    for c in m.tool_calls:
                        if c["name"] == "write_todos":
                            self._plan_calls.add(c.get("id"))
                            yield Event("plan", c["args"].get("todos", []))
                        else:
                            yield Event("call", {"name": c["name"], "args": c["args"], "id": c.get("id"), "sub": sub})
