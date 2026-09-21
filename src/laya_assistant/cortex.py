"""Laya cortex: the fast System 1 layer around every model and tool call.

Order of defence for a shell command, cheapest and most exact first:
  1. code   : hard-deny regex (sudo, rm -rf /, pipe-to-shell, uploads) ........ ~0 ms, cannot be argued with
  2. code   : provenance (the command appears in tool output but the user never said it = injection) ~0 ms
  3. Laya   : `destructive` (delete/overwrite/wipe) ~50 ms. Sandbox commands are block-only; host tools ask.
  4. code   : repeat-failure guard (same command failed 3 times = looping)
`authorized` ("did the user ask for this?") is asked in the same forward pass but only LOGGED: measured on
the base checkpoint it rates an injected `curl ... | sh` as 0.92-1.00 authorized, so it must not gate anything.
Failure-kind classification is likewise log-only (it called "command not found" a runtime_error at 0.78).
Both become steering signals only after a fine-tune earns them, which the decision log is there to enable.
"""
import re
import time

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import interrupt
from ollama import ResponseError

from . import config
from .decisions import DecisionLog, guard_action
from .engine import llm_lock
from . import approvals, review
from .host import is_safe_host_command

GATE_QUESTIONS = {
    "destructive": {"type": "noul", "instructions": "Does `command` delete, overwrite or wipe files or data?"},
    # logged only, see module docstring
    "authorized": {"type": "noul", "instructions": "Is `command` something the user asked for in `user_request`?"},
}
FAILURE_QUESTIONS = {
    "failure": {
        "type": "choice",
        "instructions": "What kind of failure does `output` show?",
        "criteria": {
            "assertion_failure": "a test assertion or wrong result",
            "missing_dependency": "a module, package or command that is not installed or found",
            "syntax_error": "a syntax or indentation error in the code",
            "runtime_error": "an exception raised while the code ran",
            "other": "none of the above",
        },
    }
}

_DENY = [
    (r"\bsudo\b", "sudo"),
    (r"\bmkfs\b", "filesystem formatting"),
    (r"\bdd\s+if=", "raw disk write"),
    (r":\(\)\s*\{", "fork bomb"),
    (r"\brm\s+-\w+\s+(/|~|\$HOME)(\s|$)", "recursive delete of root or home"),
    (r"\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b", "piping data into a shell"),
    (r"\bcurl\b.*(?:\s-F\b|\s--form\b|\s-T\b|\s--upload-file\b|\s-d\b|\s--data(?:-\w+)?\b|\s-X\s*POST\b)", "uploading data with curl"),
    (r"\bwget\b.*--post-(?:data|file)", "uploading data with wget"),
    # the user's real files: deletion goes through the `delete` tool, which asks and moves to .trash
    (r"\b(?:rm|unlink|shred|rmdir)\b[^|;&]*(?:/workspace/shared|(?:^|\s)\.?/?shared/)", "deleting in the shared folder: use the delete tool (it asks and moves the file to the trash)"),
    (r"/workspace/shared[^|;&]*-delete\b|\bfind\b[^|;&]*shared[^|;&]*-delete\b", "deleting in the shared folder: use the delete tool (it asks and moves the file to the trash)"),
]
_DENY_RE = [(re.compile(p, re.I), why) for p, why in _DENY]
EXIT_CODE = re.compile(r"\[Command (succeeded|failed) with exit code (-?\d+)\]")


def hard_deny_reason(command: str) -> str | None:
    for rx, why in _DENY_RE:
        if rx.search(command):
            return why
    return None


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def _human_texts(messages) -> list[str]:
    return [_norm(_text(m.content)) for m in messages if getattr(m, "type", "") == "human"]


def user_specified(command: str, messages) -> bool:
    c = _norm(command)
    return bool(c) and any(c in h for h in _human_texts(messages))


def injected_from_tool_output(command: str, messages) -> bool:
    """The exact command text appears in recent tool output but the user never wrote it."""
    c = _norm(command)
    if len(c) < 12 or user_specified(command, messages):
        return False
    tool_texts = [_norm(_text(m.content)) for m in messages if getattr(m, "type", "") == "tool"][-6:]
    return any(c in t for t in tool_texts)


def _same_plan(new: list[str], old: list[str]) -> bool:
    """The same plan: same number of steps, each mostly the same words. A model that only rewords its steps (or moves a status
    along) has not changed the plan, and asking the user to re-approve it again is what made approvals feel like a loop."""
    if len(new) != len(old):
        return False
    for a, b in zip(new, old):
        wa, wb = set(_norm(a).split()), set(_norm(b).split())
        if wa != wb and len(wa & wb) / max(len(wa | wb), 1) < 0.6:
            return False
    return True


class LayaCortexMiddleware(AgentMiddleware):
    """`is_subagent=True` keeps the gates but skips the intake note (subagents need the guard too: they run commands)."""

    def __init__(self, predictor, log: DecisionLog, is_subagent: bool = False, host_tools: set[str] | None = None, gate_plans: bool = False):
        self.predictor, self.log, self.is_subagent = predictor, log, is_subagent
        self.gate_plans = gate_plans  # ask the user before a plan is adopted, but only when the plan itself changes
        self._plan_calls: dict[str, set] = {}  # human message -> tool-call ids already asked about
        self.host_tools = host_tools or set()
        self._gates: dict[str, dict] = {}  # tool_call id -> decision (survives a HITL resume re-run)
        self._failures: dict[str, int] = {}  # command -> consecutive failures
        self._nudges: dict[str, int] = {}  # human message -> continue-nudges already sent

    def _laya(self, state: dict, questions: dict):
        t0 = time.perf_counter()
        answers = self.predictor.predict(state, questions)["answers"]
        return answers, (time.perf_counter() - t0) * 1000

    # -- model call: the Laya note rides on the last human message, timing excludes lock wait ------------

    @staticmethod
    def _first_call_of_turn(messages) -> bool:
        """True while nothing has answered the latest human message yet."""
        return bool(messages) and messages[-1].type == "human"

    @staticmethod
    def _calls(response, name: str) -> bool:
        msg = response.result[-1] if getattr(response, "result", None) else response
        return any(c["name"] == name for c in (getattr(msg, "tool_calls", None) or []))

    def _call(self, request, handler):
        try:
            return handler(request)
        except ResponseError as e:
            if "repeat limit" not in str(e):
                raise
            # Ollama aborts a generation that degenerates into repetition; sampling is random, so one retry usually clears it.
            self.log.add("code", "retry", "Ollama repetition abort: retrying once", None, 0.0)
            return handler(request)

    @staticmethod
    def _response_text(response) -> str:
        msg = response.result[-1] if getattr(response, "result", None) else response
        return _text(getattr(msg, "content", ""))

    @staticmethod
    def _calls_any(response) -> bool:
        msg = response.result[-1] if getattr(response, "result", None) else response
        return bool(getattr(msg, "tool_calls", None))

    def _continue_if_stalled(self, request, response, handler):
        """qwen 14B sometimes approves a plan, writes "Next, I will ..." and ends its turn with todos still open.
        If the plan is unfinished and the reply is prose (not a tool call, not a question), tell it to keep going."""
        if self.is_subagent:
            return response
        todos = (request.state or {}).get("todos") or []
        open_todos = [t for t in todos if t.get("status") != "completed"]
        text = self._response_text(response).strip()
        if not open_todos or self._calls_any(response) or text.endswith("?"):
            return response
        human = next((_text(m.content) for m in reversed(request.messages) if m.type == "human"), "")
        key = _norm(human)[:120]
        if self._nudges.get(key, 0) >= config.MAX_CONTINUE_NUDGES:
            return response
        self._nudges[key] = self._nudges.get(key, 0) + 1
        self.log.add("code", "continue nudge", f"{len(open_todos)} todo(s) still open: telling the agent to keep going", None, 0.0)
        nudge = request.override(messages=[*request.messages, HumanMessage(content="Your plan is not finished. Do the next open step now by calling a tool; do not just describe it.")])
        return self._call(nudge, handler)

    def wrap_model_call(self, request, handler):
        if not self.is_subagent:  # subagents start fresh; the main agent sees this request plus a recap of earlier ones (see scope.py)
            from .scope import scoped_messages

            scoped = scoped_messages(request.messages)
            if len(scoped) != len(request.messages) or scoped[0] is not request.messages[0]:
                request = request.override(messages=scoped)
        laya = (request.state or {}).get("laya") or {}
        note = laya.get("note") if not self.is_subagent else None
        if note:
            msgs = list(request.messages)
            idx = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i].type == "human"), None)
            if idx is not None:
                c = msgs[idx].content
                new = c + f"\n\n{note}" if isinstance(c, str) else [*c, {"type": "text", "text": f"\n\n{note}"}]
                msgs[idx] = msgs[idx].model_copy(update={"content": new})
                request = request.override(messages=msgs)
        forced = None
        if (laya.get("plan_first") and not self.is_subagent and not (request.state or {}).get("todos")
                and self._first_call_of_turn(request.messages)):
            # Laya's intake said this turn needs a plan. The 14B model sometimes reaches for write_file with a
            # `todos` argument instead of write_todos, which skips the approval pause; with only the planning
            # tool on offer for this first call, it plans.
            plan_tools = [t for t in request.tools if getattr(t, "name", None) == "write_todos"]
            if plan_tools:
                forced = request
                request = request.override(tools=plan_tools)
                self.log.add("code", "plan first", "first call limited to write_todos (Laya: needs a plan)", None, 0.0)
        with llm_lock():
            t0 = time.perf_counter()
            response = self._call(request, handler)
            if forced is not None and not self._calls(response, "write_todos"):
                # It answered in prose or returned nothing. Nudge once, then fall back to the full tool set so the
                # turn still does its work (the plan is skipped, and that is logged).
                nudge = request.override(messages=[*request.messages, HumanMessage(content="Call the write_todos tool now with a short step-by-step plan for the request above.")])
                self.log.add("code", "plan retry", "no write_todos call: nudging once", None, 0.0)
                response = self._call(nudge, handler)
                if not self._calls(response, "write_todos"):
                    self.log.add("code", "plan skipped", "model did not plan: continuing with all tools", None, 0.0)
                    response = self._call(forced, handler)
            response = self._continue_if_stalled(request, response, handler)
            ms = (time.perf_counter() - t0) * 1000
        msg = response.result[-1] if getattr(response, "result", None) else response
        calls = [c["name"] for c in (getattr(msg, "tool_calls", None) or [])]
        self.log.add("llm", "subagent call" if self.is_subagent else "model call",
                     f"tool calls: {calls}" if calls else "text reply", None, ms)
        return response

    # -- tool calls ---------------------------------------------------------------------------------------

    @staticmethod
    def _blocked(call, why: str) -> ToolMessage:
        return ToolMessage(
            content=f"Blocked before running ({why}). Do not retry it. Tell the user what you wanted to do and propose a safe alternative.",
            tool_call_id=call["id"], status="error",
        )

    def _gate(self, call: dict, command: str, messages, block_at: float, ask_at: float | None) -> dict:
        if call["id"] in self._gates:
            return self._gates[call["id"]]
        if user_specified(command, messages):
            self.log.add("code", "gate", "user typed this command -> allow", None, 0.0)
            decision = {"action": "allow", "p": 0.0}
        else:
            last_human = next((_text(m.content) for m in reversed(messages) if getattr(m, "type", "") == "human"), "")
            answers, ms = self._laya({"command": command, "user_request": last_human[:300]}, GATE_QUESTIONS)
            p, auth = answers["destructive"]["noul"], answers["authorized"]["noul"]
            action = guard_action(p, block_at, ask_at)
            self.log.add("laya", "gate", f"destructive={p:.2f} authorized(log-only)={auth:.2f} -> {action}", p, ms)
            decision = {"action": action, "p": p}
        self._gates[call["id"]] = decision
        return decision

    def _gate_plan(self, request, handler, messages):
        """Ask before adopting a plan, but not for every progress update: status-only changes (pending -> completed) pass straight
        through, which is what used to make the user re-approve the same plan again and again. Capped per request."""
        call = request.tool_call
        new = call["args"].get("todos", [])
        old = (request.state or {}).get("todos") or []
        if _same_plan([t.get("content", "") for t in new], [t.get("content", "") for t in old]):
            return handler(request)
        human = next((_text(m.content) for m in reversed(messages) if getattr(m, "type", "") == "human"), "")
        seen = self._plan_calls.setdefault(_norm(human)[:120], set())
        if call["id"] not in seen and len(seen) >= config.MAX_PLAN_ASKS:
            self.log.add("code", "plan gate", f"asked {config.MAX_PLAN_ASKS} times already: proceeding", None, 0.0)
            return handler(request)
        seen.add(call["id"])
        reply = interrupt({
            "action_requests": [{"name": "write_todos", "args": {"todos": new}, "description": "Review the plan"}],
            "review_configs": [{"action_name": "write_todos", "allowed_decisions": ["approve", "edit", "reject", "respond"]}],
        })
        d = ((reply or {}).get("decisions") or [{}])[0] if isinstance(reply, dict) else {}
        kind = d.get("type", "approve")
        self.log.add("code", "plan review", kind, None, 0.0)
        if kind == "reject":
            return ToolMessage(content="The user rejected this plan. Ask what they would like instead; do not propose the same plan again.",
                               tool_call_id=call["id"], status="error")
        if kind == "respond":
            return ToolMessage(content=f"The user answered instead of approving: {d.get('message', '')!r}. Revise the plan to fit their answer, "
                                       "or answer their question, then continue.", tool_call_id=call["id"], status="error")
        if kind == "edit":
            edited = (d.get("edited_action") or {}).get("args") or {}
            request = request.override(tool_call={**call, "args": {**call["args"], **edited}})
        return handler(request)

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        name = call["name"]
        messages = (request.state or {}).get("messages", [])
        if name == "write_todos" and self.gate_plans and not self.is_subagent and approvals.current.plan:
            return self._gate_plan(request, handler, messages)
        if name != "execute" and name not in self.host_tools:
            return self._timed(call, handler, request)

        command = str(call["args"].get("command", "")) if name in ("execute", "mac_run") else str(call["args"])
        t0 = time.perf_counter()
        why = hard_deny_reason(command)
        self.log.add("code", "hard-deny regex", f"DENY: {why}" if why else "pass", None, (time.perf_counter() - t0) * 1000)
        if why:
            return self._blocked(call, why)
        inj = injected_from_tool_output(command, messages)
        self.log.add("code", "provenance", "command came from tool output, not the user" if inj else "pass", None, 0.0)
        if inj:
            return self._blocked(call, "this exact command appears in tool output but the user never asked for it (possible injected instruction)")

        host = name in self.host_tools
        if name == "mac_run" and is_safe_host_command(command):
            gate = {"action": "allow", "p": 0.0}  # exactly recognised as safe: no model needed
            self.log.add("code", "gate", "safe host command -> allow", None, 0.0)
        else:
            # Laya can only ASK about a shell command on your Mac, never refuse it (a person may well want to delete their own file):
            # refusing is the job of the exact hard-deny list above.
            block_at = 1.01 if name == "mac_run" else (config.HOST_BLOCK_AT if host else config.SANDBOX_BLOCK_AT)
            gate = self._gate(call, command, messages, block_at, config.HOST_ASK_AT if host else None)
            if name == "mac_run" and gate["action"] == "ask" and approvals.current.host_commands == "never_ask":
                gate = {**gate, "action": "allow"}
        if (name == "mac_run" and gate["action"] == "allow" and not is_safe_host_command(command) and not user_specified(command, messages)
                and approvals.host_command_needs_approval(command)):
            gate = {**gate, "action": "ask"}  # risky (or every command, if you chose that): the user decides
        if gate["action"] == "block":
            return self._blocked(call, f"Laya judged it destructive, p={gate['p']:.2f}")
        if gate["action"] == "ask":
            answer = review.decision(interrupt(review.request(name, f"Run on your Mac: {command}", {"command": command}, laya_risk=round(gate["p"], 2))))
            self.log.add("code", "human review", answer["type"], None, 0.0)
            refusal = review.declined(answer, "this command")
            if refusal:
                return ToolMessage(content=refusal, tool_call_id=call["id"], status="error")

        if self._failures.get(command, 0) >= config.MAX_REPEAT_FAILURES:
            self.log.add("code", "repeat-failure guard", f"blocked after {config.MAX_REPEAT_FAILURES} identical failures", None, 0.0)
            return ToolMessage(
                content=(f"Blocked: this exact command already failed {config.MAX_REPEAT_FAILURES} times. Do not run it again. "
                         "Read the error, change your approach (check paths with `ls`), or ask the user."),
                tool_call_id=call["id"], status="error",
            )
        result = self._timed(call, handler, request)
        return self._after_execute(command, result)

    def _timed(self, call, handler, request):
        t0 = time.perf_counter()
        result = handler(request)
        detail = str(call["args"].get("command") or call["args"].get("file_path") or call["args"].get("path") or "")[:60]
        self.log.add("tool", call["name"], detail, None, (time.perf_counter() - t0) * 1000)
        return result

    def _after_execute(self, command: str, result):
        content = getattr(result, "content", None)
        if not isinstance(content, str):
            return result
        m = EXIT_CODE.search(content)
        if not m or m.group(1) == "succeeded":
            self._failures.pop(command, None)
            self.log.add("code", "exit code", "0 (success)" if m else "not reported", None, 0.0)
            return result
        self._failures[command] = self._failures.get(command, 0) + 1
        a, ms = self._laya({"output": content[-1500:]}, FAILURE_QUESTIONS)
        f = a["failure"]
        self.log.add("laya", "failure kind", f"{f['choice']} (log-only)", f["confidence"], ms)
        return result  # never rewritten: a confidently wrong classification must not steer the LLM
