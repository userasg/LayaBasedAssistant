"""Laya intake: ONE forward pass per turn decides what the user wants and how to handle it.

Questions answered together (adding questions costs almost nothing, they share the pass):
  intent      choice  chat | question | write_code | research | computer_task | other
  needs_plan  noul    does this need several steps, tools, files, code or research?
  risky       noul    could carrying it out delete data, send messages, spend money, or change things outside a sandbox?
  simple      noul    is it a greeting or small talk?
  factual     noul    is it a trivia or general-knowledge question?

Routing: a turn goes to the small fast model only when Laya is confident AND it looks simple AND not
risky AND needs no plan. Anything else goes to the Deep Agent executor. A wrongly fast-routed hard
task is the costly error, so the rule is conservative.
"""
import re
import time
from dataclasses import dataclass

from . import config
from .decisions import DecisionLog, confidence_band

INTENT_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the user want in `request`?",
        "criteria": {
            "chat": "a greeting, thanks or small talk",
            "question": "a factual or conceptual question that can be answered from knowledge",
            "write_code": "write, fix, refactor or run code, or analyse data with code",
            "research": "look things up, compare sources or produce a written report",
            "computer_task": "operate a browser, an app or the computer, or change files and accounts",
            "other": "none of the above, or too vague to tell",
        },
    },
    "needs_plan": {
        "type": "noul",
        "instructions": "Does fulfilling `request` need several steps, tools, files, code or research?",
    },
    "risky": {
        "type": "noul",
        "instructions": "Could carrying out `request` delete data, send messages, spend money, or change things outside a sandbox?",
    },
    # Advice only: measured 12/20 overall, 8/11 when confident, so the LLM decides and this is a hint (see docs).
    "tool": {
        "type": "choice",
        "instructions": "Which tool should handle the request in `request`?",
        "criteria": {
            "browser": "a website or web app, for example Gmail, Amazon, a shop, a dashboard, booking or logging in online",
            "desktop": "a native Mac app, for example Finder, Notes, Calendar, Messages, Mail or System Settings",
            "files": "files and folders on disk: list, rename, move, read or edit them",
            "code": "programming: write, run, fix or test code",
            "research": "look up information online and summarise it, no site to operate",
            "none": "a greeting or a simple question that needs no tool",
        },
    },
    "simple": {
        "type": "noul",
        # Measured on the probe set: abstract wording ("answerable with no tools, files or research")
        # separated nothing (0/14 fast turns routed with zero wrong routes); naming the category does.
        "instructions": "Is `request` a greeting or small talk?",
    },
    "factual": {
        "type": "noul",
        "instructions": "Is `request` a trivia or general knowledge question?",
    },
}


@dataclass(frozen=True)
class Intake:
    intent: str
    needs_plan: float
    risky: float
    simple: float
    factual: float
    confidence: float
    band: str  # act | verify | defer, from the intent choice's confidence
    route: str  # "fast" (small model, no agent) or "executor" (Deep Agent)
    ms: float
    tool: str = ""  # Laya's suggestion (advice only)
    tool_conf: float = 0.0


def decide_route(band: str, simple: float, factual: float, risky: float, needs_plan: float, thresholds: dict | None = None) -> str:
    t = {"simple_at": config.SIMPLE_AT, "factual_at": config.FACTUAL_AT, "risky_at": config.RISKY_AT,
         "plan_at": config.PLAN_AT, **(thresholds or {})}
    easy = simple >= t["simple_at"] or factual >= t["factual_at"]
    fast = band != "defer" and easy and risky < t["risky_at"] and needs_plan < t["plan_at"]
    return "fast" if fast else "executor"


# Shapes that mean "do something", never small talk or trivia. Found in real use: "Plan first, then create a file greeting.md in
# /workspace/shared containing one friendly line" scored 0.63 on `simple` (the word "greeting") and went to the small model, which has
# no tools and just described the command instead of running it. Laya is for fuzzy calls; this is an exact veto that can only turn a
# fast route into an executor route, so the worst it costs is a slower (correct) answer, never a request answered without its tools.
_WORK = re.compile(
    r"~/|(?<![\w/])/(?:workspace|users|home|tmp|var|etc)\b"  # a path on disk
    r"|\b[\w-]+\.(?:md|txt|py|js|ts|json|csv|html|css|sh|ya?ml|toml|pdf|png|jpe?g|docx?|xlsx?)\b"  # a file name
    r"|https?://|\bwww\."
    r"|\bplan (?:it )?first\b"
    r"|\b(?:create|make|write|save|edit|fix|build|generate|run|execute|open|launch|delete|remove|move|rename|copy|download|upload|install)\b"
    r"[^.?!]{0,40}\b(?:files?|folders?|directory|scripts?|programs?|apps?|notes?|documents?|code|commands?|terminal|browser|project|repo)\b"
    r"|\bsearch (?:the )?(?:web|internet|online)\b|\blook up\b",
    re.I,
)


def looks_like_work(text: str) -> bool:
    return bool(_WORK.search(text))


def classify(predictor, text: str, thresholds: dict | None = None, log: DecisionLog | None = None) -> Intake:
    t0 = time.perf_counter()
    a = predictor.predict({"request": text}, INTENT_QUESTIONS)["answers"]
    ms = (time.perf_counter() - t0) * 1000
    band = confidence_band(a["intent"]["confidence"], config.INTAKE_ACT, config.INTAKE_VERIFY)
    p_simple, p_fact, p_risky, p_plan = a["simple"]["noul"], a["factual"]["noul"], a["risky"]["noul"], a["needs_plan"]["noul"]
    route = decide_route(band, p_simple, p_fact, p_risky, p_plan, thresholds)
    vetoed = route == "fast" and looks_like_work(text)
    intake = Intake(
        intent=a["intent"]["choice"],
        needs_plan=p_plan,
        risky=p_risky,
        simple=p_simple,
        factual=p_fact,
        confidence=a["intent"]["confidence"],
        band=band,
        route="executor" if vetoed else route,
        ms=ms,
        tool=a["tool"]["choice"],
        tool_conf=a["tool"]["confidence"],
    )
    if log is not None:
        log.add(
            "laya", "intake",
            f"{intake.intent} simple={p_simple:.2f} fact={p_fact:.2f} risky={p_risky:.2f} plan={p_plan:.2f} -> {intake.route}" + (" (work marker)" if vetoed else ""),
            intake.confidence, ms,
        )
    return intake


_WORKFLOW = {
    "chat": "Reply briefly and naturally.",
    "question": "Answer directly and concisely; no tools unless you are unsure.",
    "write_code": "Plan with write_todos, write code in the sandbox, run it or its tests to verify.",
    "research": "Plan with write_todos, delegate searching to `researcher`, then synthesise a short written result.",
    "computer_task": "Plan with write_todos; take the smallest safe steps and confirm anything irreversible.",
    "other": "Decide the best approach yourself.",
}


def intake_note(i: Intake) -> str:
    """Appended AFTER the stable system prompt (never edited into it) so Ollama's prompt-prefix cache stays valid."""
    head = f"[Laya note: intent `{i.intent}` (confidence {i.confidence:.2f}, {i.band}); risk {i.risky:.2f}.]"
    if i.intent == "computer_task" and i.tool in ("browser", "desktop") and i.tool_conf >= 0.6:
        head = head[:-1] + f" Laya suggests the `{i.tool}` target for computer_use ({i.tool_conf:.2f}); you decide.]"
    if i.band == "act":
        return f"{head} {_WORKFLOW[i.intent]}"
    if i.band == "verify":
        return f"{head} {_WORKFLOW[i.intent]} If the request is ambiguous, ask ONE clarifying question first."
    return f"{head} Low confidence, so the call is yours: if the request is too vague to act on, ask ONE clarifying question first."


_PLANNED_INTENTS = {"write_code", "research", "computer_task"}


def agent_hint(i: Intake) -> dict:
    """The `laya` entry for the agent's state: the note the cortex appends, plus what the cortex needs to
    act on Laya's decision (`plan_first` limits the first model call to the planning tool)."""
    return {
        "note": intake_note(i), "intent": i.intent, "route": i.route, "band": i.band,
        "plan_first": i.route == "executor" and (i.intent in _PLANNED_INTENTS or i.needs_plan >= config.PLAN_AT),
        "probabilities": {"needs_plan": round(i.needs_plan, 3), "risky": round(i.risky, 3)},
    }
