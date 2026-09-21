"""`laya-advisor`: a CompiledSubAgent whose only node is a Laya forward pass.

The executor can call it through the normal `task` tool to get "which specialist fits this?" in ~50 ms
instead of spending an LLM turn deciding. It answers with the choice, its confidence and the full
probability vector, so the executor LLM can weigh it (and ignore it when confidence is low): the base
checkpoint is weak at open choices, so this is advice, never an order.
"""
import json

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from deepagents import CompiledSubAgent

from .decisions import DecisionLog
from .engine import ENGINE  # noqa: F401  (Laya predict already takes ENGINE via LockedLaya)

QUESTIONS = {
    "subagent": {
        "type": "choice",
        "instructions": "Which specialist should handle the task in `task`?",
        "criteria": {
            "researcher": "search the web, look things up, compare sources or summarise findings",
            "coder": "write, run, fix or test code and work with files in the sandbox",
            "reviewer": "check, verify or critique work that already exists, without changing it",
            "none": "no delegation needed: a greeting, a simple question, or something to answer directly",
        },
    }
}


def _text(content) -> str:
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content if isinstance(b, dict))


def build_advisor(predictor, log: DecisionLog) -> CompiledSubAgent:
    import time

    def advise(state: MessagesState):
        task = next((_text(m.content) for m in reversed(state["messages"]) if m.type == "human"), "")
        t0 = time.perf_counter()
        a = predictor.predict({"task": task[:400]}, QUESTIONS)["answers"]["subagent"]
        ms = (time.perf_counter() - t0) * 1000
        log.add("laya", "advisor", f"{a['choice']}", a["confidence"], ms)
        reply = {"subagent": a["choice"], "confidence": round(a["confidence"], 3), "probabilities": a["probabilities"]}
        return {"messages": [AIMessage(content=json.dumps(reply))]}

    graph = StateGraph(MessagesState)
    graph.add_node("advise", advise)
    graph.add_edge(START, "advise")
    graph.add_edge("advise", END)
    return {
        "name": "laya-advisor",
        "description": (
            "Instant (~50 ms) advice on which specialist subagent (researcher, coder, reviewer, or none) fits a task. "
            "Send the task text; it replies with JSON {subagent, confidence, probabilities}. Use it instead of deliberating; "
            "ignore it when confidence is below 0.5."
        ),
        "runnable": graph.compile(),
    }
