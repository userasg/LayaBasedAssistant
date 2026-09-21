"""The LLM's job in computer use: turn a goal into a short list of steps in words. Called once per goal, and again only
when the fast loop hands a step back. It never picks elements: it says what it wants ("type 'sony' into the search box")
and the application's menu + Laya + verification take it from there."""
from __future__ import annotations

import json
import re

from ..engine import llm_lock, make_llm
from .. import config
from .types import DO_KINDS, Observation, StepPlan

PROMPT = """You plan steps for an agent that operates {surface}. Output ONLY a JSON array of at most {max_steps} steps, nothing else.
Each step is an object: {{"do": <one of {kinds}>, "target": "<a few words describing the element, as a person would say it>", "value": "<text to type, option to choose, or a full URL>"}}.
Rules:
- "type" needs a value; "select" needs the option as value; "navigate" needs a full https:// URL as value and no target.
- "open_app" launches a Mac app (value = the app name, e.g. "Notes"); "check" ticks a checkbox or radio; "click" presses a button or link; "press_enter" submits a text field; use "done" as the last step.
- NEVER type passwords, card numbers, codes or any secret: leave those steps out, the user will do them.
- Stay on the current site. Use "navigate" only if the goal names another site or URL, or the page is blank (about:blank): then start with a
  navigate step to the right page; for a search use https://duckduckgo.com/?q=<url-encoded query>. Never invent a site otherwise.
- Do exactly what the goal asks and nothing more. The step that achieves the goal is the last one before "done": for "search for X" that is
  submitting the search; do NOT open results, press extra buttons or search again unless the goal says so.
- Plan only what is possible from the current page and the pages it leads to. Keep steps small and literal.

Goal: {goal}
{history}{failure}
Current page:
{page}
JSON:"""


def parse_plan(text: str, max_steps: int = 8) -> list[StepPlan]:
    """Tolerant JSON extraction: the model may wrap the array in prose or code fences."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON array in the planner reply")
    blob = m.group(0)
    try:
        raw = json.loads(blob)
    except json.JSONDecodeError:
        # the usual small-model slips: trailing commas, comments, single quotes, smart quotes, Python literals
        fixed = re.sub(r"//[^\n]*", "", blob)
        fixed = fixed.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        fixed = re.sub(r",\s*([\]}])", r"\1", fixed)
        fixed = re.sub(r"\bNone\b", "null", re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", fixed)))
        try:
            raw = json.loads(fixed)
        except json.JSONDecodeError:
            import ast

            try:
                raw = ast.literal_eval(fixed)  # single-quoted dicts
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"the planner's JSON could not be repaired: {e}") from e
    steps = []
    for item in raw[:max_steps]:
        do = str(item.get("do", "")).strip().lower().replace(" ", "_")
        if do == "press_enter" or do == "submit":
            do = "press_enter"
        if do not in DO_KINDS:
            raise ValueError(f"unknown step kind {do!r}")
        val = item.get("value")
        steps.append(StepPlan(do, str(item.get("target", "") or "").strip(), None if val in (None, "") else str(val)))
    if not steps:
        raise ValueError("empty plan")
    return steps


def make_planner(llm=None, surface: str = "a web browser"):
    """A planner function: (goal, observation, done_steps, failure) -> list[StepPlan]."""
    model = llm or make_llm(config.EXECUTOR_MODEL, temperature=0.1)

    def plan(goal: str, obs: Observation, done: list[str] | None = None, failure: str | None = None, max_steps: int = 8) -> list[StepPlan]:
        history = ("Already done: " + "; ".join(done) + "\n") if done else ""
        fail = f"The last plan failed because: {failure}. Plan again from the current page.\n" if failure else ""
        prompt = PROMPT.format(surface=surface, kinds="|".join(DO_KINDS), max_steps=max_steps, goal=goal, history=history, failure=fail, page=obs.summary(30))
        last = None
        for _ in range(3):  # retries on malformed output, each told what was wrong
            with llm_lock():
                text = str(model.invoke(prompt).content)
            try:
                return parse_plan(text, max_steps)
            except (ValueError, json.JSONDecodeError) as e:
                last = e
                prompt += f"\nYour last reply was not usable ({str(e)[:120]}). Output ONLY a JSON array of objects with double-quoted keys and strings."
        raise ValueError(f"the planner did not return a usable plan: {last}")

    return plan
