"""The LLM's job in computer use: turn a goal into a short list of steps in words. Called once per goal, and again only
when the fast loop hands a step back. It never picks elements: it says what it wants ("type 'sony' into the search box")
and the application's menu + Laya + verification take it from there."""
from __future__ import annotations

import json
import re

from ..engine import llm_lock, make_llm
from .. import config
from .types import DESKTOP_KINDS, DO_KINDS, Observation, StepPlan

PROMPT = """You plan steps for an agent that operates {surface}. Output ONLY a JSON array of at most {max_steps} steps, nothing else.
Each step is an object: {{"do": <one of {kinds}>, "target": "<a few words describing the element, as a person would say it>", "value": "<text to type, option to choose, or a full URL>"}}.
Rules:
- "type" needs a value; "select" needs the option as value; {navigate_rule}
- "check" ticks a checkbox or radio; "click" presses a button, link, list row or tab; "press_enter" submits a text field.
- NEVER type passwords, card numbers, codes or any secret: leave those steps out, the user will do them.
{surface_rules}
- Do exactly what the goal asks and nothing more. The step that achieves the goal is the last one before "done": for "search for X" that is
  submitting the search; do NOT open results, press extra buttons or search again unless the goal says so. Never repeat a step listed under
  "Already done".
- Plan ONLY the steps you can carry out on the CURRENT screen: each one must be about a control you can see in the list below. If the goal needs more
  steps once the screen changes (search results appear, a conversation opens, a dialog or a new page loads), end with {{"do": "more"}}: you will be
  shown the new screen and asked again. Use "more" ONLY when a step the goal asks for cannot be named yet because its control is not on screen; if the
  last thing the goal asks for is in your list, end with {{"do": "done"}} (also when the goal is already achieved). Never use "more" just to look at the result.
- Keep steps small and literal.

Goal: {goal}
{history}{failure}
Current screen:
{page}
JSON:"""

BROWSER_RULES = """- "navigate" is only for a goal that names another site or URL, or a blank page (about:blank): then start with a navigate step to the right
  page; for a search use https://duckduckgo.com/?q=<url-encoded query>. Stay on the current site otherwise; never invent a site."""
DESKTOP_RULES = """- The app is already open and in front. Work only inside its window with what it shows. NEVER use "navigate", never search the web, never
  use another app: the contact, file, song or item you need is found with the app's own search box, sidebar or list.
- The window title names what is open right now (the conversation, document or item). If that is already what the goal needs, do NOT search for it,
  compose a new one or open it again: go straight to the step that uses it. If the goal names a person or item that is not the one open, find it
  with the app's own search box or list by its exact name, never by a different name.
- Type the user's words EXACTLY as given: never reword, extend or add a greeting to them.
- To open an item from a list or search results, click its row; to send or submit text, type it into the input box, then press_enter.
- Example (a chat app; goal: send "on my way" to Sam; the open conversation is Lee, so Sam must be found): the screen has a search box, so plan
  [{{"do":"type","target":"search box","value":"Sam"}},{{"do":"more"}}]; once the list shows a row "Sam", plan
  [{{"do":"click","target":"Sam row"}},{{"do":"type","target":"message box","value":"on my way"}},{{"do":"press_enter","target":"message box"}},{{"do":"done"}}].
  If the open conversation had been Sam already, only the last three steps."""


HISTORY_STEPS = 6  # finished steps the planner is shown


def parse_plan(text: str, max_steps: int = 8, allowed: tuple = DO_KINDS) -> list[StepPlan]:
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
        if do not in allowed:
            raise ValueError(f"step kind {do!r} is not available on this surface (use only: {', '.join(k for k in allowed if k not in ('done', 'more'))})")
        val = item.get("value")
        steps.append(StepPlan(do, str(item.get("target", "") or "").strip(), None if val in (None, "") else str(val)))
    if not steps:
        raise ValueError("empty plan")
    return steps


def make_planner(llm=None, surface: str = "a web browser", desktop: bool = False):
    """A planner function: (goal, observation, done_steps, failure) -> list[StepPlan]. `desktop`: a Mac app, which has no `navigate`."""
    model = llm or make_llm(config.EXECUTOR_MODEL, temperature=0.1)
    kinds = DESKTOP_KINDS if desktop else DO_KINDS

    def plan(goal: str, obs: Observation, done: list[str] | None = None, failure: str | None = None, max_steps: int = 8) -> list[StepPlan]:
        earlier = f"({len(done) - HISTORY_STEPS} earlier steps are also done) " if done and len(done) > HISTORY_STEPS else ""
        history = ("Already done: " + earlier + "; ".join(done[-HISTORY_STEPS:]) + "\n") if done else ""  # the last few, never the whole run: prompt size stays flat
        fail = f"The last plan failed because: {failure}. Plan again from the current page.\n" if failure else ""
        navigate_rule = '"open_app" launches another Mac app (value = its name, e.g. "Notes").' if desktop else '"navigate" needs a full https:// URL as value and no target; "open_app" launches a Mac app (value = its name).'
        prompt = PROMPT.format(surface=surface, kinds="|".join(kinds), max_steps=max_steps, goal=goal, history=history, failure=fail, page=obs.summary(30),
                               navigate_rule=navigate_rule, surface_rules=DESKTOP_RULES if desktop else BROWSER_RULES)
        last = None
        for _ in range(3):  # retries on malformed output, each told what was wrong
            with llm_lock():
                text = str(model.invoke(prompt).content)
            try:
                return parse_plan(text, max_steps, kinds)
            except (ValueError, json.JSONDecodeError) as e:
                last = e
                prompt += f"\nYour last reply was not usable ({str(e)[:120]}). Output ONLY a JSON array of objects with double-quoted keys and strings."
        raise ValueError(f"the planner did not return a usable plan: {last}")

    return plan
