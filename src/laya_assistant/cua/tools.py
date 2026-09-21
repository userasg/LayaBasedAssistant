"""The agent-facing tools. `computer_confirm` is gated by human-in-the-loop (interrupt_on), so the user's approval is
enforced by the framework *before* it runs: the loop is never re-executed by a resume."""
from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool


def make_tools(cu):
    @tool
    def computer_use(target: Literal["browser", "my_chrome", "desktop"], goal: str, max_steps: int = 12, app: str = "") -> str:
        """Click, type and read in a website or app to reach a goal. target='browser' (the default choice) is a browser window this app opens
        and owns: it works immediately, keeps its own logins, and starts on a blank page, so the goal should say where to go. target='my_chrome'
        is the user's own Chrome (needs setup). target='desktop' drives a Mac app (pass its name in `app`, e.g. app='Notes').
        Prefer mac_run / mac_notes_create when a script can do the job; use this when something must be clicked or typed on a page. Give the whole goal in plain words ("search for sony headphones and open the first result").
        It plans, acts and checks fast, and returns DONE, NEEDS_USER (a password, captcha or code: tell the user to do it, then call again),
        NEEDS_APPROVAL (irreversible step: ask the user, then call computer_confirm or computer_cancel), BLOCKED or FAILED."""
        try:
            return cu.run(target, goal, max_steps, app)
        except RuntimeError as e:
            return f"UNAVAILABLE: {e}"

    @tool
    def computer_confirm(pending_id: str) -> str:
        """Perform the irreversible step the user has just approved (the pending_id comes from NEEDS_APPROVAL) and continue the goal."""
        try:
            return cu.confirm(pending_id)
        except RuntimeError as e:
            return f"UNAVAILABLE: {e}"

    @tool
    def computer_cancel(pending_id: str) -> str:
        """Drop a pending step the user declined."""
        return cu.cancel(pending_id)

    return [computer_use, computer_confirm, computer_cancel]
