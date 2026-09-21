"""One shape for every question the agent puts to the person: a plan, a risky command, a file to delete, an irreversible step.

The chat's approval card and the text box both speak the Deep Agents human-in-the-loop protocol: a pause carries
`action_requests`, and the answer comes back as `{"decisions": [{"type": "approve" | "reject" | "respond" | "edit", ...}]}`.
The plan gate already used it, but the risky-command and delete gates asked with a flat payload and read the answer as a bare
string. In the real app that meant the card could not even draw them, and a typed "yes" arrived as a dict that was never equal to
"approve": every such request was silently treated as rejected. Both now go through `request()` and `decision()`.
"""
from __future__ import annotations

KINDS = ("approve", "reject", "respond", "edit")


def request(tool: str, description: str, args: dict, **extra) -> dict:
    """The payload to hand to `interrupt()`. `extra` rides along on the action (for example Laya's risk score)."""
    return {
        "action_requests": [{"name": tool, "args": args, "description": description, **extra}],
        "review_configs": [{"action_name": tool, "allowed_decisions": ["approve", "reject", "respond"]}],
    }


def decision(reply) -> dict:
    """The person's answer as one decision dict. Anything that is not clearly an approval is a rejection, never an accidental yes:
    an empty or malformed reply, an unknown type, a missing list."""
    if isinstance(reply, str):  # plain words, as older callers and tests pass
        return {"type": reply} if reply in KINDS else {"type": "reject"}
    if isinstance(reply, dict):
        decisions = reply.get("decisions") or []
        first = decisions[0] if decisions and isinstance(decisions[0], dict) else {}
        if first.get("type") in KINDS:
            return first
    return {"type": "reject"}


def declined(d: dict, what: str) -> str | None:
    """None if the person approved; otherwise the message the agent should receive instead of the tool running."""
    if d["type"] == "approve":
        return None
    if d["type"] == "respond":
        return (f"The user answered instead of approving {what}: {str(d.get('message', ''))!r}. Do not do it as asked; "
                "follow their answer (change the approach, or answer their question).")
    return f"Rejected by the user. Do not retry {what}; choose a safer approach or ask what they would like instead."
