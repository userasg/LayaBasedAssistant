"""Per-run agent state and context.

`AssistantState` extends Deep Agents' state with `laya`: the intake decision for the current turn
(intent, route, probabilities, the note the cortex appends), so tools, middleware and the UI can read
Laya's probabilities from agent state, the way LangChain's Jev router keeps them there.
`AssistantContext` carries per-run configuration to tools without polluting the prompt.
"""
from dataclasses import dataclass
from typing import NotRequired

from deepagents import DeepAgentState


class AssistantState(DeepAgentState):
    laya: NotRequired[dict]


@dataclass
class AssistantContext:
    session_id: str = ""
    workspace_label: str = ""
