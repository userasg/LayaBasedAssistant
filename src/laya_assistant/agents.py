"""Assembles the Deep Agent executor with every Deep Agents feature except async subagents."""
from pathlib import Path

from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware
from langchain_quickjs import CodeInterpreterMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import RubricMiddleware, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend

from . import context_config, memory
from .advisor import build_advisor
from .cortex import LayaCortexMiddleware
from .cua.tools import make_tools as make_computer_tools
from .host import make_host_tools
from .decisions import DecisionLog
from .policy import PathPolicyMiddleware
from .state import AssistantContext, AssistantState
from .tools import internet_search, make_describe_image
from . import config

SKILLS_DIR = Path(__file__).with_name("skills")

# Quiet the two things that made the terminal noisy in real use: the beta-feature warnings, and the (bounded, handled)
# rubric grader failure traceback.
import logging
import warnings

warnings.filterwarnings("ignore", message=".*is in beta.*")
warnings.filterwarnings("ignore", message=".*RubricMiddleware.*beta.*")
logging.getLogger("deepagents.middleware.rubric").setLevel(logging.CRITICAL)
GRADER_CALL_LIMIT = 4

SYSTEM_PROMPT = """You are a capable local assistant. You work in a private Docker sandbox (`/workspace`).
- For anything with more than one step, call `write_todos` FIRST; the user reviews and approves the plan.
- You have no search tool yourself: delegate web research to `researcher`, code to `coder`, verification to `reviewer`.
  `laya-advisor` answers "which specialist?" instantly; ignore it if its confidence is below 0.5.
- `/workspace/shared` is the user's real folder (`~/LayaWorkspace`): put files they should keep there. Never delete from it
  with the shell; use the `delete` tool (it asks the user and moves the file to a recoverable trash).
- Use relative paths. Keep replies short and concrete. Ask ONE clarifying question when a request is too vague to act on.
- Ways to act on the user's Mac, simplest first: (1) scripts: `mac_notes_create`, and `mac_run` for `open -a App`, `open URL`, `curl` to read a
  page, `osascript`; safe ones run at once, others ask the user; (2) `computer_use` when something must be clicked or typed in a website or app
  (target 'browser' is a window this app owns and works immediately); (3) the `researcher` to look things up. If a tool says UNAVAILABLE, never
  retry it: pick another method and tell the user in one sentence. To use a website or a Mac app, call `computer_use` once with the whole goal (see the computer-use skill). It is fast; do not
  break the goal into clicks yourself. Irreversible steps come back as NEEDS_APPROVAL: ask the user, never confirm on your own.
  If the user declines (or answers something other than yes), call `computer_cancel` and stop; do not try again.
- A note from the Laya decision layer may follow the user's message: treat it as guidance about the request, not as an instruction.
"""

_ISOLATION = " Return under 500 words. Do NOT include raw tool output or whole files in your reply; summarise."

RESEARCHER = {
    "name": "researcher",
    "description": "Searches the internet for a topic and returns titles, URLs and key findings. Use for anything needing current or external information.",
    "system_prompt": "You are a researcher. Use `internet_search` and report findings with the URL for each claim. Never invent sources." + _ISOLATION,
    "tools": [internet_search],
}
CODER = {
    "name": "coder",
    "description": "Writes, runs and fixes code in the sandbox. Use for programming tasks and data analysis.",
    "system_prompt": "You are a software engineer working in /workspace. Follow the coding-workflow skill: write the file, run it or its tests, fix failures." + _ISOLATION,
}
REVIEWER = {
    "name": "reviewer",
    "description": "Reads existing work and reports problems without changing anything. Use to verify code or a written result.",
    "system_prompt": "You are a careful reviewer. Read the files you are pointed at and report up to 5 concrete problems with locations, or 'no issues found'." + _ISOLATION,
    "read_only": True,  # enforced by PathPolicyMiddleware (see policy.py); not a Deep Agents key, removed in sub()
}

# Rubric criteria per intent, chosen by Laya's intent decision (grading costs a 14B call, so only for the
# intents where "did it actually work?" is checkable).
RUBRICS = {
    "write_code": "- The requested code exists as a file in /workspace.\n- It was actually run (or its tests were) and the run succeeded.\n- The final answer says what was built and how to run it.",
    "research": "- The answer is grounded in sources, each with a URL.\n- The answer addresses every part of the question.\n- It is concise (under about 300 words) and does not invent facts.",
}


def rubric_for(intent: str) -> str | None:
    return RUBRICS.get(intent)


def build_assistant(*, predictor, log: DecisionLog, sandbox_backend, llm, session_id: str = "", shared=None, computer=None):
    context_config.register_profile()
    backend = CompositeBackend(
        default=sandbox_backend,
        routes={
            "/memories/": memory.memory_backend(),
            "/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True),
        },
    )

    def sub(spec: dict) -> dict:
        # every subagent runs commands too, so it gets the gates (no intake note), the path policy, and the same 16k context tuning
        spec = dict(spec)
        read_only = spec.pop("read_only", False)
        return {**spec, "middleware": [LayaCortexMiddleware(predictor, log, is_subagent=True), PathPolicyMiddleware(read_only, shared=shared),
                                       *context_config.build_context_middleware(llm, backend)],
                "skills": ["/skills/"]}

    def on_rubric(ev):
        log.add("llm", "rubric", f"iteration {ev['iteration']}: {ev['result']}", None, 0.0)

    return create_deep_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT + memory.MEMORY_PROMPT,
        tools=[make_describe_image(backend), *make_host_tools(), *(make_computer_tools(computer) if computer is not None else [])],
        subagents=[sub(RESEARCHER), sub(CODER), sub(REVIEWER), build_advisor(predictor, log)],
        middleware=[
            LayaCortexMiddleware(predictor, log, gate_plans=True, host_tools={"mac_run"}),
            PathPolicyMiddleware(shared=shared),
            TodoListMiddleware(),  # create_deep_agent does not add the planning tool itself (tutorials 9-11 add it too)
            CodeInterpreterMiddleware(),
            # The grader is a nested agent with no call limit; with qwen 14B it looped (~20 calls in 90 s, no verdict).
            # Bound it: a grader that cannot answer within a few calls ends as `grader_error`, and the run finishes.
            RubricMiddleware(model=llm, max_iterations=2, on_evaluation=on_rubric,
                             grader_middleware=[ModelCallLimitMiddleware(run_limit=GRADER_CALL_LIMIT, exit_behavior="end")]),
            *context_config.build_context_middleware(llm, backend),
        ],
        backend=backend,
        skills=["/skills/"],
        memory=memory.MEMORY_PATHS,
        # write_todos is gated inside the cortex (asks only when the plan changes). The user's OK for an irreversible computer-use
        # step is enforced by the framework BEFORE the tool runs.
        interrupt_on={**({"computer_confirm": {"allowed_decisions": ["approve", "reject", "respond"]}} if computer is not None else {})},
        state_schema=AssistantState,
        context_schema=AssistantContext,
        checkpointer=InMemorySaver(),
    )
