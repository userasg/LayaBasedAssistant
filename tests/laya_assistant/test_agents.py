"""Full-stack integration: real Ollama executor, real Docker sandbox, real Laya intake. Slow."""
import uuid

import pytest
from langgraph.types import Command

from laya_assistant import agents, config, intake
from laya_assistant.decisions import DecisionLog
from laya_assistant.engine import check_models, make_llm
from laya_assistant.sandbox import make_backend

pytestmark = pytest.mark.slow

PROMPT = "Plan the work first, then create hello.py that prints hello world and run it to check it works."


@pytest.fixture(scope="module")
def stack(laya_model):
    assert check_models([config.EXECUTOR_MODEL]) == [], "pull the executor model first"
    handle = make_backend(prefer_docker=True)
    assert handle.label.startswith("Docker"), handle.label
    log = DecisionLog()
    agent = agents.build_assistant(predictor=laya_model, log=log, sandbox_backend=handle.backend, llm=make_llm(config.EXECUTOR_MODEL))
    yield agent, handle, log
    handle.stop()


def drive(agent, first_input, config_):
    """Run until the graph finishes or pauses; returns the interrupt payload or None."""
    for chunk in agent.stream(first_input, config=config_, stream_mode="updates"):
        if "__interrupt__" in chunk:
            return chunk["__interrupt__"][0].value
    return None


def test_code_task_pauses_for_plan_then_executes_in_the_sandbox_and_is_graded(stack, laya_model, plan_approval_on):
    agent, handle, log = stack
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": config.STEP_BUDGET}
    i = intake.classify(laya_model, PROMPT, log=log)
    assert i.route == "executor"
    state_in = {"messages": [{"role": "user", "content": PROMPT}], "laya": intake.agent_hint(i),
                **({"rubric": agents.rubric_for(i.intent)} if agents.rubric_for(i.intent) else {})}

    pending = drive(agent, state_in, cfg)
    assert pending is not None, "expected the run to pause at write_todos for plan approval"
    assert pending["action_requests"][0]["name"] == "write_todos"

    approve = lambda pend: Command(resume={"decisions": [{"type": "approve"}] * len(pend["action_requests"])})  # one decision per pending action
    pending = drive(agent, approve(pending), cfg)
    while pending is not None:  # each todo update pauses again for review: approve to let the run finish
        pending = drive(agent, approve(pending), cfg)

    listing = handle.backend.execute("ls /workspace").output
    print("workspace:", listing.split())
    print("log:", [(r["system"], r["step"], r["result"][:40], r["ms"]) for r in log.rows])
    assert "hello.py" in listing
    assert "hello" in handle.backend.execute("python hello.py").output.lower()
    systems = {r["system"] for r in log.rows}
    assert {"laya", "llm", "tool", "code"} <= systems
