import types

import pytest
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from laya_assistant import config, context_config, memory


def test_effective_limits_fit_the_16k_window():
    lim = context_config.effective_limits()
    assert lim["window"] == 16384
    assert lim["tool_offload_tokens"] < lim["summary_trigger_tokens"] < lim["window"] * 0.75
    assert lim["summary_trigger_tokens"] + lim["summary_keep_tokens"] < lim["window"]
    # the stock defaults are the problem being fixed: they would not trigger inside 16k
    assert 20000 > lim["window"]


def _request(name="execute", call_id="c1"):
    return types.SimpleNamespace(tool_call={"id": call_id, "name": name, "args": {}})


def test_big_tool_result_is_offloaded_with_a_pointer_and_preview(tmp_path):
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    mw = context_config.ToolResultOffloadMiddleware(backend, limit_tokens=500)
    big = "\n".join(f"line {i}: some sample output text" for i in range(2000))
    out = mw.wrap_tool_call(_request(), lambda r: ToolMessage(content=big, tool_call_id="c1"))
    assert "saved to /workspace/.large_tool_results/c1.txt" in out.content
    assert "line 0:" in out.content and "line 9:" in out.content and "line 10:" not in out.content
    assert len(out.content) < len(big) / 20
    saved = (tmp_path / "workspace/.large_tool_results/c1.txt").read_text()
    assert saved == big  # nothing lost, the model can read_file/grep it later


def test_small_results_and_read_file_are_left_alone(tmp_path):
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    mw = context_config.ToolResultOffloadMiddleware(backend, limit_tokens=50)
    small = ToolMessage(content="ok", tool_call_id="c1")
    assert mw.wrap_tool_call(_request(), lambda r: small) is small
    big = ToolMessage(content="x " * 5000, tool_call_id="c2")
    assert mw.wrap_tool_call(_request("read_file", "c2"), lambda r: big) is big  # read_file is already paginated


def _captured_main_stack(monkeypatch, **kwargs):
    """The middleware list create_deep_agent hands to LangChain's create_agent for the MAIN agent."""
    import deepagents.graph as graph

    seen = []
    real = graph.create_agent

    def spy(*a, **kw):
        seen.append([m.name for m in kw.get("middleware", [])])
        return real(*a, **kw)

    monkeypatch.setattr(graph, "create_agent", spy)
    create_deep_agent(**kwargs)
    return seen[-1]  # the main agent is built last


def test_profile_swaps_the_stock_summarizer_for_the_tuned_one(monkeypatch):
    context_config.register_profile()
    llm = ChatOllama(model="qwen2.5:3b", num_ctx=config.NUM_CTX)
    backend = StateBackend()
    names = _captured_main_stack(
        monkeypatch, model=llm, backend=backend, middleware=context_config.build_context_middleware(llm, backend)
    )
    assert "SummarizationMiddleware" not in names, names  # the stock, 85%-of-window one is gone
    assert "TunedSummarizationMiddleware" in names, names
    assert "ToolResultOffloadMiddleware" in names, names
    assert "FilesystemMiddleware" in names, names  # required scaffolding intact (its 20k eviction is the backstop)


def test_memory_persists_across_backend_instances(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOME", tmp_path)
    first = CompositeBackend(default=StateBackend(), routes={"/memories/": memory.memory_backend()})
    assert first.write("/memories/sessions/2026-09-20.md", "built the assistant").error is None
    second = CompositeBackend(default=StateBackend(), routes={"/memories/": memory.memory_backend()})
    assert "built the assistant" in second.read("/memories/sessions/2026-09-20.md").file_data["content"]
    assert "What I know about the user" in second.read("/memories/AGENTS.md").file_data["content"]  # seeded


def test_memory_prompt_documents_the_layout():
    for token in ["/memories/AGENTS.md", "/memories/sessions/", "/memories/research/", "SHORT"]:
        assert token in memory.MEMORY_PROMPT
