"""Context management for a 16k window.

Deep Agents' defaults assume the model profile's full window: tool results are offloaded past 20,000
tokens and history is summarized at 85% of `max_input_tokens`. Our Ollama window is 16k, so a 15k-token
tool result would overflow before either triggers. `FilesystemMiddleware` is a required middleware (it
cannot be excluded or reconfigured through `create_deep_agent`), so:

  * `ToolResultOffloadMiddleware` clips big tool results at ~3k tokens BEFORE they reach
    FilesystemMiddleware's own 20k eviction (which stays as a backstop): full text goes to the sandbox
    filesystem, the model sees a path plus the first 10 lines and pulls fragments with read_file/grep.
  * The stock `SummarizationMiddleware` (excludable by name) is excluded via a HarnessProfile and replaced
    by `TunedSummarizationMiddleware`, a subclass whose different `.name` survives that exclusion, with
    triggers sized for 16k. Old write/edit arguments are truncated the same way.
  * `create_summarization_tool_middleware` adds a `compact_conversation` tool the agent can call itself.
"""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from deepagents import HarnessProfile, register_harness_profile
from deepagents.middleware import create_summarization_tool_middleware
from deepagents.middleware.summarization import SummarizationMiddleware, count_tokens_approximately

from . import config

# Tools whose output must not be re-offloaded (read_file is already paginated; it would loop).
_NEVER_OFFLOAD = {"read_file", "write_todos", "compact_conversation"}
OFFLOAD_DIR = "/workspace/.large_tool_results"


class ToolResultOffloadMiddleware(AgentMiddleware):
    def __init__(self, backend, limit_tokens: int = config.TOOL_EVICT_TOKENS):
        self.backend = backend
        self.limit_tokens = limit_tokens

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        name = request.tool_call["name"]
        if name in _NEVER_OFFLOAD or not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result
        tokens = count_tokens_approximately([result])
        if tokens <= self.limit_tokens:
            return result
        path = f"{OFFLOAD_DIR}/{request.tool_call['id']}.txt"
        written = self.backend.write(path, result.content)
        if getattr(written, "error", None):
            return result  # could not offload; better a big result than a lost one
        preview = "\n".join(result.content.splitlines()[:10])
        note = (f"Tool result too large (~{tokens} tokens), saved to {path}. "
                f"Use read_file or grep on that path to inspect it. First 10 lines:\n{preview}")
        return result.model_copy(update={"content": note})


class TunedSummarizationMiddleware(SummarizationMiddleware):
    """Same behaviour as the stock one, sized for 16k. A different class name keeps it from being
    caught by the profile's `excluded_middleware={"SummarizationMiddleware"}`."""


def effective_limits() -> dict:
    return {
        "tool_offload_tokens": config.TOOL_EVICT_TOKENS,
        "summary_trigger_tokens": config.SUMMARY_TRIGGER_TOKENS,
        "summary_keep_tokens": config.SUMMARY_KEEP_TOKENS,
        "window": config.NUM_CTX,
    }


def build_context_middleware(llm, backend) -> list[AgentMiddleware]:
    return [
        ToolResultOffloadMiddleware(backend),
        TunedSummarizationMiddleware(
            model=llm,
            backend=backend,
            trigger=("tokens", config.SUMMARY_TRIGGER_TOKENS),
            keep=("tokens", config.SUMMARY_KEEP_TOKENS),
            truncate_args_settings={
                "trigger": ("tokens", int(config.NUM_CTX * 0.5)),
                "keep": ("messages", 6),
                "max_length": 1500,
            },
        ),
        create_summarization_tool_middleware(llm, backend),
    ]


def register_profile() -> None:
    """Provider-level ("ollama") profile: drop the stock summarizer everywhere (main agent and subagents);
    ours is added explicitly via `build_context_middleware`. Idempotent."""
    register_harness_profile("ollama", HarnessProfile(excluded_middleware=frozenset({"SummarizationMiddleware"})))
