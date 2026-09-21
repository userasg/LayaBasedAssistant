"""Model engine: one lock for the GPU, the Laya wrapper, the LLM factory, and startup checks.

ONE model call at a time, process-wide. Apple's MPS backend is not thread-safe (concurrent Laya
calls abort the process with `failed assertion _status < MTLCommandBufferStatusCommitted`), and the
GPU is shared with Ollama, so a second LLM request piled on top makes both crawl. Laya calls take
~30-60 ms, so waiting for a turn is not noticeable.
"""
import contextlib
import threading

import requests
from langchain_ollama import ChatOllama

from . import config

ENGINE = threading.Lock()


class LockedLaya:
    """A Laya model whose `predict` takes its turn on ENGINE. Never parallel, even when
    ENGINE_LLM_PARALLEL is on (torch/MPS is the fragile part)."""

    def __init__(self, model):
        self._model = model
        self.device = model.device

    def predict(self, state, questions):
        with ENGINE:
            return self._model.predict(state, questions)


def load_laya() -> LockedLaya:
    import laya

    predictor = LockedLaya(laya.load(config.LAYA_CHECKPOINT))
    predictor.predict({"request": "warm up"}, {"q": {"type": "noul", "instructions": "Is `request` a greeting?"}})
    return predictor


def make_llm(name: str, **overrides) -> ChatOllama:
    """Ollama chat model. num_ctx is identical for every request: changing it reloads the model."""
    kwargs = dict(
        model=name,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.TEMPERATURE,
        num_ctx=config.NUM_CTX,
        keep_alive=config.KEEP_ALIVE,
        repeat_penalty=1.1,  # low-temperature qwen otherwise sometimes loops until Ollama aborts the generation
    )
    kwargs.update(overrides)
    return ChatOllama(**kwargs)


def llm_lock():
    """The lock an LLM call must hold: ENGINE by default, a no-op when parallel LLMs are enabled."""
    return contextlib.nullcontext() if config.ENGINE_LLM_PARALLEL else ENGINE


def check_models(required: list[str] | None = None) -> list[str]:
    """Names of required Ollama models that are not pulled (empty list = all present).
    Raises requests.ConnectionError if Ollama is not running."""
    required = required if required is not None else [config.EXECUTOR_MODEL, config.FAST_MODEL, config.VISION_MODEL]
    tags = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5).json()["models"]
    have = {m["name"] for m in tags} | {m["name"].removesuffix(":latest") for m in tags}
    return [m for m in required if m not in have]


def warm_ollama(name: str) -> float:
    """Load a model into GPU memory with a 1-token request so the first real turn is fast. Returns ms."""
    import time

    t0 = time.perf_counter()
    with llm_lock():
        make_llm(name, num_predict=1).invoke("hi")
    return (time.perf_counter() - t0) * 1000
