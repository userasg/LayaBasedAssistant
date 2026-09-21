import threading

from laya_assistant import config, engine


def test_concurrent_predicts_do_not_crash_the_process(laya_model):
    """Regression: unlocked concurrent MPS predicts abort the interpreter with
    'failed assertion _status < MTLCommandBufferStatusCommitted'. A crash kills pytest itself."""
    q = {"intent": {"type": "noul", "instructions": "Is `request` a greeting?"}}
    out = []

    def work(i):
        for _ in range(8):
            out.append(laya_model.predict({"request": f"hello {i}"}, q))

    ts = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(out) == 48


def test_make_llm_uses_constant_context_and_keep_alive():
    llm = engine.make_llm(config.FAST_MODEL)
    assert llm.num_ctx == 16384 == config.NUM_CTX
    assert llm.keep_alive == config.KEEP_ALIVE
    assert llm.model == config.FAST_MODEL


def test_check_models_reports_missing_only():
    assert engine.check_models([config.FAST_MODEL]) == []
    assert engine.check_models(["definitely-not-a-model:1b"]) == ["definitely-not-a-model:1b"]


def test_llm_lock_is_the_engine_lock_unless_parallel_is_enabled(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_LLM_PARALLEL", False)
    assert engine.llm_lock() is engine.ENGINE
    monkeypatch.setattr(config, "ENGINE_LLM_PARALLEL", True)
    assert engine.llm_lock() is not engine.ENGINE  # LLM calls may overlap; Laya/Whisper never do
