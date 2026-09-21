import json

import pytest

from laya_assistant.decisions import DecisionLog, confidence_band, guard_action


@pytest.mark.parametrize(
    "conf,band",
    [(0.92, "act"), (0.65, "act"), (0.64, "verify"), (0.50, "verify"), (0.49, "defer"), (0.0, "defer")],
)
def test_confidence_band(conf, band):
    assert confidence_band(conf, 0.65, 0.50) == band


@pytest.mark.parametrize(
    "p,action",
    [(1.0, "block"), (0.90, "block"), (0.89, "ask"), (0.70, "ask"), (0.69, "allow"), (0.0, "allow")],
)
def test_guard_action(p, action):
    assert guard_action(p, block_at=0.90, ask_at=0.70) == action


def test_guard_can_be_block_only():
    # sandbox commands: a container's blast radius is small, so never ask, only block
    assert guard_action(0.85, block_at=0.90, ask_at=None) == "allow"
    assert guard_action(0.95, block_at=0.90, ask_at=None) == "block"


def test_log_rows_jsonl_and_stats(tmp_path):
    log = DecisionLog(path=tmp_path / "d.jsonl")
    log.add("laya", "intake", "chat -> act", 0.91234, 30.0)
    log.add("laya", "gate", "allow", 0.02, 20.0)
    log.add("llm", "model call", "text reply", None, 2000.0)
    assert log.rows[0]["n"] == 1 and log.rows[2]["n"] == 3
    assert log.stats("laya") == {"count": 2, "total_ms": 50.0, "avg_ms": 25.0}
    assert log.stats("tool") == {"count": 0, "total_ms": 0.0, "avg_ms": 0.0}
    lines = (tmp_path / "d.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["confidence"] == 0.912
