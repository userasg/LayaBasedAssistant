"""Decision log and confidence policy: pure, no models."""
import json
from dataclasses import dataclass, field
from pathlib import Path


def confidence_band(conf: float, high: float, low: float) -> str:
    """act: trust Laya. verify: act but stay careful. defer: hand back to the LLM."""
    if conf >= high:
        return "act"
    return "verify" if conf >= low else "defer"


def guard_action(p_destructive: float, block_at: float, ask_at: float | None) -> str:
    """block / ask / allow. `ask_at=None` makes the gate block-only (used for sandbox commands)."""
    if p_destructive >= block_at:
        return "block"
    if ask_at is not None and p_destructive >= ask_at:
        return "ask"
    return "allow"


@dataclass
class DecisionLog:
    """One row per step, tagged with the system that made it (laya | llm | code | tool | stt).
    Every Laya row is also a {state, question, answer} example for a future Laya fine-tune."""

    path: Path | None = None
    rows: list[dict] = field(default_factory=list)

    def add(self, system: str, step: str, result: str, confidence: float | None = None, ms: float = 0.0) -> dict:
        row = {
            "n": len(self.rows) + 1,
            "system": system,
            "step": step,
            "result": result,
            "confidence": None if confidence is None else round(confidence, 3),
            "ms": round(ms, 1),
        }
        self.rows.append(row)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    def stats(self, system: str) -> dict:
        ms = [r["ms"] for r in self.rows if r["system"] == system]
        return {"count": len(ms), "total_ms": float(sum(ms)), "avg_ms": sum(ms) / len(ms) if ms else 0.0}
