"""How Laya EARNS control: measured from recorded runs, never assumed.

Every ambiguous step the loop records what Laya chose, what was finally chosen, and (once the next observation arrives)
whether the action verifiably landed. Here those rows are turned into an accuracy figure per confidence band. Laya is
allowed to decide alone at a confidence level only after it has been right on at least `min_samples` recorded steps at
that level with accuracy >= `min_accuracy`. Until then it is one vote in a blend, and code verification checks the result.
Base-checkpoint measurements that motivate this: target choice 5/12, action choice 7/15, tool choice 12/20, goal-done 5/10.
"""
from __future__ import annotations

BANDS = ((0.90, 1.01), (0.75, 0.90), (0.60, 0.75), (0.0, 0.60))


def paired(rows: list[dict]) -> list[dict]:
    """Join each recorded decision with the outcome row that followed it (matched on step text, in order)."""
    out, pending = [], {}
    for r in rows:
        if r.get("kind") == "outcome":
            d = pending.pop(r.get("step"), None)
            if d is not None:
                out.append({**d, "verified": bool(r.get("verified"))})
        elif r.get("laya_choice") is not None:
            pending[r["step"]] = r
    return out


def accuracy_report(rows: list[dict]) -> dict:
    """{'n': total, 'bands': {(lo, hi): {'n', 'correct', 'accuracy'}}}; 'correct' = Laya's pick was the final pick AND it landed."""
    data = paired(rows)
    bands = {}
    for lo, hi in BANDS:
        sel = [d for d in data if lo <= (d.get("laya_conf") or 0) < hi]
        ok = sum(1 for d in sel if d["laya_choice"] == d.get("chosen") and d["verified"])
        bands[(lo, hi)] = {"n": len(sel), "correct": ok, "accuracy": (ok / len(sel)) if sel else None}
    return {"n": len(data), "bands": bands}


def min_confidence_for_authority(rows: list[dict], min_samples: int = 30, min_accuracy: float = 0.9) -> float | None:
    """The lowest confidence at or above which Laya has earned the right to decide alone (None = not yet).
    Going down band by band, a lower band is only added if it has its OWN evidence (at least half the sample requirement
    inside that band) and the cumulative accuracy still holds: high-confidence samples never vouch for lower confidence."""
    data = paired(rows)
    best = None
    for k, (lo, hi) in enumerate(BANDS[:-1]):
        cum = [d for d in data if (d.get("laya_conf") or 0) >= lo]
        own = [d for d in data if lo <= (d.get("laya_conf") or 0) < hi]
        acc = (sum(1 for d in cum if d["laya_choice"] == d.get("chosen") and d["verified"]) / len(cum)) if cum else 0.0
        if len(cum) >= min_samples and acc >= min_accuracy and (k == 0 or len(own) >= min_samples // 2):
            best = lo
        else:
            break
    return best


def summary_line(rows: list[dict], min_samples: int = 30, min_accuracy: float = 0.9) -> str:
    rep = accuracy_report(rows)
    if rep["n"] == 0:
        return "Laya has no recorded choices yet."
    got = min_confidence_for_authority(rows, min_samples, min_accuracy)
    top = rep["bands"][BANDS[0]]
    if got is not None:
        return f"Laya has earned control of target choice at confidence >= {got:.2f} ({rep['n']} recorded)."
    return (f"Laya advises, code verifies: {rep['n']} recorded choices; at >=0.90 confidence it was right {top['correct']}/{top['n']} "
            f"(needs {min_samples}+ at {min_accuracy:.0%} to take over).")
