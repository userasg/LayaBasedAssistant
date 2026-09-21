"""Remembers which element a step meant, once that choice has verifiably worked (the idea from Stagehand's action cache).

The expensive part of a step is the ambiguous one: Laya's pass, and the LLM tie-break behind it. When the same step (same words, same site or app)
was resolved before AND the action landed, the answer is stored as (role, label) and replayed next time, skipping both. Element ids change between
observations, so the label is the key; a cached choice is only used if an element with that role and label is on the screen now, and a cached
choice that fails to land is evicted, so a stale entry costs one wasted step, never a loop.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .types import Candidate, Observation, StepPlan

MAX_ENTRIES = 500
_WORD = re.compile(r"[a-z0-9]+")


def key(step: StepPlan, obs: Observation) -> str:
    where = (obs.host or obs.app or "").lower()
    return f"{where}|{step.do}|{' '.join(_WORD.findall(step.target.lower()))}"


class DecisionCache:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.entries: dict[str, dict] = {}
        self.hits = self.misses = 0
        if path and path.exists():
            try:
                self.entries = json.loads(path.read_text())
            except (OSError, ValueError):
                self.entries = {}  # a damaged cache is just an empty one

    def get(self, step: StepPlan, obs: Observation) -> tuple[str, str] | None:
        e = self.entries.get(key(step, obs))
        if e is None:
            self.misses += 1
            return None
        self.hits += 1
        return e["role"], e["label"]

    def put(self, step: StepPlan, obs: Observation, cand: Candidate, tier: str) -> None:
        if cand.element is None or not step.target.strip():
            return  # only element choices are worth remembering (a scroll or a navigation needs no decision)
        self.entries[key(step, obs)] = {"role": cand.element.role, "label": cand.element.label, "tier": tier}
        while len(self.entries) > MAX_ENTRIES:
            self.entries.pop(next(iter(self.entries)))  # oldest first
        self._save()

    def evict(self, step: StepPlan, obs: Observation) -> None:
        if self.entries.pop(key(step, obs), None) is not None:
            self._save()

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.entries))
        except OSError:
            pass  # the cache is an optimisation: never fail a step over it
