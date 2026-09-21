"""Choosing among the menu. Three tiers, cheapest first; only as much intelligence as the situation needs.

  code   unambiguous (one candidate, or a clear winner by lexical+embedding score): no model call, ~0 ms
  laya   ambiguous: ONE Laya pass answers which candidate, plus irreversible / needs_user / blocked, ~55 ms;
         its probabilities are blended with the embedding scores (Laya alone picks the right one only about half
         the time on the base checkpoint, so it is one vote, not the decision)
  llm    Laya and the embeddings disagree or are unsure: a small tie-break call to the LLM (slow, rare)
If even the LLM cannot separate them the step is handed back to the planner.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from .types import Candidate, Observation, StepPlan

CODE_SCORE = 0.85  # a candidate this well matched...
CODE_MARGIN = 0.15  # ...and this far ahead of the runner-up needs no model at all
MIN_SCORE = 0.30  # below this the best candidate is not a match at all: never click the least-bad element
BLEND_MARGIN = 0.12  # after blending Laya + embeddings, the winner must lead by this much
W_EMB, W_LAYA = 0.6, 0.4
SAFE_CLICK = re.compile(r"^(next|back|previous|search|cancel|close|dismiss|skip|accept all cookies|accept cookies|reject.*|ok|got it|show more|expand|open|menu|home)$", re.I)

STEP_QUESTIONS = {
    "irreversible": {"type": "noul", "instructions": "Would the step in `step` be irreversible, such as paying, deleting, sending or posting?"},
    "needs_user": {"type": "noul", "instructions": "Is the step in `step` blocked on something only the user can provide, such as a password, a verification code or a captcha?"},
    "blocked": {"type": "noul", "instructions": "Is the page in `page` blocked by a dialog, error or login wall so that the step cannot proceed?"},
}


@dataclass
class Decision:
    cand: Candidate | None
    tier: str  # code | laya | llm | none
    confidence: float = 0.0
    margin: float = 0.0
    irreversible: float = 0.0
    needs_user: float = 0.0
    blocked: float = 0.0
    laya_choice: str | None = None
    laya_conf: float = 0.0
    agreement: bool | None = None  # did Laya's pick match the final pick (None when Laya was not asked)
    laya_ms: float = 0.0
    llm_ms: float = 0.0
    probs: dict = field(default_factory=dict)


def _softmax(x: np.ndarray, t: float = 0.08) -> np.ndarray:
    z = (x - x.max()) / t
    e = np.exp(z)
    return e / e.sum()


def laya_pass(predictor, step: StepPlan, obs: Observation, cands: list[Candidate], want_target: bool = True) -> dict:
    """One forward pass: which candidate + the three safety questions. Shrinks the menu if Laya's option budget overflows."""
    cands = list(cands)
    while True:
        questions = dict(STEP_QUESTIONS)
        if want_target and len(cands) > 1:
            crit = {c.cid: c.label()[:50] for c in cands if c.element}
            crit["none"] = "none of these fits the step"
            questions["target"] = {"type": "choice", "instructions": "Which element is the step in `step` about?", "criteria": crit}
        state = {"step": step.text(), "page": obs.title[:70], "top": cands[0].label()[:60] if cands else ""}
        try:
            return predictor.predict(state, questions)["answers"]
        except ValueError:  # option text exceeded Laya's head budget: offer fewer candidates
            if len(cands) <= 2:
                raise
            cands = cands[:-2]


def decide(step: StepPlan, obs: Observation, cands: list[Candidate], predictor, llm_tiebreak=None) -> Decision:
    if not cands:
        return Decision(None, "none")
    top = cands[0]
    if top.element is not None and top.emb < MIN_SCORE:
        return Decision(None, "none", confidence=top.emb)
    second = cands[1] if len(cands) > 1 else None
    margin = top.emb - (second.emb if second else 0.0)
    unambiguous = len(cands) == 1 or (top.emb >= CODE_SCORE and margin >= CODE_MARGIN)
    label = top.element.label if top.element else ""
    needs_safety = top.kind in ("click", "press_enter") and not SAFE_CLICK.match(label.strip())

    if unambiguous and not needs_safety:
        return Decision(top, "code", confidence=top.emb, margin=margin)

    t0 = time.perf_counter()
    a = laya_pass(predictor, step, obs, cands, want_target=not unambiguous)
    laya_ms = (time.perf_counter() - t0) * 1000
    d = Decision(top, "code" if unambiguous else "laya", confidence=top.emb, margin=margin, laya_ms=laya_ms,
                 irreversible=a["irreversible"]["noul"], needs_user=a["needs_user"]["noul"], blocked=a["blocked"]["noul"])
    if unambiguous or "target" not in a:
        return d

    t = a["target"]
    d.laya_choice, d.laya_conf, d.probs = t["choice"], t["confidence"], t["probabilities"]
    emb = _softmax(np.array([c.emb for c in cands]))
    lp = np.array([d.probs.get(c.cid, 0.0) for c in cands])
    blend = W_EMB * emb + W_LAYA * lp
    order = np.argsort(-blend)
    best, runner = int(order[0]), (int(order[1]) if len(order) > 1 else None)
    d.cand, d.confidence = cands[best], float(blend[best])
    d.margin = float(blend[best] - (blend[runner] if runner is not None else 0.0))
    d.agreement = (t["choice"] == cands[best].cid)
    if d.probs.get("none", 0.0) > 0.6 and blend[best] < 0.5:
        d.cand, d.tier = None, "none"
        return d
    if d.margin >= BLEND_MARGIN:
        return d

    # unsure: a small LLM tie-break among the top three
    if llm_tiebreak is None:
        d.cand, d.tier = None, "none"
        return d
    t1 = time.perf_counter()
    pick = llm_tiebreak(step, obs, [cands[i] for i in order[:3]])
    d.llm_ms = (time.perf_counter() - t1) * 1000
    d.tier = "llm"
    d.cand = pick
    d.agreement = bool(pick and t["choice"] == pick.cid)
    if pick is None:
        d.tier = "none"
    return d
