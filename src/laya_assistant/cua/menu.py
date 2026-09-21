"""The candidate menu: the application, not the model, decides what actions are on offer.

The LLM said what it wants in words ("type 'sony' into the search box"). Here code turns that into a short list of
concrete candidates (real elements on the real page, each with an id), ranked by lexical match plus embedding
similarity. Laya can only choose among these; it cannot invent a target.
"""
from __future__ import annotations

import re

import numpy as np

from .types import Candidate, Element, Observation, StepPlan

ROLES_FOR = {
    "click": {"button", "link", "tab", "menuitem", "checkbox", "radio", "option", "switch", "combobox", "select"},
    "type": {"textbox", "combobox"},
    "select": {"select", "combobox"},
    "check": {"checkbox", "radio", "switch"},
    "press_enter": {"textbox", "combobox"},
}
# Words a person uses for the KIND of element. If the step says "link" and an element is a button (or vice versa) it is
# probably not what was meant, however similar the labels: "click the first result link" must not press a "Search" button.
ROLE_WORDS = {
    "link": "link", "links": "link", "button": "button", "btn": "button",
    "checkbox": "checkbox", "radio": "radio", "dropdown": "select", "select": "select", "tab": "tab",
    "box": "textbox", "field": "textbox", "input": "textbox", "textbox": "textbox", "bar": "textbox",
}
ROLE_MISMATCH = 0.4
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_WORD = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def lexical(query: str, label: str) -> float:
    """1.0 for an exact match, else the share of query words present in the label."""
    q, l = _norm(query), _norm(label)
    if not q or not l:
        return 0.0
    if q == l:
        return 1.0
    qw, lw = set(q.split()), set(l.split())
    if qw <= lw or lw <= qw:
        return 0.6 + 0.4 * len(qw & lw) / max(len(qw | lw), 1)
    return 0.6 * len(qw & lw) / len(qw)


class Ranker:
    """Sentence embeddings (bge-small, on CPU: no contention with the GPU lock), with a cache per element text."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _m(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def warm(self) -> None:
        self.score("warm up", ["button: ok"])

    def _embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self._m().encode(missing, normalize_embeddings=True, batch_size=64, show_progress_bar=False)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        q = self._m().encode(QUERY_PREFIX + query, normalize_embeddings=True, show_progress_bar=False)
        return self._embed(texts) @ q


def _text(e: Element) -> str:
    return f"{e.role}: {e.label} {e.value}".strip()


def build(step: StepPlan, obs: Observation, ranker: Ranker, k: int = 6) -> list[Candidate]:
    """Ranked candidates for one step. `score` is 0..1: 0.5 lexical + 0.5 embedding (clipped)."""
    if step.do == "scroll":
        return [Candidate("scroll", "scroll", None, step.value)]
    if step.do == "navigate":
        return [Candidate("navigate", "navigate", None, step.value or step.target)]
    if step.do == "open_app":
        return [Candidate("open_app", "open_app", None, step.value or step.target)]
    if step.do in ("wait", "done"):
        return [Candidate(step.do, step.do, None, None)]
    kind = "click" if step.do == "click" else step.do
    roles = ROLES_FOR.get(step.do, ROLES_FOR["click"])
    pool = [e for e in obs.elements if not e.disabled and e.role in roles] or [e for e in obs.elements if not e.disabled]
    if not pool:
        return []
    lex = np.array([max(lexical(step.target, e.label), lexical(step.target, e.value) * 0.5) for e in pool])
    strong = lex >= 1.0
    emb = np.clip(ranker.score(step.target, [_text(e) for e in pool]), 0, 1) if not strong.all() else np.ones(len(pool))
    comb = 0.5 * lex + 0.5 * emb
    hinted = {ROLE_WORDS[w] for w in _norm(step.target).split() if w in ROLE_WORDS}
    if hinted and step.do in ("click", "check"):
        comb = np.array([c * (1.0 if e.role in hinted else ROLE_MISMATCH) for c, e in zip(comb, pool)])
    order = np.argsort(-comb)[:k]
    return [Candidate(pool[i].id, kind, pool[i], step.value, float(comb[i])) for i in order]
