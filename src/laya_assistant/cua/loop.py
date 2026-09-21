"""The fast loop. Per step, with no LLM in the common case:

    observe -> menu -> decide (code / Laya / LLM tie-break) -> validate -> act -> verify

Speed techniques (each measured in the benchmark):
  * the observation at the start of step N is also the verification of step N-1 (one observe per step, not two);
  * consecutive type/select/check steps share one observation and run back to back (form filling);
  * element embeddings are cached; unambiguous targets never touch a model;
  * Laya answers target + safety questions in a single forward pass.

"A score is not proof the action worked": every action is verified against the fresh observation, and a failed
verification or an unsure decision hands the step back to the planner instead of pushing on.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace

from . import menu as menu_mod
from .cache import DecisionCache
from .decide import Decision, decide, decide_known
from .policy import DomainPolicy, Verdict, host_of, validate
from .recorder import StepRecorder
from .types import Action, Candidate, Element, Observation, StepPlan

BATCHABLE = {"type", "select", "check"}
MAX_BATCH = 8
CHOICE_FLOOR = 0.12  # a candidate below this is not a plausible reading of the step: it is never offered to the user
MAX_CHOICES = 3
RETRY_WAIT_S = 0.6  # a desktop app updates a moment after an action: an unresolved step gets one more look before anyone is asked


@dataclass
class StepResult:
    step: StepPlan
    status: str  # ok | failed | needs_approval | needs_user | blocked | done
    reason: str = ""
    cand: Candidate | None = None
    decision: Decision | None = None
    timings: dict = field(default_factory=dict)
    verdict: Verdict | None = None

    @property
    def ms(self) -> float:
        return sum(self.timings.values())


@dataclass
class Pending:
    """An action that needs the user's OK. Resolved by `Loop.resume`, never re-decided."""
    pid: str
    step_index: int
    step: StepPlan
    cand: Candidate
    reason: str
    description: str


@dataclass
class RunResult:
    status: str  # done | failed | needs_choice | needs_approval | needs_user | blocked | stopped
    results: list[StepResult] = field(default_factory=list)
    next_index: int = 0
    reason: str = ""
    pending: Pending | None = None
    last_obs: Observation | None = None
    choices: list[Candidate] = field(default_factory=list)  # needs_choice: the plausible elements, best first
    choice_step: StepPlan | None = None
    more: bool = False  # done: the plan ended with "more": the goal needs steps that only make sense once the screen has changed

    @property
    def ms(self) -> float:
        return sum(r.ms for r in self.results)


def to_action(cand: Candidate, step: StepPlan) -> Action:
    if cand.kind == "navigate":
        return Action("navigate", url=cand.value)
    if cand.kind == "open_app":
        return Action("open_app", value=cand.value)
    if cand.kind == "scroll":
        return Action("scroll", dy=-500 if "up" in (cand.value or step.target).lower() else 500)
    if cand.kind == "wait":
        return Action("wait", dy=800)
    if cand.kind == "press_enter":
        return Action("key", id=cand.cid, key="Enter")
    return Action(cand.kind, id=cand.cid, value=cand.value)


def _label_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def find_element(cand: Candidate, obs: Observation) -> Element | None:
    """The element `cand` meant, looked up in a NEWER observation. A desktop id is a position in the accessibility tree, so it shifts whenever the
    UI changes (typing in Messages' search box brings up a results list and every index moves): the id is trusted only while it still names the same
    thing (same role and label); otherwise the element is found by role and label wherever it went. Judging by id alone reported a successful
    "type Manjit Gahir" as "field holds None" and sent the task into a re-plan.
    Measured on the real Messages window: once text is typed, the search field's LABEL becomes that text (it was "Search"), so a field that was
    just typed into is also recognised by holding, or being named after, what was typed."""
    want = cand.element
    if want is None:
        return obs.by_id(cand.cid)
    typed = _label_key((cand.value or "")[:120]) if cand.kind == "type" else ""

    def same(e: Element) -> bool:
        return e.role == want.role and (_label_key(e.label) == _label_key(want.label) or bool(typed and typed in (_label_key(e.value), _label_key(e.label))))

    el = obs.by_id(cand.cid)
    if el is not None and same(el):
        return el
    found = [e for e in obs.elements if same(e)]
    if not found:
        return None

    def gap(e: Element) -> int:  # several with the same label (three "Close" buttons): the one nearest where it was
        try:
            return abs(int(e.id[1:]) - int(cand.cid[1:]))
        except ValueError:
            return 0

    return min(found, key=gap)


def _shows(after: Observation, text: str) -> bool:
    """Is `text` visible anywhere in the window? The fallback when the field itself cannot be found again."""
    return bool(text) and text.lower() in (after.content or "").lower()


def verify(cand: Candidate, before: Observation, after: Observation, res: dict) -> tuple[bool, str]:
    """Did the action land? Judged from the fresh observation, not from any model's confidence."""
    if not res.get("ok"):
        return False, res.get("error") or "action failed"
    k = cand.kind
    if k in ("scroll", "wait"):
        return True, ""
    if k == "open_app":
        return (after.app or "").lower() == (cand.value or "").lower() or bool(after.elements), "the app did not open a window"
    if k == "navigate":
        return (host_of(after.url) == host_of(cand.value) or (after.url or "") != (before.url or "")), "did not reach the page"
    el_after = find_element(cand, after)
    if k == "type":
        want = (cand.value or "").strip()[:120].strip()
        if el_after is None:  # the field moved or was renamed: the typed text showing up on screen is the evidence
            return _shows(after, want), f"the field is gone and {cand.value!r} is not on screen"
        got = el_after.value
        return (got is not None and got.strip() == want), f"field holds {got!r}, expected {cand.value!r}"
    if k == "select":
        option = (cand.value or "").lower()
        if el_after is None:
            return _shows(after, option), "the option was not selected"
        return (option in el_after.value.lower()), "the option was not selected"
    if k == "check":
        want = str(cand.value).lower() not in ("false", "0", "off", "no") if cand.value is not None else True
        return (el_after is not None and el_after.checked == want), "the checkbox did not change"
    # click / press_enter: something observable must change (page, url, dialog, or the element itself)
    changed = (after.fingerprint != before.fingerprint or (after.url or "") != (before.url or "") or after.dialogs != before.dialogs)
    return changed, "nothing on the page changed after the click"


class Loop:
    def __init__(self, driver, predictor, ranker, domain: DomainPolicy | None = None, recorder: StepRecorder | None = None,
                 llm_tiebreak=None, log=None, allowed_apps: set[str] | None = None, stop_flag=None, sleep_after=0.0,
                 cache: DecisionCache | None = None):
        self.cache = cache
        self.driver, self.predictor, self.ranker = driver, predictor, ranker
        self.domain = domain or DomainPolicy()
        self.rec = recorder or StepRecorder()
        self.llm_tiebreak, self.log = llm_tiebreak, log
        self.allowed_apps = allowed_apps
        self.stop_flag = stop_flag  # threading.Event: the global Stop
        self._pid = 0

    # -- one step -------------------------------------------------------------------------------------

    def _resolve(self, step: StepPlan, obs: Observation, timings: dict, forced: tuple[str, str] | None = None):
        """(decision, menu). The target is known without asking a model when the user just chose it (`forced`: role, label) or when a cached
        choice for this exact step still resolves to an element on this screen; otherwise it is decided as before."""
        t0 = time.perf_counter()
        cands = menu_mod.build(step, obs, self.ranker)
        timings["menu_ms"] = (time.perf_counter() - t0) * 1000
        if not cands and forced is None:
            return None, cands
        known, tier = None, ""
        if forced is not None:
            known, tier = self._find(step, obs, cands, forced), "human"
        elif self.cache is not None and step.do in menu_mod.ROLES_FOR and (hit := self.cache.get(step, obs)) is not None:
            known, tier = self._find(step, obs, cands, hit, search_all=False), "cache"
        t0 = time.perf_counter()
        d = decide_known(step, obs, known, cands, self.predictor, tier) if known is not None else decide(step, obs, cands, self.predictor, self.llm_tiebreak)
        timings["decide_ms"] = (time.perf_counter() - t0) * 1000
        return d, cands

    @staticmethod
    def _find(step: StepPlan, obs: Observation, cands: list[Candidate], want: tuple[str, str], search_all: bool = True) -> Candidate | None:
        """The candidate for element (role, label). A user's choice may fall outside the shortlist on a changed screen, so it also looks at every element."""
        role, label = want
        for c in cands:
            if c.element is not None and (c.element.role, c.element.label) == (role, label):
                return c
        if search_all:
            for e in obs.elements:
                if (e.role, e.label) == (role, label) and not e.disabled:
                    return Candidate(e.id, "click" if step.do == "click" else step.do, e, step.value, 1.0)
        return None

    @staticmethod
    def choices_for(cands: list[Candidate]) -> list[Candidate]:
        """What to offer the user when nothing could be chosen with confidence: the plausible elements, distinct by (role, label), best first."""
        out, seen = [], set()
        for c in cands:
            if c.element is None or c.emb < CHOICE_FLOOR or (c.element.role, c.element.label) in seen:
                continue
            seen.add((c.element.role, c.element.label))
            out.append(c)
            if len(out) == MAX_CHOICES:
                break
        return out

    def _learn(self, r: "StepResult", before: Observation, cand: Candidate, ok: bool) -> None:
        """Remember a choice that verifiably worked; forget a remembered one that did not."""
        if self.cache is None or r.decision is None:
            return
        if ok and r.decision.tier in ("laya", "llm", "human"):
            self.cache.put(r.step, before, cand, r.decision.tier)
        elif not ok and r.decision.tier == "cache":
            self.cache.evict(r.step, before)

    def _validate(self, step: StepPlan, d: Decision, obs: Observation, timings: dict) -> Verdict:
        t0 = time.perf_counter()
        v = validate(step.do, d.cand, obs, self.domain, laya_irreversible=d.irreversible, allowed_apps=self.allowed_apps)
        if v.action == "allow" and d.needs_user >= 0.75 and d.cand.kind == "type":
            v = Verdict("handover", f"Laya thinks this needs you (p={d.needs_user:.2f})")
        timings["validate_ms"] = (time.perf_counter() - t0) * 1000
        return v

    def _record(self, step, obs, cands, d, chosen, verdict, res, ok, why, timings):
        self.rec.add(step=step.text(), page=obs.title, host=obs.host,
                     cands=[{"id": c.cid, "label": c.label(), "score": round(c.emb, 3)} for c in cands[:6]],
                     tier=d.tier if d else "none", laya_choice=getattr(d, "laya_choice", None), laya_conf=getattr(d, "laya_conf", 0),
                     laya_probs={k: round(v, 3) for k, v in (getattr(d, "probs", {}) or {}).items() if v > 0.02},
                     chosen=chosen.cid if chosen else None, agreement=getattr(d, "agreement", None),
                     safety={"irreversible": round(getattr(d, "irreversible", 0), 3), "needs_user": round(getattr(d, "needs_user", 0), 3),
                             "blocked": round(getattr(d, "blocked", 0), 3)},
                     verdict=verdict.action if verdict else None, verified=ok, why=why, ms={k: round(v, 1) for k, v in timings.items()})

    # -- a plan ---------------------------------------------------------------------------------------

    def _settle(self, awaiting: list, obs: Observation, record: bool = True):
        """Verify every action still owed a check against this fresh observation. Returns (index, result, why) of the first failure."""
        for idx, cand, before, res, r in awaiting:
            t0 = time.perf_counter()
            ok, why = verify(cand, before, obs, res)
            if record:
                r.timings["verify_ms"] = (time.perf_counter() - t0) * 1000
                self.rec.add(kind="outcome", step=r.step.text(), verified=ok, why="" if ok else why)  # joins to the decision row: see authority.py
                self._learn(r, before, cand, ok)
            if not ok:
                if record:
                    r.status, r.reason = "failed", why
                return idx, r, why
        return None

    def _settle_patiently(self, awaiting: list, obs: Observation):
        """(failure or None, the observation to use next). A desktop app often updates a moment AFTER the click (animations, async lists), so a
        click that looks like it changed nothing gets one more look; if it still changed nothing, the same click is retried a second way (a
        real click at the element's position instead of the accessibility press: some controls only answer one of the two)."""
        desktop = getattr(self.driver, "name", "") == "desktop"
        if awaiting and self._settle(awaiting, obs, record=False) is not None and desktop:
            time.sleep(0.9)
            obs = self.driver.observe()
            bad = self._settle(awaiting, obs, record=False)
            if bad is not None and bad[1].step.do == "click":
                idx, _, _ = bad
                cand = next(c for i, c, *_ in awaiting if i == idx)
                el = find_element(cand, obs) if cand.kind == "click" and cand.cid else None  # ids are positions: use where it is NOW
                if el is not None and getattr(self.driver, "click_by_position", None):
                    for foreground in (False, True):  # the driver's ladder: background pixel click, then the window briefly fronted
                        if not self.driver.click_by_position(el.id, foreground=foreground).get("ok"):
                            continue
                        time.sleep(0.9)
                        obs = self.driver.observe()
                        if self._settle(awaiting, obs, record=False) is None:
                            break
        return self._settle(awaiting, obs), obs

    def run(self, steps: list[StepPlan], start: int = 0, obs: Observation | None = None, forced: dict[int, tuple[str, str]] | None = None) -> RunResult:
        """`forced` maps a step index to the (role, label) of the element the user chose for it."""
        out = RunResult("done", next_index=start)
        i = start
        awaiting: list = []  # (index, candidate, observation before, act result, StepResult): verified by the NEXT observation
        while i < len(steps):
            if self.stop_flag is not None and self.stop_flag.is_set():
                out.status, out.reason, out.next_index = "stopped", "stopped by the user", i
                return out
            step = steps[i]
            if step.do in ("done", "more"):
                out.more = step.do == "more"
                break
            timings: dict = {}
            if obs is None or awaiting:  # after any action the page may have changed: look again (this is also the check)
                t0 = time.perf_counter()
                obs = self.driver.observe()
                timings["observe_ms"] = (time.perf_counter() - t0) * 1000
            failed, obs = self._settle_patiently(awaiting, obs)
            out.last_obs = obs
            awaiting = []
            if failed:
                idx, r, why = failed
                out.status, out.reason, out.next_index = "failed", f"step {r.step.text()!r} did not land: {why}", idx
                return out

            batch = [step]  # consecutive form-filling steps share this one observation
            if step.do in BATCHABLE:
                j = i + 1
                while j < len(steps) and steps[j].do in BATCHABLE and len(batch) < MAX_BATCH:
                    batch.append(steps[j])
                    j += 1
            for k, st in enumerate(batch):
                r = StepResult(st, "ok", timings=dict(timings) if k == 0 else {})
                spec = (forced or {}).get(i)
                d, cands = self._resolve(st, obs, r.timings, spec)
                if (d is None or d.cand is None) and getattr(self.driver, "name", "") == "desktop":
                    time.sleep(RETRY_WAIT_S)  # cheap local retry: look once more before anyone (a model or the user) is involved
                    obs = self.driver.observe()
                    out.last_obs = obs
                    d, cands = self._resolve(st, obs, r.timings, spec)
                r.decision = d
                if d is None or d.cand is None:
                    r.status = "failed"
                    r.reason = "no candidate fits this step" if d is None else "Laya and the embeddings could not tell which element"
                    self._record(st, obs, cands or [], d, None, None, None, False, r.reason, r.timings)
                    out.results.append(r)
                    out.status, out.reason, out.next_index = "failed", r.reason, i
                    if choices := self.choices_for(cands or []):  # something plausible exists: the user can pick, no model needs to guess
                        out.status, out.choices, out.choice_step = "needs_choice", choices, st
                    return out
                r.cand = d.cand
                v = self._validate(st, d, obs, r.timings)
                r.verdict = v
                if v.action != "allow":
                    r.status = {"block": "blocked", "handover": "needs_user", "ask": "needs_approval"}[v.action]
                    r.reason = v.reason
                    self._record(st, obs, cands, d, d.cand, v, None, False, v.reason, r.timings)
                    out.results.append(r)
                    out.status, out.reason, out.next_index = r.status, v.reason, i
                    if v.action == "ask":
                        self._pid += 1
                        where = obs.host or (f"{obs.app}: {obs.title[:60]}" if obs.title else obs.app)  # the title names the open conversation or document: WHO a message goes to
                        out.pending = Pending(f"p{self._pid}", i, st, d.cand, v.reason, f"{st.do} {d.cand.label()} ({where})")
                    return out
                t0 = time.perf_counter()
                res = self.driver.act(to_action(d.cand, st))
                r.timings["act_ms"] = (time.perf_counter() - t0) * 1000
                self._record(st, obs, cands, d, d.cand, v, res, None, "", r.timings)
                out.results.append(r)
                if not res.get("ok"):
                    r.status, r.reason = "failed", res.get("error", "action failed")
                    out.status, out.reason, out.next_index = "failed", r.reason, i
                    return out
                awaiting.append((i, d.cand, obs, res, r))
                i += 1
        if awaiting:  # close the loop: one last look verifies the final action(s)
            t0 = time.perf_counter()
            final = self.driver.observe()
            final_ms = (time.perf_counter() - t0) * 1000
            awaiting[-1][4].timings["observe_ms"] = awaiting[-1][4].timings.get("observe_ms", 0) + final_ms
            failed, final = self._settle_patiently(awaiting, final)
            out.last_obs = final
            if failed:
                idx, r, why = failed
                out.status, out.reason, out.next_index = "failed", f"step {r.step.text()!r} did not land: {why}", idx
                return out
        out.next_index = len(steps)
        return out

    def resume(self, steps: list[StepPlan], pending: Pending) -> RunResult:
        """The user approved `pending`: perform exactly that action, verify it, then carry on with the rest."""
        obs = self.driver.observe()
        cand = pending.cand
        if pending.step.do not in ("navigate", "scroll", "wait"):
            el = find_element(cand, obs)  # the ids may have moved while the user was deciding; the control itself must still be there
            if el is None:
                return RunResult("failed", reason="the page changed while waiting for your approval", next_index=pending.step_index, last_obs=obs)
            if el.id != cand.cid:
                cand = replace(cand, cid=el.id, element=el)
        if obs.source == "browser":
            self.domain.approve(obs.url)
        res = self.driver.act(to_action(cand, pending.step))
        after = self.driver.observe()
        ok, why = verify(cand, obs, after, res)
        result = StepResult(pending.step, "ok" if ok else "failed", why, cand)
        if not ok:
            return RunResult("failed", [result], pending.step_index, why, last_obs=after)
        rest = self.run(steps, start=pending.step_index + 1, obs=after)
        rest.results.insert(0, result)
        return rest
