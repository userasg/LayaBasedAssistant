"""ComputerUse: runs a goal end to end and is what the agent's tools call.

  plan (LLM, once) -> fast loop (no LLM) -> on a failed/unsure step the ladder, all in code and each rung bounded:
      1. look again (in the loop)  2. re-plan once from the screen as it is now (skipped when it would repeat a failure already seen)
      3. ask the user, offering the 2-3 plausible actions (NEEDS_CHOICE; the answer is resolved without a model)  4. STALLED report
  on needs-approval: park the exact action and return; the user's OK (a HITL-gated tool) executes that action.
Nothing on the ladder hands control back to free-form retrying: after NEEDS_CHOICE or STALLED the agent is told to relay it and stop, and both the same
target and `mac_run` are refused until the user answers or sends a new request (`blocked_message`).
Every outcome is text the agent can relay: done / needs_user / needs_choice / needs_approval / blocked / stalled / stopped.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from .cache import DecisionCache
from .loop import Loop, Pending, RunResult
from .policy import DomainPolicy
from .recorder import StepRecorder
from .types import Observation, StepPlan

MAX_REPLANS = 1
STALL_HOLD_S = 90  # after a stall the same target is refused for this long (or until the user sends a new request)
PLAN_HISTORY = 6  # finished steps the planner is shown: what it needs to continue, never the whole run
MAX_CHUNKS = 6  # plans for one goal: the planner writes what it can see and says "more" when the next steps depend on what appears
MAX_TOTAL_STEPS = 25  # finished steps for one goal, across all its chunks


@dataclass
class Job:
    pid: str
    target: str
    goal: str
    steps: list[StepPlan]
    pending: Pending
    loop: Loop
    done_texts: list[str] = field(default_factory=list)
    replans: int = 0
    app: str = ""
    chunks: int = 0
    last_fp: str = ""  # the screen when the current chunk was planned: a chunk that leaves it unchanged made no progress
    seen_failures: set = field(default_factory=set)  # (step, reason) already met: meeting one again goes straight to asking, not to another plan
    choice: "Choice | None" = None


@dataclass
class Choice:
    """A step Laya could not settle, parked with the options offered to the user."""
    pid: str
    step_index: int
    step: StepPlan
    options: list[tuple[str, str]]  # (role, label), in the order shown
    described: list[str]
    created: float = field(default_factory=time.time)


def describe(obs: Observation | None) -> str:
    if obs is None:
        return "(no page)"
    return f"'{obs.title}' at {obs.url or obs.app}: " + (obs.text[:220] or ", ".join(e.label for e in obs.elements[:6]))


class ComputerUse:
    def __init__(self, predictor, ranker, planner_factory, driver_factory, domain: DomainPolicy | None = None,
                 recorder: StepRecorder | None = None, llm_tiebreak=None, log=None, allowed_apps=None, own_domain: DomainPolicy | None = None, cache=None, methods=None):
        self.predictor, self.ranker, self.planner_factory, self.driver_factory = predictor, ranker, planner_factory, driver_factory
        self.domain = domain or DomainPolicy()
        self.own_domain = own_domain  # policy for the agent-owned browser
        self._unavailable: dict[str, tuple[float, str]] = {}
        self.rec = recorder or StepRecorder()
        self.llm_tiebreak, self.log, self.allowed_apps = llm_tiebreak, log, allowed_apps
        self.methods = methods  # a MethodMemory (which way of working in each app has worked), or None
        self.cache = cache  # a DecisionCache, or None (tests, and anything that must not learn)
        self.stop = threading.Event()  # the global Stop: checked before every step
        self.jobs: dict[str, Job] = {}
        self.choices: dict[str, Job] = {}  # pid -> the job waiting for the user to pick an option
        self._cid = 0
        self._stalled: dict[str, tuple[float, str]] = {}  # "target:app" -> (when, the STALLED text)
        self.last_run_ms = 0.0
        self.step_log: list[dict] = []  # for the UI feed

    def _loop(self, driver) -> Loop:
        domain = self.own_domain if (getattr(driver, "own", False) and self.own_domain is not None) else self.domain
        return Loop(driver, self.predictor, self.ranker, domain, self.rec, self.llm_tiebreak, self.log, self.allowed_apps, self.stop, cache=self.cache)

    # -- the stall guard --------------------------------------------------------------------------------

    def new_request(self) -> None:
        """The user sent something new: any stall is over, and a question left unanswered is dropped."""
        self._stalled.clear()
        self.choices.clear()

    def blocked_message(self) -> str | None:
        """Why other tools must not act right now, or None. While a computer-use task is stalled or waiting for the user's pick, the agent used to work
        around it with scripts (wrong paths, retries): now the only way forward is the user's answer."""
        now = time.time()
        if any(now - t < STALL_HOLD_S for t, _ in self._stalled.values()) or self.choices:
            return ("NOT RUN: a computer-use task has stopped and is waiting for the user. Do not work around it with scripts or other tools. "
                    "Tell the user what happened in one or two sentences and wait for their answer.")
        return None

    def pending_choices(self) -> list[tuple[str, Choice]]:
        return [(pid, j.choice) for pid, j in self.choices.items() if j.choice]

    def _stall(self, key: str, text: str) -> str:
        self._stalled[key] = (time.time(), text)
        return text

    def _driver(self, target: str, app: str):
        """A driver for the target, or a RuntimeError. A target that just failed is not retried for a minute: the agent used to
        call it again and again, each time apologising at length, and that was the 'stuck' loop."""
        hit = self._unavailable.get(target)
        if hit and time.time() - hit[0] < 60:
            raise RuntimeError(f"{hit[1]} (still unavailable: do NOT call computer_use with target={target!r} again; use another method "
                               "such as mac_run, mac_notes_create or the researcher, and tell the user in one sentence)")
        try:
            return self.driver_factory(target, app)
        except RuntimeError as e:
            self._unavailable[target] = (time.time(), str(e))
            raise

    def _note(self, r: RunResult) -> None:
        for x in r.results:
            self.step_log.append({"step": x.step.text(), "status": x.status, "ms": round(x.ms), "tier": x.decision.tier if x.decision else "-",
                                  "target": x.cand.label() if x.cand else ""})

    # -- run -------------------------------------------------------------------------------------------

    def run(self, target: str, goal: str, max_steps: int = 12, app: str = "") -> str:
        from .focus import chat_out_of_the_way

        self.stop.clear()
        self.step_log = []
        if target == "desktop":
            from .. import app_actions
            from .apps import AppCatalog

            catalog = AppCatalog()
            found = catalog.resolve(app) if app else app_actions.find_app(goal, catalog)  # "Apple Music" -> "Music": the driver needs the real name
            if found is None and not app:  # without an app the driver looks at nothing ("Page: '' at ''") and the planner invents a website
                raise RuntimeError("target='desktop' needs app=<the Mac app's name>, and none could be found in the goal")
            app = found.name if found else app
        stalled = self._stalled.get(f"{target}:{app}")
        if stalled and time.time() - stalled[0] < STALL_HOLD_S:  # the same task, again, would only stall again: the user has to answer first
            return stalled[1] + " (This task is already stopped and waiting for the user: do not call computer_use again.)"
        driver = self._driver(target, app)  # raises RuntimeError with instructions if the surface is not available
        loop = self._loop(driver)
        t0 = time.perf_counter()
        plan = self.planner_factory(target)
        # The chat window is hidden while the task runs and shown again when it ends, however it ends (done, failed, needs you, stopped).
        from contextlib import nullcontext

        from .. import approvals

        with (chat_out_of_the_way(app) if approvals.current.windows else nullcontext()):
            obs = driver.observe()
            steps = plan(goal, obs, None, None, max_steps)
            job = Job("", target, goal, steps, None, loop)
            job.app, job.last_fp = app, obs.fingerprint
            return self._drive(job, plan, driver, t0)

    def _drive(self, job: Job, plan, driver, t0: float, resume: Pending | None = None, start: int = 0,
               forced: dict[int, tuple[str, str]] | None = None) -> str:
        loop, steps = job.loop, job.steps
        while True:
            r = loop.resume(steps, resume) if resume else loop.run(steps, start=start, forced=forced)
            resume, start, forced = None, 0, None
            self._note(r)
            job.done_texts += [x.step.text() for x in r.results if x.status == "ok"]
            self.last_run_ms = (time.perf_counter() - t0) * 1000
            timing = f"{len(job.done_texts)} steps in {self.last_run_ms / 1000:.1f}s"
            if r.status == "done" and r.more:  # the goal needs steps that depend on the screen as it is now: look, then plan the next chunk
                nxt = self._next_chunk(job, plan, r, driver)
                if isinstance(nxt, str):
                    return nxt
                steps = job.steps = nxt
                continue
            if r.status == "done":
                self._stalled.pop(f"{job.target}:{job.app}", None)
                self._learn_method(job, True)
                return f"DONE ({timing}). Now showing {describe(r.last_obs)}"
            if r.status in ("needs_user", "blocked", "stopped"):
                return f"{r.status.upper()}: {r.reason}. Page: {describe(r.last_obs)}"
            if r.status == "needs_approval":
                pid = r.pending.pid + f"-{len(self.jobs) + 1}"
                r.pending.pid = pid
                job.pid, job.pending, job.steps = pid, r.pending, steps
                self.jobs[pid] = job
                return (f"NEEDS_APPROVAL [{pid}]: {r.pending.description}. Reason: {r.pending.reason}. "
                        f"Ask the user; if they agree call computer_confirm('{pid}'), otherwise computer_cancel('{pid}').")
            # a step did not land, or could not be settled: the ladder. Rung 2 (one re-plan from the screen as it is now) unless this exact
            # failure was already met, which means the plan is not the problem.
            sig = (r.choice_step.text() if r.choice_step else "", r.reason)
            repeated = sig in job.seen_failures
            job.seen_failures.add(sig)
            obs = r.last_obs or driver.observe()
            if job.replans < MAX_REPLANS and not repeated:
                job.replans += 1
                try:
                    steps = plan(job.goal, obs, job.done_texts, r.reason)
                except ValueError as e:
                    return self._stalled_text(job, f"{r.reason}; re-planning failed: {e}", obs)
                job.steps = steps
                continue
            if r.status == "needs_choice" and r.choices:  # rung 3: ask, with the plausible actions
                return self._ask(job, r, steps, obs)
            return self._stalled_text(job, r.reason, obs)  # rung 4

    def _next_chunk(self, job: Job, plan, r: RunResult, driver) -> list[StepPlan] | str:
        """The next steps, planned from the screen as it is now, or the STALLED text when the goal has used up its budget or nothing is changing."""
        obs = r.last_obs or driver.observe()
        job.chunks += 1
        if job.chunks >= MAX_CHUNKS or len(job.done_texts) >= MAX_TOTAL_STEPS:
            return self._stalled_text(job, "the goal is not reached and this task has used up its step budget", obs)
        if obs.fingerprint == job.last_fp:
            return self._stalled_text(job, "the screen did not change after the last steps, so planning more would repeat them", obs)
        job.last_fp = obs.fingerprint
        try:
            return plan(job.goal, obs, job.done_texts, None)
        except ValueError as e:
            return self._stalled_text(job, f"planning the next steps failed: {e}", obs)

    def _learn_method(self, job: Job, ok: bool) -> None:
        """Remember that working the app's screen did (not) work here: the next request about this app can start with what worked."""
        if self.methods is not None and job.target == "desktop" and job.app:
            self.methods.record(job.app, "ui", ok)

    def _stalled_text(self, job: Job, reason: str, obs: Observation | None) -> str:
        self._learn_method(job, False)
        done = "; ".join(job.done_texts[-PLAN_HISTORY:]) or "nothing yet"
        return self._stall(f"{job.target}:{job.app}",
                           f"STALLED: {reason}. Done so far: {done}. Screen: {describe(obs)}. Do NOT retry this task and do not work around it with mac_run, "
                           "scripts or another tool: tell the user what was done and where it stopped, and ask how they want to proceed.")

    def _ask(self, job: Job, r: RunResult, steps: list[StepPlan], obs: Observation | None) -> str:
        self._cid += 1
        pid = f"c{self._cid}"
        described = [c.element.short() for c in r.choices]
        job.choice = Choice(pid, r.next_index, r.choice_step, [(c.element.role, c.element.label) for c in r.choices], described)
        job.steps = steps
        self.choices[pid] = job
        lines = "\n".join(f"{n}. {d}" for n, d in enumerate(described, 1))
        return (f"NEEDS_CHOICE [{pid}]: I could not tell which control the step {r.choice_step.text()!r} means. Screen: {describe(obs)}.\n"
                f"Options:\n{lines}\n0. none of these\n"
                "Ask the user exactly this question (which number?) and STOP. Their next message answers it and is handled for you: "
                "do not call computer_use again, do not use other tools.")

    def choose(self, pid: str, n: int) -> str:
        """The user picked option `n` (1-based; 0 = none of them). The chosen element is used for that step (still validated), then the goal continues."""
        job = self.choices.pop(pid, None)
        if job is None or job.choice is None:
            return f"FAILED: no pending question {pid!r}."
        ch, job.choice = job.choice, None
        if n < 1 or n > len(ch.options):
            return f"CANCELLED: stopped at {ch.step.text()!r}. Tell me what to do instead."
        self.rec.add(kind="human_choice", step=ch.step.text(), options=ch.described, chosen=ch.described[n - 1], chosen_index=n)  # training data for Laya
        self.stop.clear()
        driver = self._driver(job.target, job.app)
        job.loop = self._loop(driver)
        job.replans, job.seen_failures = 0, set()  # what follows gets its own ladder
        self._stalled.pop(f"{job.target}:{job.app}", None)
        return self._drive(job, self.planner_factory(job.target), driver, time.perf_counter(), start=ch.step_index,
                           forced={ch.step_index: ch.options[n - 1]})

    def answer(self, text: str) -> str | None:
        """Resolve the user's message as the answer to the newest pending question ("2", "b", "the second one", a label, "none"), or None if it is not an answer."""
        pending = self.pending_choices()
        if not pending:
            return None
        pid, ch = max(pending, key=lambda x: x[1].created)
        n = parse_choice(text, ch.described)
        return None if n is None else self.choose(pid, n)

    def confirm(self, pid: str) -> str:
        job = self.jobs.pop(pid, None)
        if job is None:
            return f"FAILED: no pending action {pid!r} (it may have been cancelled or already run)."
        self.stop.clear()
        driver = self.driver_factory(job.target, job.app)
        job.loop = self._loop(driver)
        plan = self.planner_factory(job.target)
        return self._drive(job, plan, driver, time.perf_counter(), resume=job.pending)

    def cancel(self, pid: str) -> str:
        return "CANCELLED." if self.jobs.pop(pid, None) else f"FAILED: no pending action {pid!r}."


_NONE = re.compile(r"^(0|none|none of (these|them)|neither|cancel|stop|never ?mind|no)$")
_ORDINAL = {"first": 1, "1st": 1, "a": 1, "second": 2, "2nd": 2, "b": 2, "third": 3, "3rd": 3, "c": 3}


def parse_choice(text: str, described: list[str]) -> int | None:
    """The option number a reply means (0 = none of them), or None when the reply is not an answer to the question."""
    t = " ".join(text.lower().replace("\u2019", "'").split()).strip(" .!?")
    if _NONE.match(t):
        return 0
    t = re.sub(r"^(?:the |option |number |choice |#)+", "", t)
    t = re.sub(r"(?: one| option)$", "", t).strip("() ")
    n = int(t) if t.isdigit() else _ORDINAL.get(t)
    if n is not None:
        return n if 1 <= n <= len(described) else None
    if len(t) >= 3:
        hits = [i for i, d in enumerate(described, 1) if t in d.lower()]
        if len(hits) == 1:
            return hits[0]
    return None
