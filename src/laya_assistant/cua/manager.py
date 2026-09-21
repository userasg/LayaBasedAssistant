"""ComputerUse: runs a goal end to end and is what the agent's tools call.

  plan (LLM, once) -> fast loop (no LLM) -> on a failed/unsure step: re-plan from the current page (max 2) ->
  on needs-approval: park the exact action and return; the user's OK (a HITL-gated tool) executes that action.
Every outcome is text the agent can relay: done / needs_user / needs_approval / blocked / failed / stopped.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .loop import Loop, Pending, RunResult
from .policy import DomainPolicy
from .recorder import StepRecorder
from .types import Observation, StepPlan

MAX_REPLANS = 2


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


def describe(obs: Observation | None) -> str:
    if obs is None:
        return "(no page)"
    return f"'{obs.title}' at {obs.url or obs.app}: " + (obs.text[:220] or ", ".join(e.label for e in obs.elements[:6]))


class ComputerUse:
    def __init__(self, predictor, ranker, planner_factory, driver_factory, domain: DomainPolicy | None = None,
                 recorder: StepRecorder | None = None, llm_tiebreak=None, log=None, allowed_apps=None, own_domain: DomainPolicy | None = None):
        self.predictor, self.ranker, self.planner_factory, self.driver_factory = predictor, ranker, planner_factory, driver_factory
        self.domain = domain or DomainPolicy()
        self.own_domain = own_domain  # policy for the agent-owned browser
        self._unavailable: dict[str, tuple[float, str]] = {}
        self.rec = recorder or StepRecorder()
        self.llm_tiebreak, self.log, self.allowed_apps = llm_tiebreak, log, allowed_apps
        self.stop = threading.Event()  # the global Stop: checked before every step
        self.jobs: dict[str, Job] = {}
        self.last_run_ms = 0.0
        self.step_log: list[dict] = []  # for the UI feed

    def _loop(self, driver) -> Loop:
        domain = self.own_domain if (getattr(driver, "own", False) and self.own_domain is not None) else self.domain
        return Loop(driver, self.predictor, self.ranker, domain, self.rec, self.llm_tiebreak, self.log, self.allowed_apps, self.stop)

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
        if target == "desktop" and app:
            from .apps import AppCatalog

            found = AppCatalog().resolve(app)  # "Apple Music" -> "Music": the driver needs the real name
            app = found.name if found else app
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
            job.app = app
            return self._drive(job, plan, driver, t0)

    def _drive(self, job: Job, plan, driver, t0: float, resume: Pending | None = None) -> str:
        loop, steps = job.loop, job.steps
        total = 0
        while True:
            r = loop.resume(steps, resume) if resume else loop.run(steps)
            resume = None
            self._note(r)
            total += len(r.results)
            job.done_texts += [x.step.text() for x in r.results if x.status == "ok"]
            self.last_run_ms = (time.perf_counter() - t0) * 1000
            timing = f"{len([x for x in r.results if x.status == 'ok'])} steps in {self.last_run_ms / 1000:.1f}s"
            if r.status == "done":
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
            # failed: hand back to the planner (the LLM), from where the page actually is
            if job.replans >= MAX_REPLANS:
                return f"FAILED: {r.reason}. Page: {describe(r.last_obs)}"
            job.replans += 1
            obs = r.last_obs or driver.observe()
            try:
                steps = plan(job.goal, obs, job.done_texts, r.reason)
            except ValueError as e:
                return f"FAILED: {r.reason}; re-planning failed: {e}. Page: {describe(obs)}"
            job.steps = steps

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
