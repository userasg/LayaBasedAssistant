# Bounded task runner: no improvising, flat per-step cost

Problem (measured on a real VS Code run and in OSWorld-Human): a failure in `computer_use` went back to the outer 14B agent, which improvised
with `mac_run`/`osascript` (wrong paths, retries, interrupt errors). Cost per step grew because history grew, and every re-plan was another LLM call.

Research basis: OSWorld-Human (each later step up to 3x slower; planning/reflection 76-96% of latency; grounding stays flat because it only sees the
current screen), Cua's System One model (closed option scoring, code owns order and gates), Jev (calibrated confidence drives escalation),
Stagehand (cache the resolved action, self-heal only when it stops resolving), CoAct/Agent S3 (script vs UI is a first-class choice).

## Decisions
1. **Failure ladder, in code.** re-observe once -> re-plan once (LLM, sees the current screen) -> **ask the user with 2-3 concrete options** (Laya's top
   candidates) -> `STALLED` report. Nothing hands control back to free-form retrying. A repeated identical failure skips the re-plan.
2. **Ask with choices.** `NEEDS_CHOICE [id]` lists the options; the user's next message ("2", "b", a label, "none") is resolved *in code* by the session
   (no model turn) and the chosen element is forced for that step (still validated: policy, secrets, irreversible).
3. **Stall guard.** After STALLED / NEEDS_CHOICE the same target is refused for 90 s and `mac_run` is refused until the user answers or sends a new
   request: the agent cannot work around a stall.
4. **Decision cache** (Stagehand-style): a verified Laya/LLM/human choice is stored as (site or app, step wording) -> (role, label). A later identical
   step replays it if that element still exists (safety pass still runs); a cached choice that stops working is evicted.
5. **Fixed-size context.** The planner sees the goal, the last 6 finished steps and a 30-element screen; never the whole history.
6. Every human choice is recorded (`kind=human_choice`) as training data for Laya.

## Deferred (not built here)
Milestone `done_when` checks written by the planner and verified in code; direct (non-UI) actions such as `code <dir>` / file creation offered in the same
menu as UI elements; vision fallback.
