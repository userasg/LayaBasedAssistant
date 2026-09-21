# Files + computer use (browser and desktop), combined spec and plan

Status: approved in chat ("do it all in one go"). Builds on phase 1 (`docs/laya_assistant_phase1.md`).
Inspiration (not followed strictly): Cua's "Jev, System One models, and the future of computer use" and the Jev-browser pattern: an LLM plans, a System 1 model decides among a menu the application built, the application validates, acts, then verifies. "A score is not proof the action worked."

## Decisions (from the brainstorm)

| Topic | Decision |
|---|---|
| Files | `~/LayaWorkspace` bind-mounted at `/workspace/shared`; file panel (list, preview, upload, download, rename, open in Finder); deletes go to `.trash` and ask first |
| Loop | LLM states intent, code resolves to a candidate menu, Laya scores + gates, code validates, acts, verifies; LLM is called for the plan and on hand-back only |
| Browser | The user's **real Chrome** through a Chrome extension ("Laya Bridge"), because Chrome 136+ ignores `--remote-debugging-port` on the default profile. The user claims tabs; only claimed tabs are controllable |
| Desktop | Any Mac app through Cua Driver (open source, MIT), accessibility tree + element indexes, background actions. Needs the user to grant Accessibility and Screen Recording |
| Safety | Real logins are in play: claimed tabs only, domain policy, first-visit approval, never type credentials, irreversible actions always ask, global Stop |

## Architecture

```
computer_use(target, goal)  [tool on the Deep Agent]
   planner (14B, once) -> [{do, target, value}] steps
   per step (no LLM):  observe -> menu (embedding shortlist) -> Laya decide -> validate -> act -> verify
   unsure / verify failed / unexpected page -> re-plan (LLM), max 2
   needs approval -> returns pending id; computer_confirm(pending id) is gated by HITL (user approves)
   every step -> recorder JSONL (training data + Laya-vs-LLM agreement)
drivers: PlaywrightDriver (tests, real Chromium) | BridgeDriver (real Chrome via extension) | DesktopDriver (Cua Driver)
```

Files: `src/laya_assistant/cua/{types,menu,decide,policy,loop,planner,recorder,tools,playwright_driver,bridge,desktop}.py`, `extension/` (MV3), `files.py`, file panel in `app.py`.

## Tasks (each ends with real tests; Laya is never faked)

1. **Files**: shared-folder mount, trash policy, `files.py` panel logic, UI panel.
2. **CUA core**: types, observation JS (single source shared by Playwright and the extension), menu, Laya decider, policy, verify, loop, recorder; tested on local HTML fixtures in real Chromium with real Laya.
3. **Chrome bridge**: MV3 extension, `BridgeServer`, `BridgeDriver`; tested by loading the extension into Playwright Chromium and claiming a tab.
4. **Desktop driver** over `cua-driver call`; tested against the real binary up to the permission boundary; live tests skip visibly until permissions are granted.
5. **Agent integration**: planner, `computer_use` / `computer_confirm` / `computer_cancel`, skill, cortex policy for host tools, session wiring, Stop.
6. **UI**: Computer-use panel (Chrome connection, claimed tabs, desktop status, Stop, domain approvals), step feed.
7. **Benchmark + docs + full suite**: per-step latency with Laya deciding vs LLM deciding every step, on the fixtures; docs; cleanup.

## Success criteria

- Files the agent creates in `/workspace/shared` appear in `~/LayaWorkspace`; deleting moves to `.trash`.
- On fixture pages, the loop completes forms/search/wizard tasks, stops at password fields and irreversible buttons, and re-plans when a step fails.
- Per step with Laya deciding: about 0.3 s (measured and printed); LLM-every-step baseline measured for comparison.
- The extension controls only claimed tabs in a real Chromium; Stop works.
- The desktop driver parses real Cua Driver output and reports its permission state; live desktop tests run once permissions are granted.

## Known risks

Laya's base accuracy at choosing (action 7/15, target 5/12) is why the LLM supplies the action and Laya ranks/gates targets, with embedding + LLM tie-breaks; Cua Driver is a nightly build; the Chrome extension must be loaded by the user; real-site behaviour will differ from fixtures.
