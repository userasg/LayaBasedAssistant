# Laya Assistant: files and computer use (browser + Mac apps)

Adds to phase 1 (`laya_assistant_phase1.md`): a shared files folder, an agent that operates **your real Chrome** and **Mac apps**, actions that start while you are still speaking, and a measured path for Laya to earn more control.
Spec: `docs/superpowers/specs/2026-09-20-computer-use-and-files-design.md`. Inspiration: Cua's "Jev, System One models, and the future of computer use" (not followed strictly).

## Setup: nothing for the common cases

The agent has three ways to act, simplest first, and the first two need **no setup at all**:

1. **Scripts on your Mac** (`mac_notes_create`, `mac_run`): open an app or a page, make a Note (only when you ask for one, and the same note is never created twice within ten minutes), read a web page with `curl`, run `osascript`. Set `LAYA_HOST_ACTIONS=off` and both do nothing: every test run does, so tests can never touch your real Notes or open windows. Obviously safe commands (open, a curl GET, date, say, a short ls) run at once; anything else asks you first and you answer in the text box; the hard-deny list (sudo, `rm -rf /`, pipe-to-shell, uploads) is refused. The shell gets a minimal environment (no API keys) and a 30 s timeout. The Docker sandbox cannot do this (it is walled off from the Mac on purpose), so these are separate host tools.
2. **A browser window the app opens and owns** (`computer_use`, target `browser`, the default): a visible Chrome window with its own saved profile in `~/.laya_assistant/browser`, so logins persist after you sign in once. It sees the real page DOM, so it needs no extension, no accessibility setting and no macOS permission. It starts on a blank page, so the goal says where to go. Blocked sites (banks, payments...) still stop it; first-visit questions are skipped there because it holds none of your logins.
3. **Your own Chrome or a Mac app's buttons** (`computer_use` target `my_chrome` / `desktop`, through Cua Driver): needs macOS's one-time consent (the panel has a **Grant permissions** button; switch on CuaDriver under Privacy & Security -> Accessibility and Screen Recording). The app starts the driver's service itself. **Measured live: Chrome does not expose a page's content through the accessibility tree unless accessibility is enabled inside Chrome** (238 browser-chrome elements, zero page elements, even after 12 s), so `my_chrome` through Cua Driver only works after `chrome://accessibility` -> *Native accessibility API support*. That is why the owned browser is the default. The optional Laya Bridge extension (`src/laya_assistant/cua/extension`) is the other way into your own Chrome.

Cua Driver telemetry is switched **off** by the app (`cua-driver telemetry disable`); the binary is checksum-verified in `~/.laya_assistant/cua-driver/`, nothing in `/Applications`.

If a target is unavailable the agent is told once, and the same call is not repeated for a minute: it must pick another method and say so in one sentence. That is what fixes the "stuck in a loop" behaviour.

## Answering approvals

Every question (a plan, a risky command, a file to delete, an irreversible step) is one highlighted card with Approve / Reject, and the text box works while it is pending: type **yes** / **no**, or tell it what to change ("yes, but skip step 2" goes to the agent as your reply, never misread as a plain yes). Anything that is not a clear yes counts as a no: an empty or malformed answer can never approve a command. (Risky-command and delete approvals used a different answer format from the plan gate and could not be answered at all in the real app until this was unified in `review.py`; see `laya_assistant_phase1.md`.) A plan is asked about only when the plan itself changes, never for progress updates, and at most 3 times per request, so it cannot loop. If you decline a computer-use step the agent is told to cancel and stop.

## What you can say or do

- "Use my browser: search the shop for sony headphones." The agent calls `computer_use` once; it plans, acts and checks on its own. Results: **DONE**, **NEEDS_USER** (password/code/captcha: you do it, then it continues), **NEEDS_APPROVAL** (an irreversible step: an approval card describes it; only your OK runs it), **BLOCKED**, **FAILED**.
- "Open Notes and write a shopping list": say it while holding the mic open. **"Open Notes" runs as soon as that clause has stopped changing and been followed by "and"/"then"/a comma**, while you are still talking; the rest ("write a shopping list") goes through the normal plan-and-approve path when you stop. Only reversible commands run early: open an app, go to a site you already approved, scroll. Power apps (Terminal, Automator, Keychain, System Settings...) never open early. Clauses commit strictly in order, so an unrecognised first clause holds everything back.
- **Files**: `~/LayaWorkspace` is mounted in the sandbox as `/workspace/shared`. Files the agent makes appear in Finder; files you drop there (or upload in the 📁 Files panel) are visible to it. The panel lists, previews (text, markdown, CSV, images), downloads, renames and opens in Finder. Deleting (by you or the agent) moves the file to `.trash` and is restorable; the agent must ask first and shell `rm` in that folder is blocked.
- **Stop** (on the live card while a request runs) ends the request at its next event and halts a running computer-use goal before its next step; a model call already in flight finishes first. **STOP everything** (Computer use panel) also releases every Chrome tab.
- The mic button is Start listening / Stop listening, as before.

## How the fast loop works

```
computer_use(goal) -> planner (LLM, once) -> steps in words: [{do, target, value}]
per step, no LLM in the common case:
  observe (DOM/AX text) -> menu (lexical + embedding shortlist, cached) -> decide -> validate -> act -> verify
    decide:  code (unambiguous, ~0 ms) | Laya (ambiguous: target + irreversible/needs_user/blocked in ONE pass, ~55 ms) | LLM tie-break (rare)
    validate: domain policy, no secrets typed, irreversible steps ask
    verify:  the NEXT observation checks the last action landed (one observe per step, not two)
  unsure or failed verification -> back to the planner (max 2 re-plans)
  irreversible -> parked as NEEDS_APPROVAL; computer_confirm is human-in-the-loop gated by the framework, so the loop is never re-run on resume
every step -> ~/.laya_assistant/cua_steps.jsonl (training data + agreement stats)
```
Speed techniques: consecutive form-filling steps share one observation; element embeddings are cached; the loop never calls the LLM unless Laya and the embeddings disagree.

## Measured (this machine; fixture pages in real Chromium, real Laya, real 14B)

| Thing | Result |
|---|---|
| Fast loop | **~87 ms per step** (11 steps: 10 by code, 1 by Laya, 0 LLM); 15-20 ms per step in the agent run through the extension |
| Same steps, LLM picks from the same narrowed menu | 160-295 ms/step (1.8x-3.4x slower; varies with the model's cache state) |
| Same steps, LLM shown the whole page (conventional agent) | ~351 ms/step (4.0x slower) |
| Real extension in real Chromium | 8 steps, all correct; a wrong token is refused; unclaimed tabs cannot be observed or acted on |
| Agent -> `computer_use` -> extension -> tab | search completed in ~18 s including the plan approval |
| Early action while speaking | "open notes" fired ~0.8 s after the clause ended, with speech still to come |
The gap is modest on small pages because a warm local 14B is fast; the loop's cost does not grow with page size and the whole-page LLM's does, but only what was measured is claimed here.

## Laya's role, and why it advises before it decides

You suggested letting Laya pick tools and judge completion. I measured that instead of assuming it (base checkpoint):

| Decision | Result |
|---|---|
| Which tool fits a request (browser/desktop/files/code/research/none) | 12/20 (9/20 with my first wording); 8/11 when confident |
| Has the goal been achieved (from the page state) | 5/10 either wording; scored an empty contact form "done" at 0.87 |
| Laya's own CUA test: which action / which element | 7/15, 5/12 (`typed-decisions` 8/12) |
| needs_user / destructive | 15/17, 12/17 |

So: the LLM chooses tools (Laya only suggests: "Laya suggests the browser target"), code verifies completion, and Laya ranks/gates targets in a blend with embeddings. **`cua/authority.py` is the mechanism for changing that**: every ambiguous step records Laya's pick, the final pick, and whether the action verifiably landed. Laya may decide alone at a confidence level only after 30+ recorded steps at >= 90% accuracy *in that band* (high-confidence samples never vouch for lower bands). The panel shows where it stands. A fine-tune on the recorded steps is the way to move those numbers.

## Safety model (real logins are in play)

- With the optional extension: only claimed tabs; the extension refuses everything else; token- and origin-authenticated localhost socket; one-click release.
- Never-automate domains (banks, payments, password managers, account security) are blocked outright; the first visit to any other site asks (localhost pre-approved; your additions in `~/.laya_assistant/policy.json`).
- The agent never types credentials: password/secret/card fields hand control to you (enforced in code, in the policy, and again inside the page script). Captchas hand over.
- Irreversible steps (send, pay, buy, delete, post, publish, transfer, cancel...) always ask, from label rules plus Laya's second opinion.
- Page text is untrusted: commands that only appear in tool output are blocked (phase 1's provenance check).
- Matching floor: a step with no good match fails instead of clicking the least-bad element.

## What is and is not verified

Verified by tests (real components; only the third-party daemon is stood in for by documented payloads, and the app launcher by a recorder): the loop on 8 fixture pages including irreversible/password/captcha/dialog cases; the planner (real 14B); the extension in real Chromium; the agent through the extension; streaming early actions with real speech; the shared-folder mount and trash policy in real Docker; the UI in a real browser.

**Not verified, please try:** your real Chrome (I used Chromium with the same extension); the desktop operator against live apps (needs your permission grants: the driver is built against the real tool schemas and its status/permission handling is tested, live tests skip until granted); real websites (only local fixture pages). Cua Driver is a nightly build.

## Bugs found by testing, and fixed

| Found | Fix |
|---|---|
| **Navigation was not policy-checked by destination.** A plan could walk a browser to any site (a real run went to google.com unasked); only the page you were already on was checked | `validate` now judges a `navigate` step by where it goes: never-automate sites block, unknown sites ask |
| **The microphone WebSocket had no authentication**: any web page you visit can open a socket to localhost and stream audio ("open calculator") | Origin check on the voice and bridge sockets: only pages served from this machine (and the extension itself) may connect; the bridge also needs its token |
| **The least-bad element was clicked.** "click the first result link" pressed a "Search" *button* because the labels overlap, and the URL changed so verification passed | Role words in the step ("link", "button", "box", "checkbox"...) now count: a mismatched role is heavily penalised, so the step fails and is re-planned |
| Laya's early "authority" rule let high-confidence samples vouch for lower bands | Each band now needs its own evidence |
| Playwright's event loop broke WebSocket tests when the whole suite ran (each module passed alone) | One shared Playwright per run; async test clients run in their own thread |

## Known limits

- Clicks are page-script clicks (no debugger banner); a few sites that require trusted events may ignore them.
- Screenshot-only surfaces (canvas apps, some Electron apps with empty accessibility trees) are not handled yet; the article's OmniParser-style parsing is the natural next step.
- Early-command recognition covers a small vocabulary (open/launch/switch to, go to <site>, scroll); everything else waits until you stop.
- The 14B planner sometimes takes shortcuts (e.g. navigating straight to a results URL); the verification step still checks the outcome.

## Opening and driving any Mac app (measured on the real Music and Calculator apps)

Why "open Apple Music" used to fail: Cua Driver's `launch_app` deliberately does not activate the app, so its window is reported off screen, and a window on another desktop (Space) cannot be read (`ax_unresolved`, empty element tree). The agent only ever heard "has no visible window" and guessed other app names.

- `cua/apps.py`: `AppCatalog` finds an installed app by what people call it (aliases such as "apple music" -> Music, then fuzzy match; ~ms, no model), `bring_up` opens it the way a person does (`open -a`, then AppleScript `reopen` + `activate`, so macOS switches to its Space) and waits for a window the driver can read, and when it cannot it says WHY (other desktop / hidden / no window) so the agent stops guessing. After this the driver read 239 controls in Music and 165 in Calculator.
- **Typed "open X ..." opens the app at once**, before any model plans (`session._open_first_clause`); "open X" alone is answered by code in ~1 s. Power apps (Terminal, Automator, System Settings...) are never auto-opened.
- `mac_open_app` is a tool for the agent; it also reports whether the app has a scripting dictionary (then AppleScript via `mac_run` is the most reliable way to act inside it, and works across Spaces).
- Not built yet: a vision fallback (screenshot + qwen3-vl) for apps with empty accessibility trees, and auto-approval of simple AppleScript (`mac_run osascript` still asks each time). "All apps" is a ladder, not a guarantee.

## Working in other apps: what was measured and fixed (Music, Notes, Finder, Calendar, Calculator, TextEdit, Reminders, Maps)

- **Windows and focus.** New setting **Show app windows while working** (Autonomy, default on). On: the app comes forward and the window that holds the chat (whatever app is in front when a task starts) is hidden until the task ends, however it ends (`cua/focus.py`; measured: Chrome hidden for a 16 s Calculator task, then shown again). Off: apps are only launched (`open -g`), nothing is raised, you stay in the chat; an app on another desktop is then reported, not dragged forward.
- **Unnamed rows.** Lists in Music, Notes, Finder and TextEdit expose actionable rows with no label and no labelled child. Apple's on-device OCR (`cua/ocr.py`) reads the words off the window screenshot and puts them on the elements they sit inside (frames are screen points; the screenshot is window-relative and scaled).
- **The driver refuses a bare element index** and answers with a refusal object, which used to count as success. Clicks now use element tokens and refusals are failures. If a click changes nothing: one patient re-look, then a position click, then the same with the window briefly fronted (the driver's documented ladder).
- The loop now sees changes that are not controls (a calculator's display) by fingerprinting the window's text and OCR words; the closed menu bar is ignored; names like "Apple Music" resolve to the installed app; Laya's irreversible score needs 0.9 in native apps (it rated "=" 0.72).
- The planner's JSON is repaired (trailing commas, comments, single or smart quotes) and retried three times.

## Context between requests (`scope.py`)
The saved thread is untouched (the chat is drawn from it), but the model is sent only the current request plus a recap of the last four requests ("request -> result", one line each, marked as not to be redone). Before, every request carried all earlier tool calls, plans and narration, the previous request's todo list stayed in state (so planning was skipped), and the 14B mixed tasks up. Measured on two real consecutive requests: the saved thread had 24-44 messages, the model was sent 1-5, and todos were reset to a fresh plan.
