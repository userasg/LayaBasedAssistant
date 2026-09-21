# Laya Assistant, phase 1

A local voice-and-text assistant. Laya (a fast System 1 decision model) decides in ~50 ms how each request is handled; a Deep Agent (qwen2.5:14b) plans and works inside its own Docker sandbox. Everything runs on this machine: Ollama, Laya, MLX Whisper, Docker. Nothing is paid or cloud.

Spec: `docs/superpowers/specs/2026-09-20-laya-assistant-phase1-design.md`
Plan: `docs/superpowers/plans/2026-09-20-laya-assistant-phase1.md`

## Run it

```bash
cd test-project
uv run streamlit run src/laya_assistant/app.py --server.fileWatcherType none
```

Needs: Ollama running with `qwen2.5:14b-instruct`, `qwen2.5:3b`, `qwen3-vl:2b`; Docker Desktop (started automatically if closed); `ffmpeg`. The first start takes ~30 s (Laya loads, Whisper warms, the sandbox image builds once). The app tells you the exact `ollama pull` if a model is missing.

Tests: `uv run pytest tests/laya_assistant -q` (real Laya, real Docker, real Ollama, real Whisper, real Chromium; about 14 minutes now that the agent tests run a real 14B). No fake Laya anywhere (one approval test scripts only the chat model, to force a tool call). If the app is already running, add `VOICE_PORT=8775` so the tests do not fight it for the voice port. A few end-to-end tests depend on the 14B choosing to plan and can fail in a full run yet pass alone (`test_session.py` typed-answer tests); `test_cua_desktop.py`'s live Calculator test drives your real Calculator app once Cua Driver has permission.

## What you can do

- **Start from an example** on the first screen (search the web, make a note, look at your files, write and run code), or **type** in the chat box, or **speak**: press **🎙 Talk**, speak (a timer and a level meter show it is hearing you, your words appear live with Laya's routing decision forming under them), press **Stop**. What you said lands in an editable card above the chat box; press **Send** when it reads right. (Turn on "Send automatically when I stop speaking" under 🎙 Voice in the sidebar only if you trust the microphone: a lone word is never auto-sent.) The mic, the 📎 attach button and that card are pinned above the chat box, so they never scroll away.
- **Attach files/images** with 📎; they land in the sandbox under `/workspace/uploads/` and the agent can look at images with `describe_image` (qwen3-vl). Each attachment is sent once (the picker resets after a send).
- Simple turns ("hi", "capital of France?") take the **fast route**: a 3B model streams the answer in ~0.1 s, no agent. A request that names a file, a path or a URL, or asks to create/run/open/delete something, never takes it (see "Problems found").
- Anything else goes to the **Deep Agent**: it writes a plan (held for your approval only if you turn that on), then works in the sandbox with subagents. Every shell command passes Laya's safety gates.
- **One answer per request.** Your message appears the moment you send it. While the assistant works, a live card says what it is doing right now ("Searching the web…"), ticks the plan off, shows the last finished steps with their timings, and has a **Stop** button. When it finishes, the card becomes ONE answer with a quiet line under it (`🧠 agent · Laya 61 ms · 4 steps · 38 s`) and **Show work** folded beneath (the plan and every step with its result; failures are marked). Before, every "I'll now search..." the model said became its own chat bubble.
- **Questions for you** (a plan, a risky command, a file to delete, an irreversible step) are one highlighted card with Approve / Reject; you can equally type *yes*, *no* or what to change in the chat box.
- **Files** (button, top right) opens the shared-folder view: a table, one file previewed at a time, rename / download / trash / restore. **← Chat** returns.
- The sidebar has two rows of badges (sandbox, Laya device, model; then Laya / LLM / first-word / STT timings), one line saying what the assistant will ask you first, and folded sections: Autonomy, Voice, Computer use, Activity log (last 25 steps, each tagged 🟢 Laya, 🔵 LLM, ⚙️ code, 🛠 tool, 🎙 STT with its milliseconds), Models and paths. The chat shows the last 8 turns; older ones fold away.

## Architecture

```
mic ─► live STT (WebSocket, partial text, Laya intake preview) ─┐
typed text / files ────────────────────────────────────────────┴► LAYA INTAKE (~55 ms)
     fast route (simple, no risk, no plan) ─► qwen2.5:3b streams the reply
     executor route ─► Deep Agent (qwen2.5:14b)
           write_todos ─► [you approve/edit the plan] ─► subagents: researcher · coder · reviewer · laya-advisor
           backend: /workspace = Docker sandbox · /memories = disk · /skills = read-only
           every model/tool call ─► LAYA CORTEX (hard-deny code, provenance, destructive gate, repeat guard, log)
```

| File | Purpose |
|---|---|
| `engine.py` | the one GPU lock, Laya wrapper, Ollama factory, model checks |
| `intake.py` | Laya intake + routing (intent, needs_plan, risky, simple, factual) |
| `cortex.py` | model/tool middleware: gates, plan-first, nudges, retry, timing |
| `policy.py` | path policy for file tools (writes only in /workspace, /memories) |
| `agents.py` | wires the Deep Agent (all features except async subagents) |
| `session.py` | UI-free turn routing, streaming events, resume |
| `sandbox.py` | per-session Docker container (host-shell fallback) |
| `stt.py`, `voice_server.py`, `voice_component/` | speech to text, live WebSocket streaming, browser mic component |
| `context_config.py`, `memory.py` | 16k-window context management, persistent memory |
| `advisor.py`, `tools.py`, `skills/` | Laya advisor subagent, tools, SKILL.md files |
| `transcript.py` | pure functions that fold the message history into turns (prompt, one answer, steps, plan, timings) and summarise the live feed; unit-tested without Streamlit |
| `review.py` | the one protocol for every question put to the person (plan, risky command, delete, irreversible step) and how their answer is read |
| `host.py` | Mac scripts (`mac_notes_create`, `mac_run`); `LAYA_HOST_ACTIONS=off` makes them inert (all test runs) |
| `app.py`, `ui/` | Streamlit UI: `app.py` lays the page out; `ui/chat.py` (turns, live card, approval card, first screen), `ui/composer.py` (mic, attach, review card), `ui/sidebar.py`, `ui/files.py`, `ui/style.py` |

## Measured on this machine (M4 Max)

| Thing | Measured |
|---|---|
| Laya intake, warm | 54-63 ms (110 ms average inside Streamlit) |
| Fast route, "hi there" | first token 31-63 ms, whole reply ~90-133 ms |
| Laya advisor subagent | 22 ms |
| Docker `execute`, warm | ~55 ms |
| Speech to text, short clip, warm | 406-550 ms (Whisper large-v3-turbo on the GPU) |
| Live speech | first partial words ~0.9 s into speaking; final text 0.2-0.5 s after you press stop |
| Code task, 14B, 2-step plan | ~45 s end to end including your approval |
| Laya accuracy (base checkpoint) | fast-route: 10/14 simple turns routed fast with 0/26 wrong routes; intent 47% overall but 14/17 when confident |

## Problems found in real use, and what was done

Reported after the first real-microphone session: junk transcripts, a bloated page, and "stuck as the conversation grows". Each was measured before being fixed.

| Symptom | Measured cause | Fix |
|---|---|---|
| "Thank you." / "No." sent as requests | Whisper answers "Thank you." to **room noise** (pink noise, `no_speech_prob` 0.00 after level boosting), identical to real speech; RMS cannot separate quiet speech (0.0024) from noise (0.0039) | A voice-activity check *before* Whisper: speech has a large energy swing between syllables and pauses and >= 0.15 s of frames well above the noise floor; noise, hum and clicks fail it. Voice now defaults to review-before-send; a lone word is never auto-sent |
| Page slower with every file the agent makes | Every rerun read every sandbox file (`docker exec cat`, ~75 ms each): 99 ms -> 751 ms with 10 files | Files are fetched once (cached by path and size); the sidebar lists sizes only |
| Bloated page | A full log table, 5 metric tiles, mic, transcript box, uploader and the whole chat, redrawn on every event | Compact sidebar (stats, folds, capped log), chat window of 12 with an "earlier messages" fold, uploader in a popover, transcript shown once |
| "Stuck" / turns aborted | The turn ran *inside* the Streamlit script, so any click, toggle or mic event aborted it mid-run and could leave the GPU lock held across a `yield` | Turns run on a **worker thread** that always finishes; the page polls its feed (0.4 s) and shows elapsed time. Tests prove an abandoned turn never strands the lock |
| A 23 s wait for a vague "No." | Not reproducible in isolation (1.5 s, 46 streamed tokens); the backend does not slow as the chat grows (16 mixed turns: 0.2-2.0 s each, state ~650 tokens, `get_state` ~1 ms). It was a single long generation the UI could not show | Live progress with a timer and "waiting for the model"; cause of that one call not isolated |

### Second round: the UX pass, and what testing it turned up

The first end-to-end run worked, but reading it was hard. Each row below was found by looking at the real app in a browser or by reading the code path while redesigning it.

| Symptom | Cause | Fix |
|---|---|---|
| A wall of assistant bubbles for one request | The history drew every AI message that had text, so the model's narration between tool calls ("I'll now search...") became its own bubble | `transcript.build_turns` folds the history into turns: ONE answer (the last plain reply), the narration, tool calls, results and plan kept under **Show work**. Unit-tested on realistic message sequences |
| Your message did not appear until the answer did (fast route) | The fast route wrote the exchange into the thread only when it finished, and the page draws from the thread | The session knows the request's id and text the instant it is submitted; the live view shows it at once. Stopping a fast turn half way still records the exchange |
| **Risky-command and delete approvals could not work in the real app** | Those two gates asked with a flat payload (`{"tool", "command"}`) and read the answer as the bare string "approve", but the UI answers with `{"decisions": [...]}` (the Deep Agents protocol the plan gate already used). The card indexed `pending["action_requests"]` (a KeyError for them) and any answer arrived as a dict never equal to "approve", so every such request was silently a rejection. Found by reading the code while building the card; the old unit tests patched `interrupt()` with plain strings, which is why nothing caught it | `review.py`: one request shape and one reader (`decision()`: anything that is not a clear approval is a rejection; a typed non-yes answer reaches the agent). Proven through a real LangGraph pause and resume with real Laya (`test_review.py`); the card now draws plans, commands, deletes and computer-use steps |
| "Plan first, then create greeting.md ... one friendly line" was answered by the tool-less 3B model, which only described the command | Laya scored `simple` 0.63 on the word "greeting" (intent chat), so the route was fast | An exact veto for requests that name a file, path or URL, or ask to create/run/open/delete something: it can only turn a fast route into an executor route (worst case a slower correct answer). The four example prompts and the 40-utterance probe set still route as before |
| The attached files rode along with every later message | The file picker kept its files after a send | The picker gets a new identity after each send |
| The autonomy switches sometimes dropped a click | Unkeyed widgets whose default is read from the settings get a new identity every time the setting changes | Keyed widgets writing to the settings from a callback |
| The mic scrolled away in a long chat | It sat at the end of the page | It, the attach button and the review card are pinned above the chat box |
| Real Notes filled with test notes; a duplicate note; a note with a figure nobody had sourced | End-to-end tests let the real 14B pick `mac_notes_create` against the real Notes app; a re-planning agent re-created the same note; the 14B wrote a number it had not read anywhere | `LAYA_HOST_ACTIONS=off` in every test run (an explicitly passed runner still works); the same note (title and text) is created once per ten minutes; the note tool's own description says: only when asked, never twice, no unsourced figures. A general version of those rules in the system prompt was tried and **removed**: in one A/B run of `test_session.py` each, the agent tests passed 17/17 without it and failed 4 with it (planning tests that timed out or never created the file). One sample each on a nondeterministic 14B is not proof, but the rule was only ever a request to the model, so it was not worth the risk. The figure problem is therefore only partly addressed (the note tool's wording), not solved |

Measured while doing this: Laya's intake shows ~340 ms per request inside the app on a busy machine (load average about 6) against 75-100 ms in a standalone script, every turn and not only the first; the safety gate's first call after start-up costs ~300 ms against ~20 ms after, which start-up now pays once. Why the intake is slower in the app is **not** explained.

## What the measurements changed (deliberate deviations from the spec)

- **`authorized` ("did the user ask for this?") is log-only.** Base Laya rated an injected `curl ... | sh` as 0.92-1.00 authorized. Injection defence is exact code instead: pipe-to-shell/upload regexes and a provenance check (a command that appears in tool output but was never typed by you is blocked).
- **Failure-kind classification is log-only.** It called "command not found" a runtime error at 0.78. Both signals are recorded so a Laya fine-tune can earn them.
- **The `simple` intake question needed different wording.** The abstract phrasing separated nothing; naming the categories ("a greeting or small talk", "a trivia or general knowledge question") did.
- **The rubric grader is bounded and flaky.** `RubricMiddleware` is beta; with qwen 14B its structured output sometimes never completes (it looped ~20 calls with no verdict). It is capped at 4 grader calls and ends as `grader_error` instead of hanging; sometimes it returns `satisfied`. Rubric criteria come from Laya's intent (code/research), not from the plan text. The spec's "Laya sufficiency check before the grader" was not built: it is unmeasured.
- **`permissions=` is not supported with a command-executing sandbox** (Deep Agents raises). The same rules are enforced by `PathPolicyMiddleware`; the container is the boundary for the shell.
- **`create_deep_agent` does not include the planning tool** in 0.7.14 (tutorials add `TodoListMiddleware` explicitly); `FilesystemMiddleware` cannot be reconfigured, so tool-result offloading at 3k tokens is a small extra middleware in front of it, and the stock summarizer is replaced by a 16k-sized one.
- **Small-model behaviours are handled in the cortex:** limiting the first call of a planning turn to `write_todos` (with a nudge and a fallback), nudging an agent that stops with open todos, retrying Ollama's "token repeat limit" abort.
- **Voice** (see the table above for the noise fix): live streaming (browser mic -> local WebSocket -> rolling Whisper decode). Browser speech APIs were rejected: they send audio to Google/Apple. Quiet microphones are handled (gain normalisation; a 1%-volume clip that used to return nothing now transcribes), language is pinned to English (set `STT_LANGUAGE=` to auto-detect).

## Known limits

- One model call at a time, process-wide (`ENGINE` lock): also what prevents the Apple GPU crash and Ollama pile-ups. Parallel subagent fan-out is therefore sequential; `ENGINE_LLM_PARALLEL=1` allows concurrent LLM requests (Laya and Whisper stay serialized) but is untested.
- Code interpreter and dynamic subagents are wired in (`CodeInterpreterMiddleware`) but not exercised by a test; they matter in phase 2.
- Laya's `risky` intake score misses consequential non-destructive actions (send email 0.02, pay invoice 0.16). Per-tool gates cover shell commands; host-touching tools (phases 2-3) will need their own gates and probably a fine-tune.
- The 14B model sometimes narrates instead of acting or repeats a sentence in its final reply.
- **Stop** ends a request at its next event, and a model call already in flight has to finish first (a 14B call can take several seconds), so the turn ends a moment after you press it, not at once. A running computer-use goal halts before its next step.
- The chat box does not take attachments itself (Streamlit's own AppTest harness cannot drive `chat_input(accept_file=...)`, which would leave the main input untestable), so attaching is the 📎 button beside it.

## Ready for phase 2 (browser operator)

Reusable as-is: `engine`, `decisions`, `cortex` (host tools already support the ask-a-human path via `host_tools=`), `policy`, `session`, the decision log, the Streamlit shell.
Phase 2 needs: Playwright (browser control), `sentence-transformers` + `bge-small` (embedding shortlist of page elements, as in Laya's `cua_test.py`), an observe -> shortlist -> Laya `{action, target, destructive, needs_user}` -> act loop with the LLM deciding what to type, and a measured accuracy pass on real pages (Laya's own CUA test: action 7/15, target 5/12, needs_user 15/17, destructive 12/17 on the base checkpoint, so Laya gates and the LLM drives until a fine-tune earns more).
