# Laya Assistant, Phase 1: voice + text agent, full Deep Agents kit, Laya as System 1

Status: draft v2 for review. Scope: phase 1 of 4 (core chat). Phases 2-4 are listed at the end and get their own specs.

## Goal

A local, free, GPU-accelerated chatbot you can talk to or type to, as fast as this machine allows. Laya works out what you want and steers the run; a Deep Agent, given the full Deep Agents toolkit (its own sandbox filesystem, planning, subagents, memory, skills, rubric grading), plans, gets your approval, and executes. Everything runs locally: Ollama for LLMs, Laya for fast decisions, MLX Whisper for speech-to-text, Docker for the sandbox.

Non-goals for phase 1: computer use (phases 2-3), text-to-speech, multi-user, auth, any cloud service, async subagents (they need the LangGraph server runtime, the same reason tutorial 16 skipped them).

## Decisions already made

| Topic | Decision |
|---|---|
| Architecture | Laya-gated pipeline around a Deep Agent (approach A) |
| Front end | Streamlit, record-then-send voice (`st.audio_input`) |
| Autonomy | Plan preview and approval once, then autonomous; Laya gates risky steps |
| Models | `qwen2.5:14b-instruct` executor, `qwen2.5:3b` fast path (below), `qwen3-vl:2b` vision, `whisper-large-v3-turbo` (MLX) STT, Laya `convaiinnovations/laya` |
| Agent workspace | Its own Docker sandbox filesystem (Docker Desktop is installed and running) |
| Deep Agents scope | Everything except async subagents |
| CUA scope (later) | Browser + real desktop |

## Evidence the design rests on (measured this session)

- Laya decisions cost ~30-60 ms; an LLM call costs 1-30 s. That gap is the whole speed strategy.
- Base Laya is weak at open choices zero-shot (Laya's CUA test: action 7/15, target 5/12 base; 7/15, 8/12 `typed-decisions`) but decent at safety-style yes/no (`needs_user` 15/17, `destructive` 12/17 base). Wrong answers usually carry low confidence, so confidence bands work.
- MPS is not thread-safe (concurrent Laya calls abort the process) and the GPU is shared with Ollama. Concurrent LLM requests piled up on the shared GPU and stalled. So model calls are serialized.
- The agent needs a repeat-failure guard and a step budget (a real run repeated one failing command 31 times without them).
- Open item: the first Laya call in a Streamlit script thread took ~5 s once (later ~50 ms). The plan must measure and mitigate it.

Tutorial 19's cortex middleware, lock and decision log are the starting point (copied into the new package, so tutorials stay standalone).

## How Laya steers (not just gates)

Laya makes a decision at each of these points, each ~50 ms, and each falls back to the LLM when confidence is low:

| # | Decision point | Laya answers | Effect |
|---|---|---|---|
| 1 | **Intake** | intent, `needs_plan`, `risky` | Skips planning for simple turns; strips tools for plain chat; sets the workflow hint |
| 2 | **Model route** | `simple` vs `hard` | Simple/chat turns go to `qwen2.5:3b` (sub-second); hard/tool turns go to `qwen2.5:14b`. The biggest single speed win. |
| 3 | **Delegation** | which subagent fits the current todo | A `laya-advisor` subagent the executor can call for a fast decision instead of a slow reasoning step |
| 4 | **Tool gate** | `destructive`, `needs_user`, `authorized` (did the user actually ask for this action, judged from the recent messages, the pattern LangChain's `AutoModeMiddleware` uses with Jev) | Block / ask you / allow, per tool, with per-tool thresholds. `authorized` is the defence against instructions injected through web pages or tool output |
| 5 | **Sufficiency** | is this subagent result good enough to continue | Cheap yes/no before spending an LLM turn on review |
| 6 | **Failure kind** | what kind of failure is this | Logged now; steers only once a fine-tuned Laya earns the trust |

Every decision is logged (state, question, answer, confidence, ms) so later phases can measure agreement with the LLM and fine-tune Laya.

## Speed strategy

1. **Skip LLM calls whenever Laya can decide** (points 1-5). A plain chat turn = one Laya call + one small-model call, no planning, no tools.
2. **Right-size the model per turn** (point 2): 3B for simple turns, 14B only when needed. Both stay resident (~2 GB + ~12 GB of 51 GB).
3. **Everything warm at startup**: Laya, Whisper, both Ollama models (via `keep_alive`), the Docker container and its image. First-request latency is the thing to eliminate.
4. **Small, constant context** (`num_ctx` 16k, not Ollama's 32k default), fewer tokens to prefill. The value never changes between requests, because changing `num_ctx` forces Ollama to reload the model. Recommend `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` on the Ollama server.
4b. **Prompt-prefix cache friendliness**: Ollama reuses the KV cache for an identical prompt prefix, so the stable parts (system prompt, memory, skills list, tool descriptions) stay first and unchanged, and the per-turn Laya note is appended after them, never edited into them. Deep Agents already orders `MemoryMiddleware` after prompt caching for the same reason. A `HarnessProfile` with `excluded_tools` drops unused built-in tools from the prompt (fewer prefill tokens on every turn), especially for the 3B fast path.
5. **Stream everything** (`stream_events` v3) so tokens and steps appear as they happen.
6. **Sandbox speed**: one long-lived container per session (never `docker run` per call), image built once with Python/pytest preinstalled. Each Deep Agents file tool is one `docker exec` (~100-200 ms), slower than the host shell, so the spec keeps a confined `LocalShellBackend` fallback when Docker is unavailable.
7. **One model call at a time** (your instruction; also what avoids the MPS crash and GPU pile-ups). Trade-off to know: parallel subagent fan-out is therefore sequential by default. An `ENGINE_LLM_PARALLEL` switch will allow concurrent Ollama requests (Laya and Whisper stay serialized) once tested.
8. **A live speed panel** in the UI: per-stage ms, Laya vs LLM, so regressions are visible.

## Deep Agents feature coverage (everything except async subagents)

| Feature | How the assistant uses it |
|---|---|
| `create_deep_agent`, checkpointer, store | The executor; `InMemorySaver` for threads, a persistent store for memory |
| **Own sandbox filesystem** | `DockerSandbox` backend (`BaseSandbox`, tutorial 14): the agent's `/workspace` lives in its container; all file tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) work there; user uploads via `upload_files`, artifacts via `download_files` (UI download buttons) |
| `CompositeBackend` routing | `/workspace` -> Docker sandbox; `/memories/` -> persistent store on disk; `/skills/` -> read-only skills dir |
| `FilesystemPermission` | `/skills/**` read-only (deny write), `/memories/**` read-write, writes outside `/workspace` denied or interrupt |
| **Planning** (`write_todos`) | Executor plans; the todo list is gated by HITL so you can edit/approve it |
| **Subagents** (`SubAgent`) | `researcher` (search tools), `coder` (sandbox shell, own skills), `reviewer` (read-only permissions), each with its own model/tools/`interrupt_on`/`permissions`/`skills`; plus the built-in general-purpose subagent with a `GeneralPurposeSubagentProfile` |
| `CompiledSubAgent` | `laya-advisor`: a small LangGraph graph whose only node is a Laya call, so "which subagent / are we done?" is answered in ~50 ms instead of an LLM turn |
| HITL (`interrupt_on`) | Plan approval; Laya-triggered approvals on risky tool calls; approve / edit / reject / respond |
| Custom middleware | `LayaCortexMiddleware` (intake note, model route, tool gate, repeat guard, timing) |
| **Memory** (`memory=`, `MemoryMiddleware`, `CompositeBackend`) | See "Context management and memory": minimal always-loaded `AGENTS.md`, plus durable `/memories/` files, persisted under `~/.laya_assistant/` |
| `state_schema` / `context_schema` | `AssistantState(DeepAgentState)` carries Laya decisions; `AssistantContext` carries session config per run |
| **Skills** (`skills=`, `SkillsMiddleware`) | `coding-workflow`, `research-workflow`, `sandbox-usage` SKILL.md files, loaded progressively |
| Context management | Offloading + `SummarizationMiddleware` + `SummarizationToolMiddleware`, explicitly re-tuned for the 16k window (see the section above, the defaults would not trigger in time) |
| `RubricMiddleware` | Evaluator-optimizer: the final result is graded against criteria derived from the approved plan (max 3 iterations); a Laya sufficiency check runs first so the LLM grader is only paid for when needed |
| `HarnessProfile` / `ProviderProfile` | Per-model tuning for the 14B and 3B (prompt tweaks, excluded tools) registered like tutorial 18 |
| Code interpreter (QuickJS) + dynamic subagents | `CodeInterpreterMiddleware` gives an `eval` tool; the coordinator can dispatch subagents from code (batch tasks). Sequential under the one-call-at-a-time rule unless `ENGINE_LLM_PARALLEL` is on. |
| Multimodal | Image upload + `describe_image` (qwen3-vl:2b); screenshots become first-class in phases 2-3 |
| Streaming | `stream_events(version="v3")`, `messages` + `subagents` projections drive the live feed |
| LangGraph patterns (tutorials 1-7) | Routing = Laya intake; orchestrator-worker = executor + subagents; evaluator-optimizer = rubric loop; parallelization = optional fan-out; prompt chaining = plan steps |

## Context management and memory (first-class, per the Deep Agents docs)

Sources: [context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering), [customization / memory](https://docs.langchain.com/oss/python/deepagents/customization), LangChain's [context management post](https://www.langchain.com/blog/context-management-for-deepagents).

**The 16k trap.** Deep Agents' defaults assume the model profile's full window: tool results are offloaded past **20,000 tokens**, and inputs are offloaded and history is summarized at **85% of `max_input_tokens`** (keeping 10%). Our Ollama window is 16k, so with defaults a single 15k-token tool result would overflow before anything triggers. `FilesystemMiddleware(tool_token_limit_before_evict=...)` and `SummarizationMiddleware(trigger=..., keep=...)` are tunable, but `create_deep_agent` builds its own copies. The plan therefore swaps in explicitly configured ones via `HarnessProfile(excluded_middleware=..., extra_middleware=...)` (verified as available fields) and asserts the effective values in a test.

| Layer | Setting for a 16k window | Why |
|---|---|---|
| Tool-result offload | ~3k tokens (default 20k) | Big results go to the sandbox filesystem; the model sees a path + 10-line preview and pulls fragments with `read_file`/`grep` |
| Tool-input offload | at ~60% of window | Old `write_file`/`edit_file` arguments replaced by file pointers |
| Summarization | trigger ~70% (about 11k tokens), keep last ~2k; original messages saved to the backend so nothing is lost | Structured summary (intent, artifacts, next steps) replaces history; agent can re-open the saved transcript |
| `compact_conversation` tool | `SummarizationToolMiddleware` | Agent can compact between tasks on demand |
| Subagent isolation | Every subagent's `system_prompt` says "return under 500 words, no raw output" | Heavy work stays out of the main context; only summaries return |
| Compaction test | Run with artificially low triggers (10-20% of window) and verify goal continuity and detail recovery, as the LangChain post recommends | Compaction failures are subtle (goal drift) |

**Memory design** (short always-loaded memory, larger recoverable memory):
- `memory=["/memories/AGENTS.md"]`: always in the system prompt, so kept minimal: user preferences, conventions, standing instructions. Written by the agent (`/memories/` route is read-write).
- `/memories/` routed by `CompositeBackend` to a disk-backed store under `~/.laya_assistant/` (survives restarts; scoped per user directory). The system prompt documents the structure explicitly, as the docs advise: `AGENTS.md` = preferences, `sessions/<date>.md` = what was done and where artifacts are, `research/<topic>.md` = findings.
- Skills carry procedures (progressive disclosure), memory carries facts about you. Nothing procedural goes in `AGENTS.md`.
- **Per-run state and context**: a `DeepAgentState` subclass holds Laya's decisions (intake, route, band, per-step probabilities) so they are available to tools, middleware and the UI, as Jev's router keeps probabilities in agent state; a `context_schema` dataclass passes session id, thresholds and workspace paths to tools per run without polluting the prompt.
- Laya's own decision log is separate: a JSONL per session for measurement and later fine-tuning; it is not agent memory.

## Architecture

```
mic ─► stt.transcribe ─┐
typed text / image ────┴► LAYA INTAKE (intent, needs_plan, risky, simple|hard)
                             │ route: simple ─► qwen2.5:3b (no plan, no tools)
                             ▼        hard   ─► EXECUTOR Deep Agent (qwen2.5:14b)
                                   write_todos ─► [you approve/edit plan] ─► execute
        subagents: researcher · coder · reviewer · general-purpose · laya-advisor
        backend: /workspace = Docker sandbox · /memories = disk · /skills = read-only
        RubricMiddleware grades the result against the plan's criteria
                 every model/tool call ─► LAYA CORTEX (gate · bands · repeat guard · log)
UI: live step feed · decision log (system badge, confidence, ms) · speed panel · file downloads
```

### Components (package `src/laya_assistant/`, one purpose each)

| File | Purpose | Depends on |
|---|---|---|
| `engine.py` | `ENGINE` lock, `LockedLaya`, `load_laya()`, `make_llm(name)` (Ollama, `num_ctx`/`keep_alive`), `warm_all()` | laya, langchain-ollama |
| `decisions.py` | `DecisionLog` (rows + JSONL), `confidence_band`, `guard_action` | none |
| `stt.py` | `Transcriber`: MLX Whisper, `transcribe(wav_bytes) -> str`, under `ENGINE` | mlx-whisper, ffmpeg |
| `intake.py` | Laya intake -> typed `Intake` (intent, needs_plan, risky, route) with bands | engine, decisions |
| `sandbox.py` | `DockerSandbox` (per-session container), image build, health check, confined `LocalShellBackend` fallback | docker |
| `cortex.py` | `LayaCortexMiddleware`: intake note, model routing, per-tool gate, repeat guard, timing | engine, decisions, intake |
| `advisor.py` | `laya-advisor` `CompiledSubAgent` | engine |
| `tools.py` | `internet_search` (ddgs), `describe_image` (qwen3-vl:2b) | ollama |
| `agents.py` | `build_assistant(session)`: `create_deep_agent` with all of the above wired | all above |
| `skills/` | SKILL.md files | none |
| `app.py` | Streamlit UI (thin: rendering only) | agents, stt |

## Data flow for one turn

1. Input: typed text, or audio -> `Transcriber` (~1 s for a short clip; transcript shown in the chat and sent). Uploads are placed in the sandbox `/workspace/uploads/` and referenced in the message.
2. `intake.classify` (one Laya pass): intent (`chat | question | write_code | research | computer_task | other`), `needs_plan`, `risky`, `simple|hard`. Confidence sets the band; thresholds come from a labelled probe set, not guesses. Defer band -> the 14B executor decides.
3. **Simple + confident**: routed to the 3B model, tools stripped, no plan, streamed straight to the UI.
4. **Otherwise** the executor calls `write_todos`; HITL shows an editable checklist; approve, edit or reject.
5. Execution: subagents work in the sandbox; each tool call goes through the cortex (hard-deny regex in code -> Laya `destructive` with per-tool thresholds -> block / ask / allow; identical failure blocked after 3; step budget). Because the sandbox is a container, the blast radius of sandbox commands is small: they use block-only at a high threshold, while host-touching tools (phases 2-3) get the strict ask-you thresholds.
6. Laya sufficiency check on subagent results; `RubricMiddleware` grades the final result and the agent revises up to 3 times.
7. UI streams the run live and shows the decision log and speed panel; artifacts are downloadable from the sandbox.
8. Memory: preferences are written to `/memories/AGENTS.md` and re-read next session.

## Error handling

- Ollama down or a model missing: startup check names the exact `ollama pull`; app still opens.
- Docker down: try starting Docker Desktop (as tutorial 14 does); if it fails, fall back to the confined host shell with a visible warning.
- STT failure or empty transcript: shown in the UI; typing still works; nothing sent.
- First Laya call slow in a new thread: warm-up per script thread (measured in the plan).
- Runaway loops: repeat guard + `recursion_limit` budget; UI shows "Stopped: step budget reached".
- Rejected plan or approval: the agent is told and must pick another approach or ask.
- Container cleanup: session container stopped on "New session" and on process exit.
- Every model call goes through `ENGINE`; the UI never calls a model directly.

## Testing

Real components, no fake Laya (same rule as tutorial 19).
- Unit: bands, guard actions, hard-deny regex, decision log, backend routing and `FilesystemPermission` rules (pure).
- Laya, real checkpoint: a labelled probe set of ~40 utterances for intake and routing. Tests assert minimum accuracy at the act band and that low-confidence cases defer. The probe set also picks thresholds and chooses between `laya` and `typed-decisions`.
- Middleware: real Laya + stubbed handlers.
- Context and memory: assert the effective offload/summarization thresholds fit 16k; force low triggers and check the agent keeps its goal and can recover an offloaded detail; write a preference in one session and read it in a new one.
- Sandbox, real Docker: write/read/edit/upload/download round-trip, permission denials, per-call latency measured.
- STT, real model: audio from macOS `say` + `ffmpeg`, transcript match, warm latency reported.
- Thread safety: concurrent Laya + Whisper + LLM calls must not crash or stall.
- Speed budget test: warm plain-chat turn and warm intake under stated targets, measured and printed.
- Agent smoke (slow, needs Ollama + Docker): plain chat, code task with plan approval and rubric pass, blocked destructive command, memory persisted across sessions.
- UI: Streamlit `AppTest` smoke plus one real-browser check.

## Success criteria

1. `uv run streamlit run src/laya_assistant/app.py --server.fileWatcherType none` starts, warms every model, and shows status.
2. Speaking a request produces a transcript and a reply; typing works identically.
3. Intake + route decision under 100 ms warm, shown with confidence. A plain chat turn starts streaming in about 1 s after send (target, to be measured and reported).
4. A code task shows an editable plan, waits for approval, executes in the Docker sandbox, is rubric-graded, and its files are downloadable.
5. Destructive or unsafe commands are blocked or ask; loops are stopped.
6. No crash or stall under repeated clicks and reruns.
7. Per-session decision log JSONL written; memory persists across sessions.

## Dependencies to add

`mlx-whisper` (weights already cached). Docker Desktop and `ffmpeg` are already installed. Sandbox image built locally. Nothing paid or cloud.

## Later phases (separate specs)

2. Browser operator: Playwright, observe -> shortlist -> Laya gate loop, more skills, parallel fan-out tuning.
3. Desktop operator: macOS accessibility tree + screenshots, stricter gates, permissions flow.
4. Learning loop: agreement dashboard, training-data export for a Laya fine-tune, eval harness.
