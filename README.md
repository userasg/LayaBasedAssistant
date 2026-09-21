# Laya Based Assistant

A local voice-and-text assistant. **Laya** (a fast System-1 decision model) decides in ~tens of ms how each request is handled and gates risky
steps; a **Deep Agent** (Ollama, qwen2.5:14b) plans and works in its own Docker sandbox; **Whisper** (MLX) turns speech into text; **computer use**
drives a browser and Mac apps. Everything runs on this Mac: no cloud model.

## Run

```bash
uv sync
uv run streamlit run src/laya_assistant/app.py --server.fileWatcherType none
```

Needs: Ollama running with `qwen2.5:14b-instruct`, `qwen2.5:3b`, `qwen3-vl:2b`; Docker Desktop; `ffmpeg`; the [`laya`](../laya) checkout next to this
folder (`../laya`, see `[tool.uv.sources]` in `pyproject.toml`). For driving Mac apps, Cua Driver plus the macOS Accessibility and Screen Recording
permissions (the app explains each step).

## Test

```bash
uv run pytest tests/laya_assistant -q
```

Real Laya, Ollama, Docker, Whisper and Chromium: about 15 minutes. If the app is already running add `VOICE_PORT=8775`. Tests never touch your real
Notes or open windows (`LAYA_HOST_ACTIONS=off`).

## Docs

- [`docs/laya_assistant_phase1.md`](docs/laya_assistant_phase1.md): what it does, architecture, measurements, problems found and fixed.
- [`docs/laya_assistant_computer_use.md`](docs/laya_assistant_computer_use.md): files, browser and Mac-app control, the click ladder, safety model.
- [`docs/superpowers/`](docs/superpowers): the original specs and plans.

Started as tutorial 19 of a Deep Agents / LangGraph tutorial series (kept in the original repository).
