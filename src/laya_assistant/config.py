"""Settings in one place. Everything is env-overridable; thresholds carry the measurements they came from."""
import os
from pathlib import Path

# transformers probes for TensorFlow at import time and can deadlock beside torch on macOS;
# Laya is torch-only, so tell it not to look (Laya's own tests do the same).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", "qwen2.5:14b-instruct")
FAST_MODEL = os.getenv("FAST_MODEL", "qwen2.5:3b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:2b")
LAYA_CHECKPOINT = os.getenv("LAYA_CHECKPOINT", "convaiinnovations/laya")
STT_MODEL = os.getenv("STT_MODEL", "mlx-community/whisper-large-v3-turbo")
# Pinned: auto-detection on a short clip often guesses the wrong language and returns garbage. Set STT_LANGUAGE= (empty) to auto-detect.
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en") or None

# Constant on purpose: changing num_ctx between requests makes Ollama reload the model.
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.2"))

# Parallel LLM requests are OFF: one model call at a time (also what avoids the MPS crash
# and GPU pile-ups). Laya and Whisper stay serialized even when this is on.
ENGINE_LLM_PARALLEL = os.getenv("ENGINE_LLM_PARALLEL", "0") == "1"

VOICE_PORT = int(os.getenv("VOICE_PORT", "8765"))  # local WebSocket for live speech-to-text
STEP_BUDGET = int(os.getenv("STEP_BUDGET", "60"))  # LangGraph recursion_limit
MAX_REPEAT_FAILURES = 3
MAX_PLAN_ASKS = 3  # per request: after this many plan reviews the agent proceeds instead of asking again
MAX_CONTINUE_NUDGES = 2  # per request: how often the cortex tells a stalled agent to keep going
AUTHORIZED_CONTEXT_MESSAGES = 30  # recent messages the `authorized` gate question sees

# Context budget for the 16k window (Deep Agents' defaults assume the model profile's full
# window: offload at 20k tokens, summarize at 85%; neither would trigger before 16k overflows).
TOOL_EVICT_TOKENS = 3000
SUMMARY_TRIGGER_TOKENS = 11000
SUMMARY_KEEP_TOKENS = 2000

# Gate thresholds. Sandbox commands run in a container, so they are block-only; host-touching
# tools (phases 2-3) ask the user first.
SANDBOX_BLOCK_AT = 0.90
HOST_BLOCK_AT, HOST_ASK_AT = 0.90, 0.70

# Intake thresholds, from the labelled probe sweep (tests/laya_assistant/intake_probe.py, 40 utterances,
# base `laya` checkpoint; `typed-decisions` was no better): with these values 10 of 14 simple turns route
# fast and 0 hard/risky turns do. Greetings score 0.8+ on `simple` (hard requests 0.00); factual questions
# score 0.09-0.34 on `factual` (hard requests <= 0.06), a thin margin, tolerable because the fast path has
# no tools and cannot act. Known gap: `risky` catches destructive phrasing (rm -rf 0.93, wipe 0.92,
# delete account 0.86) but misses consequential non-destructive actions (send email 0.02, pay invoice 0.16,
# transfer money 0.00), so per-tool gates (authorized / needs_user) cover those, not intake.
INTAKE_ACT, INTAKE_VERIFY = 0.65, 0.50
SIMPLE_AT, FACTUAL_AT, RISKY_AT, PLAN_AT = 0.50, 0.10, 0.40, 0.50

SHARED_DIR = Path(os.getenv("LAYA_SHARED_DIR", "~/LayaWorkspace")).expanduser()  # mounted into the sandbox at /workspace/shared
HOME = Path(os.getenv("LAYA_ASSISTANT_HOME", "~/.laya_assistant")).expanduser()
