"""Live speech-to-text over a real WebSocket, with real speech audio (macOS `say`), real Whisper and real Laya."""
import asyncio
import json
import subprocess
import time

import numpy as np
import pytest

from .conftest import run_async
import websockets

from laya_assistant.stt import Transcriber
from laya_assistant.voice_server import VoiceServer

PORT = 8791


def speech_pcm(tmp_path, text: str) -> bytes:
    aiff, raw = tmp_path / "s.aiff", tmp_path / "s.raw"
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-f", "s16le", "-ar", "16000", "-ac", "1", str(raw)], check=True)
    return raw.read_bytes()


@pytest.fixture(scope="module")
def server(laya_model):
    t = Transcriber()
    t.warm()
    s = VoiceServer(t, laya_model, port=PORT)
    s.start()
    yield s
    s.stop()


async def stream_utterance(pcm: bytes, speed: float = 3.0, seconds_per_chunk: float = 0.25):
    """Send audio like a microphone would (chunked, paced), then stop. Returns (messages with arrival offsets, stop_time)."""
    msgs, t0 = [], time.perf_counter()
    async with websockets.connect(f"ws://localhost:{PORT}/stt", max_size=None) as ws:
        await ws.send(json.dumps({"type": "start"}))
        step = int(16000 * 2 * seconds_per_chunk)

        async def reader():
            async for m in ws:
                d = json.loads(m)
                msgs.append((time.perf_counter() - t0, d))
                if d["type"] == "final":
                    return

        task = asyncio.create_task(reader())
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i + step])
            await asyncio.sleep(seconds_per_chunk / speed)
        stop_at = time.perf_counter() - t0
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(task, 30)
    return msgs, stop_at


def test_partial_text_appears_while_speaking_and_final_is_correct(server, tmp_path):
    pcm = speech_pcm(tmp_path, "write a python function that reverses a string and then test it with a few examples")
    msgs, stop_at = run_async(stream_utterance(pcm))
    partials = [(t, d) for t, d in msgs if d["type"] == "partial"]
    final = next(d for _, d in msgs if d["type"] == "final")
    final_at = next(t for t, d in msgs if d["type"] == "final")
    print(f"{len(partials)} partials; first at {partials[0][0]:.1f}s; stop at {stop_at:.1f}s; final {final_at - stop_at:.2f}s after stop")
    print("last partial:", partials[-1][1]["text"], "| final:", final["text"])
    assert len(partials) >= 2
    assert partials[0][0] < stop_at  # words appeared BEFORE the user stopped speaking
    assert "python" in final["text"].lower() and "reverses" in final["text"].lower()
    assert final_at - stop_at < 3.0  # final text arrives shortly after stop


def test_laya_previews_the_intake_decision_on_the_live_text(server, tmp_path):
    pcm = speech_pcm(tmp_path, "write a python function that reverses a string and then test it")
    msgs, _ = run_async(stream_utterance(pcm))
    with_intake = [d for _, d in msgs if d["type"] in ("partial", "final") and d.get("intake")]
    assert with_intake
    last = with_intake[-1]["intake"]
    print("intake preview:", last)
    assert set(last) >= {"intent", "route", "confidence", "risky", "ms"}
    assert last["route"] == "executor" and last["ms"] < 300


def test_silence_yields_an_empty_final_and_no_partials(server):
    silence = bytes(16000 * 2 * 2)  # 2 s
    msgs, _ = run_async(stream_utterance(silence))
    assert [d for _, d in msgs if d["type"] == "partial" and d["text"]] == []
    assert next(d for _, d in msgs if d["type"] == "final")["text"] == ""


def test_two_simultaneous_speakers_do_not_crash_the_process(server, tmp_path):
    pcm = speech_pcm(tmp_path, "what is the capital of France")

    async def both():
        return await asyncio.gather(stream_utterance(pcm), stream_utterance(pcm))

    (a, _), (b, _) = run_async(both())
    for msgs in (a, b):
        assert "france" in next(d for _, d in msgs if d["type"] == "final")["text"].lower()


def test_a_new_start_resets_the_buffer(server, tmp_path):
    first = speech_pcm(tmp_path, "hello there my friend")
    msgs, _ = run_async(stream_utterance(first))
    second = speech_pcm(tmp_path, "open the notes file")
    msgs2, _ = run_async(stream_utterance(second))
    text2 = next(d for _, d in msgs2 if d["type"] == "final")["text"].lower()
    assert "notes" in text2 and "friend" not in text2


# --- a web page must not be able to drive the microphone socket ------------------------------------------------------


@pytest.mark.parametrize("origin,allowed", [("https://evil.example", False), ("http://localhost:8501", True), ("http://127.0.0.1:8599", True),
                                            ("http://localhost.evil.example", False), ("null", False)])
def test_only_pages_served_from_this_machine_may_connect(server, origin, allowed):
    from websockets.exceptions import InvalidStatus
    from websockets.sync.client import connect

    url = f"ws://localhost:{PORT}/stt"
    if allowed:
        with connect(url, origin=origin) as ws:
            ws.send('{"type": "start"}')
    else:
        with pytest.raises(InvalidStatus) as e:
            connect(url, origin=origin)
        assert e.value.response.status_code == 403
