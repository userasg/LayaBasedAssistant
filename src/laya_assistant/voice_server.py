"""Live speech-to-text over a local WebSocket.

The browser component streams 16 kHz mono int16 PCM chunks here while you speak. Whisper is not a streaming
model, so the server keeps a rolling buffer of the utterance and re-decodes it whenever about a second of new
audio has arrived, sending `partial` text back so your words appear as you talk. Each partial is also run
through Laya's intake (~55 ms) so the routing decision ("write_code -> Deep Agent") forms live. On `stop` the
whole utterance is decoded once more and sent as `final`; that is what goes to the assistant.

Protocol (JSON text messages, plus binary audio frames):
  client -> {"type": "start"}          reset the buffer
  client -> <bytes>                    PCM16 mono 16 kHz
  client -> {"type": "stop"}           finish
  server -> {"type": "partial", "text": ..., "intake": {...}|null, "seconds": float, "ms": float}
  server -> {"type": "final",   "text": ..., "intake": {...}|null, "seconds": float, "ms": float}
"""
import asyncio
import json
import re
import threading

import numpy as np
import websockets

from . import intake as intake_mod
from .stream_intent import ClauseStreamer

# A web page you visit can open a WebSocket to localhost. Browsers always send the page's Origin, so only pages served from
# this machine (the Streamlit app) are accepted; non-browser clients (tests, scripts) send no Origin and are allowed.
LOCAL_ORIGINS = [None, re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")]

MIN_NEW_SECONDS = 0.8  # decode again once this much new audio has arrived
MAX_SECONDS = 30.0  # Whisper's window; longer utterances keep only the latest 30 s for partials


class LiveUtterance:
    """The audio of one utterance so far."""

    def __init__(self):
        self.pcm = bytearray()
        self.decoded_bytes = 0

    def feed(self, chunk: bytes) -> None:
        self.pcm += chunk

    @property
    def seconds(self) -> float:
        return len(self.pcm) / 2 / 16000

    def new_seconds(self) -> float:
        return (len(self.pcm) - self.decoded_bytes) / 2 / 16000

    def samples(self, tail_seconds: float | None = None) -> np.ndarray:
        raw = bytes(self.pcm)
        if tail_seconds is not None:
            raw = raw[-int(tail_seconds * 16000) * 2:]
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class VoiceServer:
    def __init__(self, transcriber, predictor=None, host: str = "localhost", port: int = 8765, quick=None, on_commit=None):
        self.transcriber, self.predictor, self.host, self.port = transcriber, predictor, host, port
        self.quick = quick  # stream_intent.QuickExecutor: reversible commands that may run while you are still speaking
        self.on_commit = on_commit  # optional callback(clause, result) for logging
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._server = None
        self.error: Exception | None = None

    # -- lifecycle -----------------------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="voice-server", daemon=True)
        self._thread.start()
        self._ready.wait(10)
        if self.error:
            raise self.error

    def _run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        try:
            self._server = await websockets.serve(self._handle, self.host, self.port, max_size=None, origins=LOCAL_ORIGINS)
        except Exception as e:  # port in use, etc.
            self.error = e
            self._ready.set()
            return
        self._ready.set()
        await self._stop_event.wait()
        self._server.close()
        await self._server.wait_closed()

    def stop(self) -> None:
        if self._loop and self._server:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    # -- per-connection ------------------------------------------------------------------------------

    def _decode(self, utt: LiveUtterance, final: bool) -> dict:
        samples = utt.samples(None if final else MAX_SECONDS)
        r = self.transcriber.transcribe_samples(samples)
        utt.decoded_bytes = len(utt.pcm)
        info = None
        if self.predictor is not None and r.text:
            i = intake_mod.classify(self.predictor, r.text)
            info = {"intent": i.intent, "route": i.route, "confidence": round(i.confidence, 3),
                    "risky": round(i.risky, 3), "ms": round(i.ms, 1)}
        return {"type": "final" if final else "partial", "text": r.text, "intake": info,
                "seconds": round(utt.seconds, 2), "ms": round(r.ms, 1)}

    async def _handle(self, ws) -> None:
        loop = asyncio.get_running_loop()
        utt = LiveUtterance()
        last_text = None
        streamer = ClauseStreamer()
        async for msg in ws:
            if isinstance(msg, bytes):
                utt.feed(msg)
                if utt.new_seconds() >= MIN_NEW_SECONDS:
                    out = await loop.run_in_executor(None, self._decode, utt, False)
                    if out["text"] and out["text"] != last_text:  # silence and repeats produce no message
                        last_text = out["text"]
                        await ws.send(json.dumps(out))
                    if self.quick is not None and out["text"]:
                        # stability needs every update, including unchanged ones, so this is fed even when nothing is sent
                        for clause in await loop.run_in_executor(None, lambda: streamer.feed(out["text"], self.quick.recognises)):
                            res = await loop.run_in_executor(None, self.quick.execute, clause)
                            if res is not None:
                                if self.on_commit:
                                    self.on_commit(clause, res)
                                await ws.send(json.dumps({"type": "commit", "clause": clause,
                                                          "at": round(utt.seconds, 2),
                                                          "result": {"ok": res.ok, "text": res.text, "ms": round(res.ms, 1), "kind": res.kind}}))
                continue
            ctl = json.loads(msg)
            if ctl.get("type") == "start":
                utt, last_text, streamer = LiveUtterance(), None, ClauseStreamer()
            elif ctl.get("type") == "stop":
                out = await loop.run_in_executor(None, self._decode, utt, True) if utt.seconds > 0.2 else \
                    {"type": "final", "text": "", "intake": None, "seconds": round(utt.seconds, 2), "ms": 0.0}
                remaining, done = streamer.remaining(out["text"]) if self.quick is not None else (out["text"], [])
                out["remaining"], out["done"] = remaining, done
                await ws.send(json.dumps(out))
