"""Speech to text with MLX Whisper (runs on the Apple GPU). Weights come from the local HF cache."""
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .engine import ENGINE


SILENCE_RMS = 0.0002  # about -74 dBFS: a dead or muted mic. Quiet speech from a low-gain mic sits well above this.
MIN_SPEECH_SECONDS = 0.15  # shortest utterance that counts ("no.", "yes")
MIN_DYNAMIC_RANGE_DB = 10.0  # speech swings between syllables and pauses; stationary noise and hum stay flat
MAX_GAIN = 30.0  # +30 dB at most: lifts a quiet mic without turning room noise into speech
TARGET_PEAK = 0.9
# What Whisper makes up for near-silent input. Dropped only when the clip is quiet, so a real "thank you" survives.
HALLUCINATIONS = {"thank you", "thanks", "thanks for watching", "you", "bye", "thank you for watching"}
QUIET_RMS = 0.01


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    ms: float


def has_speech(samples, sr: int = 16000) -> bool:
    """Is there speech in this clip? A voice-activity check that runs BEFORE Whisper.

    Whisper answers "Thank you." to room noise (with no_speech_prob 0.00 once the level is boosted), and RMS cannot
    tell quiet speech from loud-ish noise, so this looks at structure instead: speech has a large energy swing
    between syllables and pauses and at least a fraction of a second of frames well above the noise floor;
    stationary noise, a hum or a click do not."""
    import numpy as np

    frame, hop = int(0.025 * sr), int(0.010 * sr)
    if samples.size < frame + 20 * hop:
        return False
    n = 1 + (samples.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    energy = np.mean(samples[idx] ** 2, axis=1)
    db = 10 * np.log10(energy + 1e-12)
    floor, top = np.percentile(db, 20), np.percentile(db, 95)
    if top < -75 or top - floor < MIN_DYNAMIC_RANGE_DB:
        return False
    return float(np.sum(db > floor + 8)) * hop / sr >= MIN_SPEECH_SECONDS


class Transcriber:
    """`transcribe(bytes) -> TranscriptResult`. Takes its turn on ENGINE like every other GPU call.
    ffmpeg (already installed) decodes any container/codec mlx-whisper is handed."""

    def __init__(self, model: str = config.STT_MODEL, language: str | None = config.STT_LANGUAGE):
        self.model = model
        self.language = language

    def warm(self) -> float:
        """Load the model by transcribing a second of silence. Returns ms."""
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "s.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                            "-t", "1", str(wav)], check=True)
            return self.transcribe(wav.read_bytes()).ms

    def transcribe(self, audio: bytes, suffix: str = ".wav") -> TranscriptResult:
        import mlx_whisper
        import numpy as np
        from mlx_whisper.audio import load_audio

        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(audio)
            f.flush()
            try:
                samples = np.array(load_audio(f.name))  # ffmpeg -> mono 16 kHz float32 (mlx array -> numpy)
            except Exception as e:
                raise RuntimeError(f"could not read the audio: {e}") from e
        return self.transcribe_samples(samples)

    def transcribe_samples(self, samples) -> TranscriptResult:
        """16 kHz mono float32 samples -> text. Shared by the clip path and the live-streaming path."""
        import mlx_whisper
        import numpy as np

        t0 = time.perf_counter()
        # Whisper invents text ("Thank you.") for silence, and a voice assistant would send it as a
        # message. Gate on energy before spending a GPU call: quiet room noise sits well under this.
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
        if rms < SILENCE_RMS or not has_speech(samples):
            return TranscriptResult(text="", ms=(time.perf_counter() - t0) * 1000)
        peak = float(np.abs(samples).max())
        # Low-gain microphones are the usual cause of poor dictation: bring the level up before Whisper hears it.
        samples = np.clip(samples * min(TARGET_PEAK / peak, MAX_GAIN), -1.0, 1.0).astype(np.float32)
        with ENGINE:
            t0 = time.perf_counter()
            out = mlx_whisper.transcribe(samples, path_or_hf_repo=self.model, language=self.language,
                                         condition_on_previous_text=False, temperature=0.0)
            ms = (time.perf_counter() - t0) * 1000
        text = out["text"].strip()
        if rms < QUIET_RMS and text.lower().strip(" .!?,") in HALLUCINATIONS:
            text = ""
        return TranscriptResult(text=text, ms=ms)
