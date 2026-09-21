"""Speech-to-text against the REAL MLX Whisper model, on audio generated with macOS `say` + ffmpeg."""
import subprocess

import pytest

from laya_assistant.stt import Transcriber


def make_wav(tmp_path, text: str) -> bytes:
    aiff, wav = tmp_path / "clip.aiff", tmp_path / "clip.wav"
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(wav)], check=True)
    return wav.read_bytes()


@pytest.fixture(scope="module")
def transcriber():
    t = Transcriber()
    t.warm()
    return t


def test_transcribes_a_spoken_request(transcriber, tmp_path):
    r = transcriber.transcribe(make_wav(tmp_path, "write a python function that reverses a string"))
    print("transcript:", repr(r.text), f"{r.ms:.0f} ms")
    assert "python" in r.text.lower() and "reverse" in r.text.lower()


def test_warm_transcription_is_fast_for_a_short_clip(transcriber, tmp_path):
    wav = make_wav(tmp_path, "open the browser and search for flights to Lisbon")
    transcriber.transcribe(wav)
    r = transcriber.transcribe(wav)
    print(f"warm transcribe: {r.ms:.0f} ms")
    assert r.ms < 3000


def test_silence_gives_empty_text_and_does_not_raise(transcriber, tmp_path):
    wav = tmp_path / "silence.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1", str(wav)], check=True)
    assert transcriber.transcribe(wav.read_bytes()).text.strip() == ""


def test_garbage_bytes_raise_a_clear_error(transcriber):
    with pytest.raises(RuntimeError, match="audio"):
        transcriber.transcribe(b"not audio at all")


def quiet_wav(tmp_path, text: str, volume: float) -> bytes:
    """Speech attenuated like a distant or low-gain microphone."""
    aiff, wav = tmp_path / "q.aiff", tmp_path / "q.wav"
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-af", f"volume={volume}", "-ar", "16000", "-ac", "1", str(wav)], check=True)
    return wav.read_bytes()


@pytest.mark.parametrize("volume", [0.1, 0.03, 0.01])
def test_quiet_microphone_speech_is_still_transcribed(transcriber, tmp_path, volume):
    r = transcriber.transcribe(quiet_wav(tmp_path, "create a new file called notes dot txt", volume))
    print(f"volume {volume}: {r.text!r}")
    assert "file" in r.text.lower() and "notes" in r.text.lower()


def test_background_noise_alone_does_not_become_a_message(transcriber, tmp_path):
    wav = tmp_path / "noise.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anoisesrc=color=white:amplitude=0.004:sample_rate=16000", "-t", "2", str(wav)], check=True)
    r = transcriber.transcribe(wav.read_bytes())
    print("noise transcript:", repr(r.text))
    assert r.text == ""


def test_language_is_pinned_so_short_clips_are_not_misdetected(transcriber):
    from laya_assistant import config

    assert config.STT_LANGUAGE == "en" and transcriber.language == "en"


# --- voice activity: room noise must never reach Whisper (it answers "Thank you." to noise, at 0.00 no_speech_prob) ---


def _ff(tmp_path, name, args):
    p = tmp_path / f"{name}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, "-ar", "16000", "-ac", "1", str(p)], check=True)
    return p


def _samples(path):
    import numpy as np
    from mlx_whisper.audio import load_audio

    return np.array(load_audio(str(path)))


@pytest.mark.parametrize("amp", [0.002, 0.006, 0.02, 0.05])
def test_stationary_room_noise_is_not_speech(tmp_path, amp):
    from laya_assistant.stt import has_speech

    assert not has_speech(_samples(_ff(tmp_path, "n", ["-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude={amp}:sample_rate=16000", "-t", "3"])))


def test_a_steady_hum_is_not_speech(tmp_path):
    from laya_assistant.stt import has_speech

    assert not has_speech(_samples(_ff(tmp_path, "hum", ["-f", "lavfi", "-i", "sine=frequency=120:duration=3", "-af", "volume=0.05"])))


def test_a_single_click_in_silence_is_not_speech():
    import numpy as np

    from laya_assistant.stt import has_speech

    x = np.zeros(16000 * 2, dtype=np.float32)
    x[8000:8100] = 0.5  # a 6 ms click
    assert not has_speech(x)


@pytest.mark.parametrize("text,volume", [("no", 1.0), ("thank you", 1.0), ("open the notes file", 1.0), ("create a new file called notes", 0.02)])
def test_real_speech_including_very_short_and_quiet_is_speech(tmp_path, text, volume):
    from laya_assistant.stt import has_speech

    aiff = tmp_path / "s.aiff"
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    assert has_speech(_samples(_ff(tmp_path, "sp", ["-i", str(aiff), "-af", f"volume={volume}"])))


def test_speech_buried_in_room_noise_is_still_speech_and_transcribed(transcriber, tmp_path):
    aiff, noise = tmp_path / "s.aiff", tmp_path / "n.wav"
    subprocess.run(["say", "-o", str(aiff), "create a new file called notes"], check=True)
    _ff(tmp_path, "n", ["-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.01:sample_rate=16000", "-t", "4"])
    mixed = _ff(tmp_path, "mix", ["-i", str(aiff), "-i", str(noise), "-filter_complex", "[0]volume=0.3[a];[a][1]amix=inputs=2:duration=longest:normalize=0"])
    r = transcriber.transcribe(mixed.read_bytes())
    print("speech in noise:", repr(r.text))
    assert "file" in r.text.lower()


def test_noise_and_hum_never_become_a_transcript(transcriber, tmp_path):
    for name, args in [("pink", ["-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.02:sample_rate=16000", "-t", "3"]),
                       ("hum", ["-f", "lavfi", "-i", "sine=frequency=120:duration=3", "-af", "volume=0.05"])]:
        assert transcriber.transcribe(_ff(tmp_path, name, args).read_bytes()).text == "", name
