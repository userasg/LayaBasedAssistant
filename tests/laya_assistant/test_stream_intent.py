"""Acting while speaking. Pure logic first, then the live path: real speech audio -> real Whisper -> real WebSocket -> real Laya.
The only stub is the launcher (it would really open an app on your screen)."""
import asyncio
import json
import subprocess
import time

import pytest

from .conftest import run_async
import websockets

from laya_assistant.stream_intent import AppIndex, ClauseStreamer, QuickExecutor, parse_quick, split_clauses


def test_split_clauses_on_spoken_connectors_and_punctuation():
    assert split_clauses("open notes and write a shopping list") == (["open notes", "write a shopping list"], False)
    assert split_clauses("open notes, then scroll down.") == (["open notes", "scroll down"], True)
    assert split_clauses("open notes and")[1] is True and split_clauses("open notes and")[0] == ["open notes"]
    assert split_clauses("") == ([], False)


def test_a_clause_commits_only_after_it_is_followed_by_a_boundary_and_is_stable():
    s = ClauseStreamer()
    assert s.feed("open") == [] and s.feed("open notes") == []  # no boundary after it yet
    assert s.feed("open notes and") == ["open notes"]  # unchanged across two updates AND a boundary has followed
    assert s.feed("open notes and write") == [] and s.feed("open notes and write a shopping list") == []  # the last clause never commits early
    assert s.acted() == ["open notes"]


def test_a_revised_clause_is_not_committed():
    s = ClauseStreamer()
    s.feed("open notice and")
    assert s.feed("open notes and write") == []  # changed between updates: not stable, not acted on


def test_commits_are_strictly_ordered_and_an_unrecognised_clause_blocks_everything_after_it():
    ok = lambda c: c.lower().startswith("open")
    s = ClauseStreamer()
    for p in ["write a report and", "write a report and open notes and", "write a report and open notes and scroll"]:
        got = s.feed(p, ok)
    assert s.acted() == []  # "open notes" must not run before "write a report"


def test_remaining_text_after_stopping_excludes_what_already_ran():
    s = ClauseStreamer()
    for p in ["open notes and", "open notes and write", "open notes and write a list"]:
        s.feed(p)
    assert s.remaining("Open notes and write a shopping list.") == ("write a shopping list", ["open notes"])
    assert s.remaining("Open the calendar and write a list") == ("Open the calendar and write a list", [])  # revised: nothing assumed


APPS = AppIndex()


@pytest.mark.parametrize("spoken,app", [("open notes", "Notes"), ("open the calculator", "Calculator"), ("launch safari", "Safari"),
                                        ("open my notes app", "Notes"), ("start finder", "Finder")])
def test_spoken_app_names_resolve_to_installed_apps(spoken, app):
    cmd = parse_quick(spoken, APPS)
    assert cmd and cmd.kind == "open_app" and cmd.value == app


@pytest.mark.parametrize("spoken", ["open the pod bay doors", "write a python script", "delete my files", "open zzzz", "send an email to bob"])
def test_other_clauses_are_not_quick_commands(spoken):
    assert parse_quick(spoken, APPS) is None


def test_navigation_and_scroll_commands_parse():
    assert parse_quick("go to example.org", APPS).value == "https://example.org"
    assert parse_quick("scroll down", APPS).kind == "scroll" and parse_quick("scroll up a bit", APPS).value == "up"


def test_navigation_needs_an_approved_domain_and_a_connected_tab(laya_model, tmp_path):
    from laya_assistant.cua.policy import DomainPolicy

    dom = DomainPolicy(tmp_path / "p.json", ask_unknown=True)  # first-visit asks are a setting; this test turns it on
    ex = QuickExecutor(laya_model, APPS, dom, launcher=lambda a: None, browser=object())
    assert not ex.recognises("go to example.org")  # first visit to a new site: asked about after you stop
    dom.approve("https://example.org")
    assert ex.recognises("go to example.org")
    assert not ex.recognises("go to www.paypal.com")  # never automated
    assert not QuickExecutor(laya_model, APPS, dom, launcher=lambda a: None, browser=None).recognises("go to example.org")


def test_power_apps_and_unlisted_commands_are_refused(laya_model):
    ex = QuickExecutor(laya_model, APPS, launcher=lambda a: None)
    assert ex.recognises("open notes")
    assert not ex.recognises("open the terminal")  # power apps are never opened by an early command
    assert not ex.recognises("delete everything")  # not on the whitelist at all


def test_execute_launches_via_the_launcher_and_reports_failures_without_raising(laya_model):
    launched = []
    ex = QuickExecutor(laya_model, APPS, launcher=launched.append)
    r = ex.execute("open notes")
    assert r.ok and r.text == "opened Notes" and launched == ["Notes"]

    def boom(app):
        raise OSError("no such app")

    r = QuickExecutor(laya_model, APPS, launcher=boom).execute("open notes")
    assert not r.ok and "no such app" in r.text


# --- live: real audio through the whole voice path -------------------------------------------------------------------

PORT = 8793


def speech_pcm(tmp_path, text):
    aiff, raw = tmp_path / "s.aiff", tmp_path / "s.raw"
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-f", "s16le", "-ar", "16000", "-ac", "1", str(raw)], check=True)
    return raw.read_bytes()


@pytest.fixture(scope="module")
def live(laya_model):
    from laya_assistant.stt import Transcriber
    from laya_assistant.voice_server import VoiceServer

    t = Transcriber()
    t.warm()
    launched = []
    ex = QuickExecutor(laya_model, AppIndex(), launcher=lambda a: launched.append((a, time.perf_counter())))
    s = VoiceServer(t, laya_model, port=PORT, quick=ex)
    s.start()
    yield s, launched
    s.stop()


async def speak(pcm, speed=2.0):
    msgs, t0 = [], time.perf_counter()
    async with websockets.connect(f"ws://localhost:{PORT}/stt", max_size=None) as ws:
        await ws.send(json.dumps({"type": "start"}))

        async def reader():
            async for m in ws:
                d = json.loads(m)
                msgs.append((time.perf_counter() - t0, d))
                if d["type"] == "final":
                    return

        task = asyncio.create_task(reader())
        step = int(16000 * 2 * 0.25)
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i + step])
            await asyncio.sleep(0.25 / speed)
        stop_at = time.perf_counter() - t0
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(task, 30)
    return msgs, stop_at


def test_open_notes_runs_before_you_finish_speaking_and_the_rest_waits(live, tmp_path):
    server, launched = live
    launched.clear()
    pcm = speech_pcm(tmp_path, "open notes and then write a shopping list with milk and eggs")
    msgs, stop_at = run_async(speak(pcm))
    commits = [(t, d) for t, d in msgs if d["type"] == "commit"]
    final = next(d for _, d in msgs if d["type"] == "final")
    print("commit at", [round(t, 1) for t, _ in commits], "stop at", round(stop_at, 1), "| final:", final["text"], "| remaining:", final.get("remaining"))
    assert [a for a, _ in launched] == ["Notes"]
    total = final["seconds"]
    # it opened Notes before the speech ended (the earliest possible is one transcript update, ~0.8 s, after the clause + "and then")
    assert commits and commits[0][1]["at"] < total - 0.3, (commits[0][1]["at"], total)
    assert commits[0][1]["result"]["text"] == "opened Notes"
    assert "shopping list" in final["remaining"].lower() and "open notes" not in final["remaining"].lower()
    assert [c.lower() for c in final["done"]] == ["open notes"]


def test_nothing_runs_early_when_the_first_clause_is_not_a_quick_command(live, tmp_path):
    server, launched = live
    launched.clear()
    msgs, _ = run_async(speak(speech_pcm(tmp_path, "write a short report and then open notes and close it")))
    assert launched == [] and not [d for _, d in msgs if d["type"] == "commit"]
    assert "write a short report" in next(d for _, d in msgs if d["type"] == "final")["remaining"].lower()
