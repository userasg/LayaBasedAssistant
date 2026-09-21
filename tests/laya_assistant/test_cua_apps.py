"""Finding an app by what people call it, and bringing it up so it can be driven. The driver is faked only where the test needs a window to
appear or not appear on cue; the real driver is exercised in test_cua_desktop.py's live test."""
import os

import pytest

from laya_assistant.cua import apps


@pytest.fixture
def catalog(tmp_path):
    for name in ("Music", "Notes", "Google Chrome", "Visual Studio Code", "System Settings", "Calculator", "Microsoft Word"):
        (tmp_path / f"{name}.app").mkdir()
    (tmp_path / "Music.app" / "Contents" / "Resources").mkdir(parents=True)
    (tmp_path / "Music.app" / "Contents" / "Resources" / "Music.sdef").write_text("<dictionary/>")
    return apps.AppCatalog([str(tmp_path)])


@pytest.mark.parametrize("spoken,expected", [
    ("Music", "Music"), ("apple music", "Music"), ("the music app", "Music"), ("chrome", "Google Chrome"), ("vs code", "Visual Studio Code"),
    ("settings", "System Settings"), ("my calculator", "Calculator"), ("word", "Microsoft Word"), ("visual studio", "Visual Studio Code"),
])
def test_apps_are_found_by_what_people_call_them(catalog, spoken, expected):
    r = catalog.resolve(spoken)
    assert r and r.name == expected, (spoken, r)


def test_things_that_are_not_installed_are_not_guessed(catalog):
    assert catalog.resolve("photoshop") is None and catalog.resolve("") is None and catalog.resolve("the") is None


def test_an_app_with_a_scripting_dictionary_is_recognised(catalog):
    assert catalog.scripting("Music") and not catalog.scripting("Notes") and not catalog.scripting("Nope")


class Windows:
    """A driver whose window for the app follows a script. `readable` says whether its controls resolve once it is on screen."""

    def __init__(self, states, readable=True):
        self.states, self.polls, self.readable = states, 0, readable

    def call(self, tool, args=None, timeout=None):
        if tool == "get_window_state":
            return {"elements": [{"element_index": 1}] if self.readable else []}
        assert tool == "list_windows"
        state = self.states[min(self.polls, len(self.states) - 1)]
        self.polls += 1
        return {"windows": [{"app_name": "Music", "pid": 5, "window_id": 9, "z_index": 1, "bounds": {"width": 900, "height": 600}, **w} for w in state]}


@pytest.fixture
def host_on(monkeypatch):
    monkeypatch.setenv("LAYA_HOST_ACTIONS", "on")


def test_it_launches_once_then_waits_until_the_window_is_on_screen(host_on):
    launched = []
    cli = Windows([[], [{"is_on_screen": False, "on_current_space": False}], [{"is_on_screen": True, "on_current_space": True}]])
    p = apps.bring_up("Music", cli, launcher=launched.append)
    assert p.ok and (p.pid, p.window_id) == (5, 9) and launched == ["Music"]


@pytest.mark.parametrize("states,words", [
    ([[{"is_on_screen": False, "on_current_space": False}]], "another desktop"),
    ([[{"is_on_screen": False, "on_current_space": True}]], "hidden or minimised"),
    ([[]], "opened no window"),
])
def test_when_it_cannot_it_says_why_so_the_agent_stops_guessing_names(host_on, states, words):
    p = apps.bring_up("Music", Windows(states), wait=0.5, launcher=lambda a: None)
    assert not p.ok and words in p.reason and "AppleScript" in p.reason


def test_a_window_that_shows_but_whose_controls_never_resolve_is_reported_as_such(host_on):
    p = apps.bring_up("Music", Windows([[{"is_on_screen": True, "on_current_space": True}]], readable=False), wait=0.6, launcher=lambda a: None)
    assert not p.ok and "controls could not be read" in p.reason


def test_a_launcher_that_times_out_does_not_fail_the_turn(host_on):
    import subprocess

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 15)

    apps.reopen_and_activate("Music", runner=slow)  # no exception: what matters is whether a window appears
    p = apps.bring_up("Music", Windows([[{"is_on_screen": True}]]), launcher=lambda a: (_ for _ in ()).throw(subprocess.TimeoutExpired("x", 1)))
    assert p.ok


def test_nothing_is_launched_when_host_actions_are_off():
    launched = []
    p = apps.bring_up("Music", Windows([[]]), launcher=launched.append)  # the test session sets the switch off
    assert not p.ok and launched == [] and "switched off" in p.reason


def test_the_app_name_travels_as_an_argument_never_inside_the_script():
    calls = []
    apps.reopen_and_activate('Music" & (do shell script "rm -rf ~") & "', runner=lambda cmd, **kw: calls.append(cmd))
    script = [c for c in calls if c[0] == "osascript"][0]
    assert all("rm -rf" not in part for part in script[:-1]) and script[-1].startswith("Music")


# --- the agent tool and the shortcut that opens an app before any model runs ---------------------------------------------------

def test_the_open_app_tool_finds_the_app_says_what_it_can_do_and_never_guesses(monkeypatch, catalog):
    import types

    from laya_assistant import host

    monkeypatch.setenv("LAYA_HOST_ACTIONS", "on")
    monkeypatch.setattr(host, "_CATALOG", catalog)
    monkeypatch.setattr(apps, "bring_up", lambda name, cli, **kw: apps.Presence(True, name, 1, 2))
    tool = {t.name: t for t in host.make_host_tools()}["mac_open_app"]
    ok = tool.invoke({"name": "apple music"})
    assert ok.startswith("Opened Music; its window is on screen") and "scriptable" in ok  # Music ships a dictionary in the fixture: the method note says to script it
    assert tool.invoke({"name": "photoshop"}).startswith("UNAVAILABLE") and "Do not guess" in tool.invoke({"name": "photoshop"})
    monkeypatch.setattr(apps, "bring_up", lambda name, cli, **kw: apps.Presence(False, name, reason="its window is on another desktop"))
    assert "another desktop" in tool.invoke({"name": "notes"})


def test_the_tool_does_nothing_when_host_actions_are_off():
    from laya_assistant import host

    assert {t.name: t for t in host.make_host_tools()}["mac_open_app"].invoke({"name": "Music"}) == host.SWITCHED_OFF


def test_a_request_that_starts_with_open_app_opens_it_before_any_model_runs(monkeypatch):
    from laya_assistant.session import AssistantSession
    from laya_assistant.decisions import DecisionLog

    opened = []
    monkeypatch.setattr(apps, "reopen_and_activate", lambda name, **kw: opened.append(name))
    monkeypatch.setenv("LAYA_HOST_ACTIONS", "on")
    s = AssistantSession.__new__(AssistantSession)
    s.log = DecisionLog()
    assert s._open_first_clause("open Notes") == ("Notes", True)
    assert s._open_first_clause("can you open the notes app") == ("Notes", True)
    assert s._open_first_clause("open notes and write a shopping list") == ("Notes", False)  # the rest is still for the agent
    assert s._open_first_clause("write a shopping list") is None and s._open_first_clause("open terminal") is None  # power apps never auto-open
    assert opened == ["Notes", "Notes", "Notes"]
    monkeypatch.setenv("LAYA_HOST_ACTIONS", "off")
    assert s._open_first_clause("open Notes") is None and len(opened) == 3
