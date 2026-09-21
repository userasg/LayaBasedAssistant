"""Verified AppleScript templates: the common "do X in <app>" requests run with no model writing a script.
The Music app itself is never touched here: the runner is recorded (osacompile only checks the scripts are valid AppleScript)."""
import shutil
import subprocess
import types

import pytest

from laya_assistant import app_actions as aa
from laya_assistant.cua.apps import AppCatalog


@pytest.fixture
def catalog(tmp_path):
    for name in ("Music", "Spotify", "Notes", "Google Chrome"):
        (tmp_path / f"{name}.app").mkdir()
    return AppCatalog(dirs=(str(tmp_path),))


@pytest.mark.parametrize("text,app,action,arg", [
    ("can you open apple music and play a song from my odd future playlist", "Music", "play_playlist", "odd future"),
    ("open Music and play my Road Trip playlist", "Music", "play_playlist", "road trip"),
    ("play the playlist chill vibes in Music", "Music", "play_playlist", "chill vibes"),
    ("play the song Redbone in Music", "Music", "play_song", "redbone"),
    ("open apple music and play rock and roll", "Music", "play_song", "rock and roll"),  # "and" inside a title is not a clause boundary
    ("pause the music in Music", "Music", "pause", ""),
    ("please skip the song on spotify", "Spotify", "next", ""),
    ("what's playing in Music?", "Music", "now_playing", ""),
    ("open Music and pause", "Music", "pause", ""),
])
def test_common_requests_parse_to_a_template_call(catalog, text, app, action, arg):
    got = aa.parse_request(text, catalog)
    assert got is not None and (got.app, got.action, got.arg) == (app, action, arg)


@pytest.mark.parametrize("text", [
    "open Notes and write a shopping list",  # no template: the agent handles it
    "play a song",  # which app? not guessable: the agent decides
    "delete my odd future playlist in Music",
    "open Music",  # just opening is the open-app path
    "play Redbone on Slack",  # Slack has no template
    "play Redbone on Spotify",  # Spotify is only scripted for transport controls: a song search needs the agent
    "hi there",
])
def test_anything_else_is_left_to_the_agent(catalog, text):
    assert aa.parse_request(text, catalog) is None


def test_the_argument_travels_as_data_never_inside_the_script():
    argv = aa.script_argv("Music", "play_playlist", 'x" & (do shell script "rm -rf ~") & "')
    assert argv[:2] == ["osascript", "-e"] and "rm -rf" not in argv[2] and argv[-1].startswith('x"')


@pytest.mark.skipif(shutil.which("osacompile") is None or not __import__("os").path.isdir("/System/Applications/Music.app"), reason="needs macOS with Music")
@pytest.mark.parametrize("action", [a for (app, a) in aa.TEMPLATES if app == "Music"])
def test_every_music_template_is_valid_applescript(action, tmp_path):
    p = subprocess.run(["osacompile", "-o", str(tmp_path / "x.scpt"), "-e", aa.script_argv("Music", action)[2]], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def _runner(stdout="", stderr="", code=0):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return types.SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)

    return calls, run


def test_a_verified_play_reports_what_is_actually_playing():
    calls, runner = _runner(stdout="playing | Nightcrawler — Travis Scott\n")
    r = aa.run("Music", "play_playlist", "odd future", runner=runner, env={"PATH": "/usr/bin"})
    assert r.ok and r.final and r.text == "Playing Nightcrawler — Travis Scott."
    assert calls[0][0][0] == "osascript" and calls[0][1]["env"] == {"PATH": "/usr/bin"}


def test_exit_zero_is_not_success_the_player_must_report_playing():
    _, runner = _runner(stdout="paused | \n")
    r = aa.run("Music", "play_playlist", "odd future", runner=runner)
    assert not r.ok and "paused" in r.text  # said honestly, not "should be starting shortly"


def test_a_missing_playlist_is_a_clear_answer_not_a_retry():
    _, runner = _runner(stderr="execution error: Music got an error: Can’t get first user playlist whose name contains \"zzz\". (-1728)", code=1)
    r = aa.run("Music", "play_playlist", "zzz", runner=runner)
    assert not r.ok and r.final and "zzz" in r.text  # the answer for the user: no model turn needed


def test_any_other_script_error_hands_over_to_the_agent_with_the_error():
    _, runner = _runner(stderr="execution error: not authorised to send Apple events to Music. (-1743)", code=1)
    r = aa.run("Music", "pause", "", runner=runner)
    assert not r.ok and not r.final and "-1743" in r.text


def test_there_is_a_template_list_for_the_agent_tool():
    assert "play_playlist" in aa.describe() and "Music" in aa.describe() and "Spotify" in aa.describe()


# -- the session's zero-model path -------------------------------------------------------------------------------------------------

def _session(monkeypatch, catalog, result):
    from laya_assistant import approvals
    from laya_assistant.decisions import DecisionLog
    from laya_assistant.session import AssistantSession

    monkeypatch.setenv("LAYA_HOST_ACTIONS", "on")
    monkeypatch.setattr(approvals, "current", approvals.Approvals())
    ran = []
    monkeypatch.setattr(aa, "run", lambda app, action, arg="", **kw: ran.append((app, action, arg)) or result)
    s = AssistantSession.__new__(AssistantSession)
    s.log = DecisionLog()
    s._app_index = types.SimpleNamespace(catalog=catalog)
    return s, ran


def test_a_recognised_request_runs_its_template_with_no_model(monkeypatch, catalog):
    s, ran = _session(monkeypatch, catalog, aa.Result(True, "Playing Nightcrawler — Travis Scott.", final=True))
    r = s._direct_action("open apple music and play a song from my odd future playlist")
    assert r.ok and ran == [("Music", "play_playlist", "odd future")]
    assert any(row["step"] == "direct action" for row in s.log.rows)


def test_an_unrecognised_request_or_host_actions_off_never_run_anything(monkeypatch, catalog):
    s, ran = _session(monkeypatch, catalog, aa.Result(True, "x", final=True))
    assert s._direct_action("open Notes and write a shopping list") is None
    monkeypatch.setenv("LAYA_HOST_ACTIONS", "off")
    assert s._direct_action("pause the music in Music") is None and ran == []


def test_ask_for_every_command_is_respected_the_template_does_not_run_silently(monkeypatch, catalog):
    from laya_assistant import approvals

    s, ran = _session(monkeypatch, catalog, aa.Result(True, "x", final=True))
    approvals.current.host_commands = "always_ask"
    assert s._direct_action("pause the music in Music") is None and ran == []


def test_the_agent_tool_runs_a_template_and_lists_them_when_there_is_none(monkeypatch, catalog):
    from laya_assistant import host

    monkeypatch.setattr(host, "_catalog", lambda: catalog)
    calls, runner = _runner(stdout="playing | Redbone — Childish Gambino\n")
    tool = {t.name: t for t in host.make_host_tools(runner=runner)}["mac_app_action"]
    assert tool.invoke({"app": "apple music", "action": "play_song", "arg": "redbone"}) == "Playing Redbone — Childish Gambino."
    out = tool.invoke({"app": "Notes", "action": "play_song", "arg": "x"})
    assert out.startswith("NO_TEMPLATE") and "Music" in out and len(calls) == 1


# -- generic, any app --------------------------------------------------------------------------------------------------------------

SDEF = """<?xml version="1.0"?><dictionary>
<suite name="Standard Suite"><command name="close" code="x"/></suite>
<suite name="Mail Suite">
  <command name="send" code="x"><direct-parameter type="message"/><parameter name="with attachments" type="file" optional="yes"/></command>
  <class name="message" code="x"><property name="subject" type="text"/><property name="sender" type="text"/><element type="attachment"/></class>
  <class name="attachment" code="y"><property name="name" type="text"/></class>
  <class name="account" code="z"><element type="message"/></class>
</suite></dictionary>"""


def test_the_dictionary_digest_lists_commands_and_the_classes_scripts_are_about():
    from laya_assistant.cua import sdef

    d = sdef.digest_xml("Mail", SDEF)
    assert 'tell application "Mail"' in d and "send <message> [with attachments <file>]" in d and "close" not in d  # standard suite skipped
    assert "message props: subject, sender | contains: attachment" in d
    assert sdef.digest_xml("X", "<dictionary/>") == "" and sdef.digest_xml("X", "not xml") == ""
    assert len(sdef.digest_xml("Big", SDEF.replace("<class", "<class name='pad' code='q'><property name='" + "p" * 3000 + "'/></class><class", 1))) <= sdef.BUDGET


@pytest.mark.skipif(not __import__("os").path.isdir("/System/Applications/Notes.app"), reason="needs macOS")
def test_a_real_app_bundle_yields_a_digest_and_an_unscriptable_one_yields_nothing(tmp_path):
    from laya_assistant.cua import sdef

    assert "note" in sdef.digest("Notes", "/System/Applications/Notes.app")
    assert sdef.digest("Nope", str(tmp_path)) == "" and sdef.digest("Nope", None) == ""


class FakeCatalog:
    """Installed-app facts without touching the machine: Notes is scriptable (with a dictionary), Slack is not."""
    def __init__(self, cat):
        self.cat = cat

    def resolve(self, s):
        return self.cat.resolve(s)

    def scripting(self, name):
        return name in ("Music", "Notes")

    def dictionary(self, name):
        return f"{name} scripting dictionary: (digest)" if self.scripting(name) else ""


def test_the_app_a_request_is_about_is_found_in_code(catalog):
    for text, want in [("write a note in Notes about my day", "Notes"), ("open the notes app and add milk", "Notes"),
                       ("send a message on Spotify", "Spotify"), ("summarise this article", None), ("do it in the morning", None)]:
        got = aa.find_app(text, catalog)
        assert (got.name if got else None) == want, text


def test_script_or_ui_is_decided_from_the_apps_dictionary_not_guessed(catalog):
    fc = FakeCatalog(catalog)
    scriptable = aa.method_note("Notes", fc)
    assert "scriptable" in scriptable and "at most one attempt" in scriptable and "refuses more scripts" in scriptable and "Notes scripting dictionary" in scriptable
    ui = aa.method_note("Google Chrome", fc)
    assert ui == ""  # browsers are for computer_use's own browser
    cat2 = AppCatalog(dirs=())
    assert "no scripting dictionary" in aa.method_note("Slack", FakeCatalog(cat2)) and "computer_use(target='desktop', app='Slack')" in aa.method_note("Slack", FakeCatalog(cat2))


def test_a_failed_script_is_explained_and_two_failures_switch_to_the_ui(monkeypatch, catalog):
    from laya_assistant import host

    err = "28:54: execution error: Notes got an error: Can’t make some data into the expected type. (-1700)"
    calls, runner = _runner(stderr=err, code=1)
    mac_run = {t.name: t for t in host.make_host_tools(runner=runner)}["mac_run"]
    cmd = "osascript -e 'tell application \"Notes\" to make new note'"
    first = mac_run.invoke({"command": cmd})
    assert "[hint]" in first and "wrong type" in first and "computer_use" not in first
    second = mac_run.invoke({"command": cmd})
    assert "failed 2 times" in second and "computer_use(target='desktop', app='Notes')" in second


def test_a_working_script_resets_the_failure_count(monkeypatch):
    from laya_assistant import host

    state = {"code": 1}
    runner = lambda cmd, **kw: types.SimpleNamespace(returncode=state["code"], stdout="ok", stderr="(-1728)" if state["code"] else "")
    mac_run = {t.name: t for t in host.make_host_tools(runner=runner)}["mac_run"]
    cmd = "osascript -e 'tell application \"Notes\" to get name of every note'"
    mac_run.invoke({"command": cmd})
    state["code"] = 0
    assert "[hint]" not in mac_run.invoke({"command": cmd})
    state["code"] = 1
    assert "failed" not in mac_run.invoke({"command": cmd}).split("[hint]")[-1]  # counting started over
