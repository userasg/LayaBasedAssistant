"""How a request about a Mac app gets done fast, for ANY app. Three layers, cheapest first:
  1. `method_note` / `find_app` (generic): a fact from the app bundle (does it ship a scripting dictionary?) decides script vs UI in code, and a
     scriptable app's dictionary digest is handed to the agent so its first script is right; other apps skip straight to computer_use.
  2. `osascript_advice` / fail-over (generic): a failed script is explained by its error code, and after two failures in one app the agent is
     told to switch to the UI instead of burning model turns.
  3. TEMPLATES (optional recipes, currently Music and Spotify playback): the most common verbs as fixed, verified scripts that need no model.
     Add an app by adding entries to TEMPLATES; nothing else changes.

Verified AppleScript templates for the things people ask a Mac app to do most: play, pause, skip, what is playing.

A scriptable app (see cua/apps.py) is driven far faster and more reliably by a script than by clicking, but a 14B model that WRITES the
script gets types wrong (`-1700`), retries, and says "should be starting shortly" on exit 0. So the common verbs are fixed here:
  * the script text is a constant; the user's words travel as an ARGUMENT (`item 1 of argv`), never spliced in (no injection);
  * each script ends by reading the player back, so "success" means the app reports `playing`, not just that osascript exited 0;
  * `parse_request` recognises the request in code (~0 ms), so a whole turn needs no model call at all; anything it does not
    recognise returns None and goes to the agent exactly as before.
Only reversible playback controls live here: nothing that sends, pays or deletes.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

TIMEOUT_S = 20


@dataclass(frozen=True)
class Template:
    body: str  # AppleScript run inside `tell application "<app>"`; `item 1 of argv` is the user's text
    expect_playing: bool = False  # the player must report `playing` afterwards
    says: str = ""  # a reply when the app does not report a track ("Paused.")
    missing: str = "I couldn't find {arg!r} in {app}."  # the reply for a -1728 "can't get ..." error


_FIND_PLAYLIST = '''try
      set pl to first playlist whose name is (item 1 of argv)
    on error
      set pl to first playlist whose name contains (item 1 of argv)
    end try'''

_MUSIC_PLAYLIST_MISSING = "I couldn't find a playlist called {arg!r} in {app} (or it has no songs)."

_TRANSPORT = {
    "pause": Template("pause", says="Paused."),
    "resume": Template("play", expect_playing=True),
    "next": Template("next track", expect_playing=True),
    "previous": Template("previous track", expect_playing=True),
    "now_playing": Template("", says="Nothing is playing."),
}

TEMPLATES: dict[tuple[str, str], Template] = {
    ("Music", "play_playlist"): Template(_FIND_PLAYLIST + "\n    play (some track of pl)", expect_playing=True, missing=_MUSIC_PLAYLIST_MISSING),
    ("Music", "play_song"): Template("play (first track of library playlist 1 whose name contains (item 1 of argv))", expect_playing=True,
                                     missing="I couldn't find a song called {arg!r} in your {app} library."),
    **{("Music", k): v for k, v in _TRANSPORT.items()},
    **{("Spotify", k): v for k, v in _TRANSPORT.items()},
}

_SCRIPT = '''on run argv
  tell application "{app}"
    {body}
    {settle}
    set s to (player state as string)
    try
      set s to s & " | " & (name of current track) & " — " & (artist of current track)
    end try
    return s
  end tell
end run'''


@dataclass
class Call:
    app: str
    action: str
    arg: str = ""


@dataclass
class Result:
    ok: bool
    text: str  # what to tell the user (ok/final) or what went wrong (handed to the agent)
    final: bool = False  # True: the answer is settled (e.g. "no such playlist"), no model turn can improve it


def script_argv(app: str, action: str, arg: str = "") -> list[str]:
    t = TEMPLATES[(app, action)]
    script = _SCRIPT.format(app=app, body=t.body, settle="delay 0.6" if t.expect_playing else "")
    return ["osascript", "-e", script, arg]


def describe() -> str:
    by_app: dict[str, list[str]] = {}
    for app, action in TEMPLATES:
        by_app.setdefault(app, []).append(action + ("(arg)" if action.startswith("play_") else ""))
    return "; ".join(f"{app}: {', '.join(acts)}" for app, acts in by_app.items())


def run(app: str, action: str, arg: str = "", runner=subprocess.run, env: dict | None = None) -> Result:
    t = TEMPLATES[(app, action)]
    kw = {"capture_output": True, "text": True, "timeout": TIMEOUT_S}
    if env is not None:
        kw["env"] = env
    try:
        p = runner(script_argv(app, action, arg), **kw)
    except subprocess.TimeoutExpired:
        return Result(False, f"{app} did not answer within {TIMEOUT_S} s.")
    if p.returncode != 0:
        err = (p.stderr or "").strip()
        if "(-1728)" in err:
            return Result(False, t.missing.format(arg=arg, app=app), final=True)
        return Result(False, err[:200] or f"osascript exited {p.returncode}")
    state, _, track = (p.stdout or "").strip().partition(" | ")
    track = track.strip()
    if t.expect_playing and state.strip() != "playing":
        return Result(False, f"{app} reports {state.strip() or 'no state'!r}, not playing.")
    if action == "now_playing":
        return Result(True, f"{app} is {state.strip()}: {track}." if track else t.says, final=True)
    if t.expect_playing:
        return Result(True, f"Playing {track}." if track else "Playing.", final=True)
    return Result(True, t.says or "Done.", final=True)


# -- recognising the request ------------------------------------------------------------------------------------------------------

_POLITE = re.compile(r"^(?:(?:hey|hi|ok|okay)\s+)?(?:(?:can|could|would|will) you\s+)?(?:please\s+)?")
_OPEN_THEN = re.compile(r"^(?:open|launch|start)\s+(?P<app>.+?)\s+(?:and then|and|then)\s+(?P<rest>.+)$")
_IN_APP = re.compile(r"^(?P<rest>.+)\s+(?:in|on|with|using)\s+(?:the\s+)?(?P<app>[\w .+-]+?)(?:\s+app)?$")
_THE_MUSIC = r"(?: (?:the |some )?(?:music|song|songs|track|tracks|playback|playing))?"
_GENERIC = {"music", "something", "a song", "song", "songs", "anything", "a track", "a tune"}

_ACTIONS = [
    ("play_playlist", re.compile(r"^play (?:(?:a|some|any) )?(?:songs?|tracks?|music|something|tunes?) (?:from|off|out of|in) (?:my |the )?(?P<arg>.+?) playlist$")),
    ("play_playlist", re.compile(r"^play (?:my |the )?(?P<arg>.+?) playlist$")),
    ("play_playlist", re.compile(r"^play (?:my |the )?playlist (?P<arg>.+)$")),
    ("resume", re.compile(r"^(?:play|resume|unpause)" + _THE_MUSIC + "$")),
    ("play_song", re.compile(r"^play (?:the |a )?(?:(?:song|track) )?(?P<arg>.+)$")),
    ("pause", re.compile(r"^(?:pause|stop)" + _THE_MUSIC + "$")),
    ("next", re.compile(r"^(?:next|skip)(?: (?:to )?(?:the )?(?:next )?(?:song|track))?$")),
    ("previous", re.compile(r"^(?:previous|go back|last)(?: (?:to )?(?:the )?(?:previous |last )?(?:song|track))?$")),
    ("now_playing", re.compile(r"^(?:what(?:'s| is)|whats) (?:playing|on)$|^(?:now playing|what song is (?:this|playing))$")),
]


def parse_request(text: str, catalog) -> Call | None:
    """A Call when the WHOLE request is one known action in one known app, else None (the agent handles it)."""
    t = " ".join(text.lower().replace("’", "'").split()).rstrip(" ?.!,")
    t = _POLITE.sub("", t)
    t = re.sub(r"\s+please$", "", t)
    app_text = None
    if m := _OPEN_THEN.match(t):
        app_text, t = m.group("app"), m.group("rest")
    elif m := _IN_APP.match(t):
        app_text, t = m.group("app"), m.group("rest")
    resolved = catalog.resolve(app_text) if app_text else None
    if resolved is None:
        return None
    for action, rx in _ACTIONS:
        m = rx.match(t)
        if not m:
            continue
        arg = (m.groupdict().get("arg") or "").strip()
        if action == "play_song" and arg in _GENERIC:
            action, arg = "resume", ""
        if (resolved.name, action) in TEMPLATES:
            return Call(resolved.name, action, arg)
        return None  # recognised, but this app has no template for it
    return None


# -- generic, for ANY app: which method, and what to do when a script fails ------------------------------------------------------------

_MENTION = re.compile(r"\b(?:in|on|with|using|open|launch|start)\s+(?:the\s+)?(?P<w>[\w.+-]+(?:\s+[\w.+-]+){0,2})")
# Browsers are for websites, which computer_use's own browser handles better than AppleScript; never steer those away from it.
_BROWSERS = {"Safari", "Google Chrome", "Firefox", "Arc", "Brave Browser", "Microsoft Edge", "Opera", "Chromium"}


def find_app(text: str, catalog):
    """The installed app a request is about ("... in Notes", "open Slack and ..."), else None. A hint only; pure code."""
    t = text.lower().replace("’", "'")
    for m in _MENTION.finditer(t):
        words = [w for w in m.group("w").split() if w not in _FILLER_WORDS]  # "open my messages app": the app is "messages"
        for n in range(len(words), 0, -1):
            if r := catalog.resolve(" ".join(words[:n])):
                return r
    return None


def method_note(name: str, catalog, methods=None) -> str:
    """Which way in, decided from facts in code rather than guessed by the model. The ladder: a scriptable app is driven by AppleScript, an app with
    a command line by that, anything else through its window (computer_use). `methods` (cua/methods.py) moves an app to the rung that has actually
    worked here: a script the model writes freehand can fail where the window does not (Messages), and that is learnt once, not per request."""
    if name in _BROWSERS:
        return ""
    ui = f"computer_use(target='desktop', app='{name}')"
    if methods is not None and methods.prefers_ui(name):
        return f"[Mac method: scripting {name} has not worked on this Mac, so work in its window: {ui}. Do not write AppleScript for it.]"
    if catalog.scripting(name):
        d = catalog.dictionary(name)
        return (f"[Mac method: {name} is scriptable: drive it with AppleScript via mac_run (mac_app_action if it lists the action). Make at most one "
                f"attempt and one repair from the error hint; if it fails twice, mac_run refuses more scripts for {name} and you use {ui}. Pass the "
                f"user's words as script text only if they are plain names. End the script by reading the result back so you can confirm it worked."
                f"{chr(10) + d if d else ''}]")
    cli = getattr(catalog, "cli", lambda n: None)(name)
    if cli:
        return (f"[Mac method: {name} has no scripting dictionary but has a command line ({cli}): use mac_run for what it can do (open a folder or "
                f"file with `{cli} <path>`; create folders and files with shell commands under the user's home, using the full path) and {ui} only for "
                f"what must be clicked or typed in its window.]")
    return f"[Mac method: {name} has no scripting dictionary, so AppleScript cannot drive it: use {ui} directly.]"


_FILLER_WORDS = {"my", "the", "app", "application", "a", "our"}
_TELL = re.compile(r'tell application (?:\\?"|\\?\')([^"\'\\]+)', re.I)
_OSA_ADVICE = [
    ("-1700", "A value had the wrong type. Check that property's type in the dictionary; do not coerce with `as`, and pass names as text."),
    ("-1728", "That object does not exist. List what is there first (for example `name of every playlist`), then use an exact name or `whose name contains`."),
    ("-1743", "macOS blocked the automation. The user must allow it in System Settings > Privacy & Security > Automation. Do not retry; use computer_use."),
    ("-1744", "macOS blocked the automation. The user must allow it in System Settings > Privacy & Security > Automation. Do not retry; use computer_use."),
    ("-1708", "The app does not understand that command. Use only commands from its dictionary."),
    ("-10004", "A privilege error: the app refused this command. Use computer_use instead."),
]
SWITCH_AFTER = 2  # failed scripts in one app before the agent is told to stop scripting it


def osascript_app(command: str) -> str | None:
    m = _TELL.search(command)
    return m.group(1) if m else None


def osascript_advice(stderr: str) -> str:
    return next((a for code, a in _OSA_ADVICE if f"({code})" in stderr), "")
