"""Desktop driver. The real cua-driver binary is exercised up to the macOS permission boundary; the parsing of window
state uses a payload in the documented shape (the daemon needs Accessibility + Screen Recording grants only you can give)."""
import json
import subprocess

import pytest

from laya_assistant.cua.desktop import CuaCli, DesktopDriver, elements_from_state, find_binary
from laya_assistant.cua.types import Action

HAVE_BINARY = find_binary() is not None
STATE = {  # shape documented by `cua-driver describe get_window_state`
    "window_title": "Untitled", "app_name": "Notes", "tree_markdown": "- [element_index 3] AXTextArea",
    "elements": [
        {"element_index": 1, "role": "AXButton", "label": "New Note", "actions": ["AXPress"], "frame": {"x": 1, "y": 2, "w": 3, "h": 4}, "parent_index": 0, "depth": 1},
        {"element_index": 2, "role": "AXStaticText", "label": "", "depth": 1},
        {"element_index": 3, "role": "AXTextArea", "label": "Note body", "value": "hello", "actions": []},
        {"element_index": 4, "role": "AXSecureTextField", "label": "Password", "value": ""},
        {"element_index": 5, "role": "AXPopUpButton", "label": "Font", "value": "Helvetica", "actions": ["AXPress"]},
        {"element_index": 6, "role": "AXCheckBox", "label": "Checklist", "actions": ["AXPress"]},
    ],
}


def test_ax_elements_map_onto_the_shared_element_model():
    els = {e.id: e for e in elements_from_state(STATE)}
    assert "e2" not in els  # unlabelled static text is noise
    assert (els["e1"].role, els["e1"].label) == ("button", "New Note")
    assert els["e3"].role == "textbox" and els["e3"].value == "hello"
    assert els["e4"].type == "password"  # so the policy hands passwords to the user on the desktop too
    assert els["e5"].role == "select" and els["e6"].role == "checkbox"


class FakeCli:
    """Stands in for the third-party daemon in driver-logic tests: records calls, returns documented shapes."""

    def __init__(self):
        self.calls = []

    def call(self, tool, args=None, timeout=None):
        self.calls.append((tool, args))
        if tool == "list_windows":
            return {"windows": [{"window_id": 7, "pid": 42, "app_name": "Notes", "title": "Untitled", "z_index": 1},
                                {"window_id": 8, "pid": 42, "app_name": "Notes", "title": "Other", "z_index": 5},
                                {"window_id": 9, "pid": 99, "app_name": "Finder", "title": "Downloads", "z_index": 9}]}
        if tool == "get_window_state":
            return STATE
        return {}


def test_the_driver_picks_the_frontmost_window_of_its_app_and_observes_it():
    cli = FakeCli()
    obs = DesktopDriver("Notes", cli).observe()
    assert obs.source == "desktop" and obs.app == "Notes" and any(e.label == "New Note" for e in obs.elements)
    tool, args = cli.calls[-1]
    assert tool == "get_window_state" and args["pid"] == 42 and args["window_id"] == 8  # z_index 5 beats 1; Finder ignored


def test_actions_address_elements_by_index_and_use_the_right_tools():
    cli = FakeCli()
    d = DesktopDriver("Notes", cli)
    d.observe()
    d.act(Action("click", id="e1"))
    d.act(Action("type", id="e3", value="milk, eggs"))
    d.act(Action("select", id="e5", value="Courier"))
    d.act(Action("key", id="e3", key="Return"))
    d.act(Action("open_app", value="Calculator"))
    tools = [(t, {k: v for k, v in a.items() if k in ("element_index", "value", "key", "name")}) for t, a in cli.calls[2:] if t not in ("list_windows", "get_window_state")]
    assert tools == [("click", {"element_index": 1}), ("set_value", {"element_index": 3, "value": "milk, eggs"}),
                     ("set_value", {"element_index": 5, "value": "Courier"}), ("press_key", {"element_index": 3, "key": "return"})]
    # opening an app is bring_up (open, reopen, activate, wait for a window), not the driver's background-only launch_app; inert in test runs
    assert d.act(Action("open_app", value="Calculator"))["ok"] is False


def test_a_driver_error_becomes_a_failed_action_not_an_exception():
    class Broken(FakeCli):
        def call(self, tool, args=None, timeout=None):
            if tool == "click":
                raise RuntimeError("element gone")
            return super().call(tool, args)

    d = DesktopDriver("Notes", Broken())
    d.observe()
    assert d.act(Action("click", id="e1")) == {"ok": False, "error": "element gone"}


def test_unsupported_actions_are_refused_clearly():
    d = DesktopDriver("Notes", FakeCli())
    d.observe()
    assert d.act(Action("navigate", url="https://x.y"))["ok"] is False


@pytest.mark.skipif(not HAVE_BINARY, reason="cua-driver binary not found")
def test_the_real_binary_is_found_and_lists_its_tools():
    out = subprocess.run([find_binary(), "list-tools"], capture_output=True, text=True, timeout=30).stdout
    for tool in ("get_window_state", "click", "set_value", "press_key", "list_windows", "launch_app", "verify_state"):
        assert tool + ":" in out


@pytest.mark.skipif(not HAVE_BINARY, reason="cua-driver binary not found")
def test_status_is_actionable_whichever_state_the_real_driver_is_in():
    st = CuaCli().status()
    print("desktop status:", st["state"])
    if st["available"]:
        pytest.skip("permissions are granted and the daemon is running: see the live test")
    assert st["installed"] and st["state"] in ("daemon_down", "needs_permission")
    # daemon_down says how to start it; needs_permission says what to press. Never a bare error.
    assert "cua-driver" in st["detail"] or "Grant permissions" in st["detail"]


@pytest.mark.skipif(not (HAVE_BINARY and CuaCli().status()["available"]), reason="Cua Driver daemon not running or permissions not granted yet")
def test_live_desktop_calculator_round_trip():
    import os
    import subprocess

    os.environ["LAYA_HOST_ACTIONS"] = "on"  # this test drives the real app on purpose
    try:
        d = DesktopDriver("Calculator")
        presence = d.launch()
        assert presence.ok, presence.reason
        obs = d.observe()
        assert obs.app == "Calculator" and obs.elements
    finally:
        os.environ["LAYA_HOST_ACTIONS"] = "off"
        subprocess.run(["osascript", "-e", 'tell application "Calculator" to quit'], capture_output=True, timeout=10)


# --- the no-setup browser path: Chrome driven through its accessibility tree -------------------------------------------

from laya_assistant.cua.desktop import (PERMISSION_HELP, address_bar, ensure_daemon, grant_permissions, normalise_url)
from laya_assistant.cua.types import Element

CHROME_STATE = {
    "window_title": "Acme Shop - Home - Google Chrome", "app_name": "Google Chrome",
    "elements": [
        {"element_index": 1, "role": "AXTextField", "label": "Address and search bar", "value": "shop.example.com/index.html", "actions": []},
        {"element_index": 2, "role": "AXButton", "label": "Reload this page", "actions": ["AXPress"]},
        {"element_index": 10, "role": "AXTextField", "label": "Search products", "value": "", "actions": []},
        {"element_index": 11, "role": "AXButton", "label": "Search", "actions": ["AXPress"]},
        {"element_index": 12, "role": "AXLink", "label": "Contact us", "actions": ["AXPress"]},
        {"element_index": 13, "role": "AXLink", "label": "Sign in", "actions": ["AXPress"]},
    ],
}


class ChromeCli(FakeCli):
    def call(self, tool, args=None, timeout=None):
        if tool == "list_windows":
            self.calls.append((tool, args))
            return {"windows": [{"window_id": 3, "pid": 77, "app_name": "Google Chrome", "title": "Acme Shop", "z_index": 4}]}
        if tool == "get_window_state":
            self.calls.append((tool, args))
            return CHROME_STATE
        return super().call(tool, args)


@pytest.mark.parametrize("raw,url", [("shop.example.com/index.html", "https://shop.example.com/index.html"), ("https://a.b/c", "https://a.b/c"),
                                     ("localhost:8501", "https://localhost:8501"), ("search or type URL", None), ("", None), ("hello", None)])
def test_urls_are_read_from_the_address_bar_text(raw, url):
    assert normalise_url(raw) == url


def test_a_browser_window_is_observed_as_a_browser_with_its_url():
    obs = DesktopDriver("", ChromeCli(), browser=True).observe()
    assert obs.source == "browser" and obs.app == "Google Chrome" and obs.url == "https://shop.example.com/index.html"
    assert obs.host == "shop.example.com"  # so the domain policy (blocked banks, first-visit asks) applies exactly as with the extension
    assert address_bar(obs.elements).label == "Address and search bar" and any(e.label == "Contact us" for e in obs.elements)


def test_the_installed_browser_that_is_open_is_used_without_being_told():
    class Safari(ChromeCli):
        def call(self, tool, args=None, timeout=None):
            if tool == "list_windows":
                return {"windows": [{"window_id": 1, "pid": 5, "app_name": "Safari", "title": "x", "z_index": 1}]}
            return super().call(tool, args)

    d = DesktopDriver("", Safari(), browser=True)
    d._window()
    assert d.app == "Safari"


def test_navigating_types_into_the_address_bar_and_presses_return():
    cli = ChromeCli()
    d = DesktopDriver("", cli, browser=True)
    d.observe()
    assert d.act(Action("navigate", url="https://example.org/x"))["ok"]
    tail = [(t, {k: v for k, v in a.items() if k in ("element_index", "value", "key")}) for t, a in cli.calls[-2:]]
    assert tail == [("set_value", {"element_index": 1, "value": "https://example.org/x"}), ("press_key", {"element_index": 1, "key": "return"})]


def test_navigation_falls_back_to_the_keyboard_when_the_address_bar_is_not_visible():
    class NoBar(ChromeCli):
        def call(self, tool, args=None, timeout=None):
            if tool == "get_window_state":
                self.calls.append((tool, args))
                return {**CHROME_STATE, "elements": CHROME_STATE["elements"][1:]}
            return super().call(tool, args)

    cli = NoBar()
    d = DesktopDriver("", cli, browser=True)
    d.observe()
    d.act(Action("navigate", url="https://example.org"))
    assert [t for t, _ in cli.calls[-3:]] == ["hotkey", "type_text", "press_key"]
    assert cli.calls[-3][1]["keys"] == ["cmd", "l"]


def test_a_browser_that_exposes_no_accessibility_tree_gives_an_actionable_error():
    class Blind(ChromeCli):
        def call(self, tool, args=None, timeout=None):
            if tool == "get_window_state":
                return {"window_title": "x", "elements": []}
            return super().call(tool, args)

    with pytest.raises(RuntimeError, match="chrome://accessibility"):
        DesktopDriver("", Blind(), browser=True).observe()


def test_status_states_are_read_from_the_drivers_real_messages():
    class Cli(CuaCli):
        def __init__(self, reply):
            self.binary, self.runner, self.timeout, self.reply = "/x", None, 1, reply

        def call(self, tool, args=None, timeout=None):
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

    assert Cli(RuntimeError("Cua Driver is not running. One-time setup")).status()["state"] == "daemon_down"
    pending = Cli({"text": "permissions_pending: macOS Accessibility or Screen Recording permission is still pending"}).status()
    assert pending["state"] == "needs_permission" and pending["detail"] == PERMISSION_HELP and not pending["available"]
    assert Cli({"accessibility": "granted", "screen_recording": "granted"}).status()["state"] == "ready"
    # the real payload shape (seen from the running daemon): explicit booleans, plus unrelated null/false-looking fields
    real = {"accessibility": True, "screen_recording": True, "screen_recording_capturable": None, "direct_capture_error": None, "direct_capture_status": "not_checked"}
    assert Cli(real).status()["state"] == "ready"
    assert Cli({**real, "screen_recording": False}).status()["state"] == "needs_permission"


def test_grant_permissions_launches_the_os_dialogs_and_nothing_else():
    launched = []
    grant_permissions(CuaCli(binary="/bin/cua-driver"), popen=lambda cmd, **kw: launched.append(cmd))
    assert launched == [["/bin/cua-driver", "permissions", "grant"]]


@pytest.mark.skipif(not HAVE_BINARY, reason="cua-driver binary not found")
def test_the_real_daemon_is_started_from_python_with_no_terminal():
    cli = CuaCli()
    subprocess.run([cli.binary, "stop"], capture_output=True)
    assert cli.status()["state"] == "daemon_down"
    try:
        st = ensure_daemon(cli)
        print("after ensure_daemon:", st["state"])
        assert st["state"] in ("needs_permission", "ready")  # running; only the macOS approval (if not yet given) remains
        assert ensure_daemon(cli)["state"] == st["state"]  # idempotent
    finally:
        subprocess.run([cli.binary, "stop"], capture_output=True)
        subprocess.run(["pkill", "-f", "cua-driver serve"], capture_output=True)
