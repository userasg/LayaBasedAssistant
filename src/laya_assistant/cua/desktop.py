"""Desktop control through Cua Driver (open source, MIT): any Mac app, in the background, by accessibility element index.

Cua Driver runs as a small daemon and is called with `cua-driver call <tool> '<json>'`. It needs two macOS permissions
that only you can grant (System Settings > Privacy & Security: Accessibility and Screen Recording, for CuaDriver.app).
Until then `status()` says exactly what is missing and the rest of the app carries on.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from .. import config
from .apps import Presence, bring_up
from .types import Action, Element, Observation

ROLE_MAP = {
    "AXButton": "button", "AXMenuButton": "button", "AXMenuItem": "menuitem", "AXMenuBarItem": "menuitem",
    "AXTextField": "textbox", "AXTextArea": "textbox", "AXSearchField": "textbox", "AXSecureTextField": "textbox", "AXComboBox": "combobox",
    "AXCheckBox": "checkbox", "AXRadioButton": "radio", "AXSwitch": "switch", "AXPopUpButton": "select", "AXLink": "link",
    "AXTab": "tab", "AXRow": "option", "AXCell": "option", "AXDisclosureTriangle": "button", "AXSlider": "textbox",
}
INSTALL_HELP = (
    "Cua Driver is not running. One-time setup: (1) cua-driver permissions grant  (opens the macOS permission dialogs for CuaDriver: "
    "Accessibility and Screen Recording), then (2) cua-driver serve --socket ~/Library/Caches/cua-driver/cua-driver.sock  (keep it running). "
    "The binary is at {bin}."
)


PERMISSION_HELP = ("One-time macOS approval needed: press 'Grant permissions', then switch on CuaDriver under System Settings > "
                   "Privacy & Security > Accessibility and Screen Recording. macOS does not allow this to be done for you.")
SOCKET = Path.home() / "Library/Caches/cua-driver/cua-driver.sock"
BROWSERS = ("Google Chrome", "Arc", "Brave Browser", "Microsoft Edge", "Safari", "Firefox")
_ADDRESS = re.compile(r"address|url|omnibox|location|search or enter|search or type", re.I)


def find_binary() -> str | None:
    cands = [shutil.which("cua-driver"), str(Path.home() / ".local/bin/cua-driver"), "/Applications/CuaDriver.app/Contents/MacOS/cua-driver"]
    cands += sorted(glob.glob(str(config.HOME / "cua-driver" / "bin" / "*" / "cua-driver")), reverse=True)
    return next((c for c in cands if c and os.path.exists(c) and os.access(c, os.X_OK)), None)


class CuaCli:
    """Thin wrapper over `cua-driver call`. Raises RuntimeError with an actionable message when the daemon is down."""

    def __init__(self, binary: str | None = None, runner=subprocess.run, timeout: float = 25.0):
        self.binary = binary or find_binary()
        self.runner, self.timeout = runner, timeout

    @property
    def installed(self) -> bool:
        return bool(self.binary)

    def call(self, tool: str, args: dict | None = None, timeout: float | None = None) -> dict:
        if not self.binary:
            raise RuntimeError("Cua Driver is not installed (see docs/laya_assistant_computer_use.md)")
        p = self.runner([self.binary, "call", tool, json.dumps(args or {})], capture_output=True, text=True, timeout=timeout or self.timeout)
        out = (p.stdout or "").strip() or (p.stderr or "").strip()  # the driver reports states such as permissions_pending on stderr
        if "daemon is not running" in out or "daemon is not running" in (p.stderr or ""):
            raise RuntimeError(INSTALL_HELP.format(bin=self.binary))
        if p.returncode != 0 and not out:
            raise RuntimeError(f"cua-driver {tool} failed")
        if "permissions_pending" in out:
            return {"text": out}
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return {"text": out}
        return data.get("structuredContent", data) if isinstance(data, dict) else {"result": data}

    def status(self) -> dict:
        """{'available', 'installed', 'state', 'detail'}; state: ready | needs_permission | daemon_down | not_installed | error. Never raises."""
        if not self.installed:
            return {"available": False, "installed": False, "state": "not_installed", "detail": "Cua Driver is not installed"}
        try:
            perm = self.call("check_permissions", {})
        except RuntimeError as e:
            msg = str(e)
            if "permissions_pending" in msg:
                return {"available": False, "installed": True, "state": "needs_permission", "detail": PERMISSION_HELP}
            state = "daemon_down" if "not running" in msg else "error"
            return {"available": False, "installed": True, "state": state, "detail": msg}
        except Exception as e:  # timeouts, unexpected output
            return {"available": False, "installed": True, "state": "error", "detail": f"{type(e).__name__}: {e}"}
        if isinstance(perm, dict) and "accessibility" in perm and "screen_recording" in perm:  # the real payload: read the booleans
            ok = all(v is True or str(v).lower() in ("granted", "true", "authorized") for v in (perm["accessibility"], perm["screen_recording"]))
            return {"available": ok, "installed": True, "state": "ready" if ok else "needs_permission", "detail": "ready" if ok else PERMISSION_HELP}
        text = json.dumps(perm).lower()
        if "permissions_pending" in text or "pending" in text and "permission" in text:
            return {"available": False, "installed": True, "state": "needs_permission", "detail": PERMISSION_HELP}
        ok = ("accessibility" in text and "screen" in text) and not any(w in text for w in ('"denied"', "not granted", "false"))
        return {"available": ok, "installed": True, "state": "ready" if ok else "needs_permission",
                "detail": "ready" if ok else PERMISSION_HELP}


def _role(raw_role: str) -> str:
    return ROLE_MAP.get(raw_role, raw_role.removeprefix("AX").lower())


def _frame(e: dict, state: dict) -> tuple | None:
    """An element's frame in the SCREENSHOT's pixels. The driver reports frames in screen points; the screenshot is the window only, and
    may be scaled (measured: subtracting the window origin and scaling by shot width / window width put OCR words inside the right rows
    in Music, Notes, Finder and TextEdit)."""
    f, wb, sw = e.get("frame"), state.get("window_bounds"), state.get("screenshot_width")
    if not (f and wb and sw and wb.get("width")):
        return None
    k = sw / wb["width"]
    return ((f["x"] - wb["x"]) * k, (f["y"] - wb["y"]) * k, f["w"] * k, f["h"] * k)


def _refusal(res) -> str | None:
    """The driver answers a request it will not perform with a refusal object, not an error. That is a failure, not a success."""
    if isinstance(res, dict) and (res.get("status") == "refused" or res.get("refusal")):
        r = res.get("refusal") or {}
        return f"the driver refused: {r.get('code', 'refused')}: {str(r.get('message', ''))[:160]}"
    return None


def _content(state: dict) -> str:
    """All the text the window's tree carries (labels and values, including static text that is not a control): the display of a calculator,
    the body of a note, a status line."""
    hidden = _under_menu_bar(state)  # the menu bar's labels never change and would bury the window's own text
    return " ".join(str(e.get(k)) for e in state.get("elements", []) if e.get("element_index") not in hidden for k in ("label", "value") if e.get(k))[:4000]


def _under_menu_bar(state: dict) -> set[int]:
    """Indexes of everything inside the menu bar. Its items are always in the tree but only exist on screen while a menu is open, so
    clicking one does nothing (measured: 'Applications', 'Songs', 'View' picked from a closed menu). Menus that belong to a control in the
    window (a popup button's list) are not under the menu bar and stay."""
    by_parent: dict = {}
    for e in state.get("elements", []):
        by_parent.setdefault(e.get("parent_index"), []).append(e.get("element_index"))
    gone, todo = set(), [e.get("element_index") for e in state.get("elements", []) if e.get("role") == "AXMenuBar"]
    while todo:
        i = todo.pop()
        if i in gone:
            continue
        gone.add(i)
        todo.extend(by_parent.get(i, []))
    return gone


def elements_from_state(state: dict) -> list[Element]:
    out = []
    hidden = _under_menu_bar(state)
    for e in state.get("elements", []):
        if e.get("element_index") in hidden:
            continue
        role = _role(str(e.get("role", "")))
        label = str(e.get("label") or e.get("title") or e.get("description") or "").strip()
        actions = tuple(e.get("actions") or ())
        if role in ("statictext", "group", "unknown", "image", "scrollarea", "splitgroup") and not actions and not label:
            continue
        idx = e.get("element_index")
        if idx is None:
            continue
        out.append(Element(id=f"e{idx}", role=role, label=label[:90], value=str(e.get("value") or "")[:120],
                           type="password" if e.get("role") == "AXSecureTextField" else "", disabled=e.get("enabled") is False, actions=actions,
                           checked=role in ("checkbox", "radio", "switch") and str(e.get("value")).strip().lower() in ("1", "true", "on"),
                           frame=_frame(e, state), token=str(e.get("element_token") or "")))
    return out


_ICON_PREFIX = re.compile(r"^(?:[^\w\s]|\w{1,2}\d?|[•·])(?:\s+(?=\S))")


def clean_ocr_label(text: str) -> str:
    """OCR reads a row's icon as a stray character ("L Applications", "8g New", "• Recents", "Q Search"): drop leading glyph-like tokens
    when a real word follows."""
    prev = None
    while prev != text:
        prev = text
        rest = _ICON_PREFIX.sub("", text, count=1)
        if len(rest.split()) >= 1 and len(rest) >= 3 and rest != text:
            text = rest
    return text.strip()


NEEDS_TEXT = {"option", "group", "button", "menuitem", "tab", "unknown"}  # roles that often have no accessible name but a visible one


def enrich_with_screenshot(els: list[Element], boxes) -> list[Element]:
    """Give unnamed elements the words drawn inside them, drop the ones that stay nameless and useless, and add the words that no element
    covers as clickable text targets (web views, custom lists). `boxes` come from ocr.read_text."""
    from .ocr import label_from_boxes

    covered: set[int] = set()
    out: list[Element] = []
    for e in els:
        if not e.label and e.frame and e.role in NEEDS_TEXT:
            e.label = clean_ocr_label(label_from_boxes(e.frame, boxes))[:90]
            e.source = "ocr" if e.label else e.source
        if e.frame:
            x, y, w, h = e.frame
            for i, b in enumerate(boxes):
                if x - 2 <= b.cx <= x + w + 2 and y - 2 <= b.cy <= y + h + 2 and e.actions:
                    covered.add(i)
        if e.label or e.role in ("textbox", "checkbox", "radio", "switch", "select", "combobox"):
            out.append(e)
    seen = {(e.role, e.label.lower()) for e in out}
    for i, b in enumerate(boxes):
        if i not in covered and len(b.text) >= 2 and ("text", b.text.lower()) not in seen:
            seen.add(("text", b.text.lower()))
            out.append(Element(id=f"t{i}", role="text", label=b.text[:90], frame=(b.x, b.y, b.w, b.h), source="ocr", actions=("click",)))
    return out


class DesktopDriver:
    name = "desktop"

    def __init__(self, app: str = "", cli: CuaCli | None = None, browser: bool = False):
        self.app, self.cli, self.browser = app, cli or CuaCli(), browser
        self._addr: Element | None = None
        self.pid: int | None = None
        self.window_id: int | None = None

    # -- app / window ---------------------------------------------------------------------------------------

    def _daemon(self) -> None:
        """Start the driver's background service if it is not running (once per driver): no terminal step for the person."""
        if not getattr(self, "_daemon_checked", False):
            self._daemon_checked = True
            if getattr(self.cli, "installed", False) and self.cli.status().get("state") == "daemon_down":
                ensure_daemon(self.cli)

    def launch(self, app: str | None = None) -> Presence:
        """Open the app the way a person would and wait for a window the driver can read (the driver's own launch_app never shows it)."""
        self._daemon()
        name = app or self.app
        try:
            from .apps import AppCatalog

            found = AppCatalog().resolve(name)  # the planner says "Apple Music"; the app is "Music"
            name = found.name if found else name
        except Exception:
            pass
        self.app = name
        from .. import approvals

        return bring_up(self.app, self.cli, foreground=approvals.current.windows)

    def _pick_browser(self) -> str:
        wins = self.cli.call("list_windows", {"on_screen_only": True}).get("windows", [])
        open_apps = {str(w.get("app_name", "")) for w in wins}
        return next((b for b in BROWSERS if b in open_apps), "Google Chrome")

    def _window(self, wait: float = 6.0) -> tuple[int, int, str]:
        if self.browser and not self.app:
            self.app = self._pick_browser()
        self._daemon()
        wins = self.cli.call("list_windows", {"on_screen_only": True}).get("windows", [])
        mine = [w for w in wins if str(w.get("app_name", "")).lower() == self.app.lower()] if self.app else wins
        # the driver also lists tiny surfaces an app owns (menu-bar strips, helper panels): those hold no controls
        mine = [w for w in mine if ((w.get("bounds") or {}).get("width", 1000) > 100 and (w.get("bounds") or {}).get("height", 1000) > 60)]
        if self.app and not self.browser:
            for w in sorted(mine, key=lambda w: (bool(w.get("title")), ((w.get("bounds") or {}).get("width", 0) * (w.get("bounds") or {}).get("height", 0)), w.get("z_index") or 0), reverse=True):
                st = self.cli.call("get_window_state", {"pid": w["pid"], "window_id": w["window_id"], "include_screenshot": False})
                if st.get("elements"):  # an app may own several windows; only some resolve
                    return int(w["pid"]), int(w["window_id"]), str(w.get("title", ""))
        elif mine:
            w = max(mine, key=lambda w: w.get("z_index") or 0)
            return int(w["pid"]), int(w["window_id"]), str(w.get("title", ""))
        p = self.launch()  # nothing readable is on screen (closed, another Space, still settling): bring it up, and say why if it will not come
        if p.ok:
            return p.pid, p.window_id, ""
        raise RuntimeError(p.reason)

    # -- Driver interface -----------------------------------------------------------------------------------

    def observe(self) -> Observation:
        t0 = time.perf_counter()
        self.pid, self.window_id, title = self._window()
        # ONE snapshot per observation: the driver's element tokens belong to the latest snapshot of a window, so a second call would make
        # the first one's tokens stale. A desktop window is fetched with its screenshot (frames for position clicks, words for OCR).
        st = self.cli.call("get_window_state", {"pid": self.pid, "window_id": self.window_id, "include_screenshot": not self.browser})
        els = elements_from_state(st)
        content = _content(st)
        nameless = [e for e in els if not e.label and e.role in NEEDS_TEXT and e.actions]
        self._px = {}
        if not self.browser and st.get("screenshot_png_b64"):
            import base64

            from .ocr import read_text

            small = st["screenshot_width"] * st["screenshot_height"] < 1_000_000  # e.g. a calculator: its display is pixels, not tree text
            if len(nameless) >= 3 or not els or small:  # unnamed controls, nothing, or a small window: read the words off the screenshot
                boxes = read_text(base64.b64decode(st["screenshot_png_b64"]), st["screenshot_width"], st["screenshot_height"])
                els = enrich_with_screenshot(els, boxes)
                content += " " + " ".join(b.text for b in boxes)
            self._px = {e.id: (e.frame[0] + e.frame[2] / 2, e.frame[1] + e.frame[3] / 2) for e in els if e.frame}
        self._sid = st.get("snapshot_id")
        self._pressable = {e.id: ("AXPress" in e.actions or "click" in e.actions) for e in els}
        self._tokens = {e.id: e.token for e in els if e.token}
        url, source = None, "desktop"
        if self.browser:
            # a browser window: the address bar gives the URL, so the domain rules apply exactly as they do with the extension
            self._addr = address_bar(els)
            url = normalise_url(self._addr.value) if self._addr else None
            source = "browser"
            if len([e for e in els if e.role in ("link", "button", "textbox", "checkbox")]) < 4 and not url:
                raise RuntimeError(f"{self.app} exposes almost no accessibility information yet. Open chrome://accessibility, tick 'Native accessibility API support', and retry")
        return Observation(source=source, title=st.get("window_title") or title, url=url, app=self.app or str(st.get("app_name", "")),
                           elements=els, text=str(st.get("tree_markdown", ""))[:500], content=content, ms=(time.perf_counter() - t0) * 1000)

    def click_by_position(self, element_id: str, foreground: bool = False) -> dict:
        """The second and third ways to click: a real mouse click at the element's centre, then the same with the window briefly fronted
        (the driver's own ladder: AX press -> background pixel -> foreground). Needs the frames from the last observation."""
        if element_id not in getattr(self, "_px", {}):
            return {"ok": False, "error": "no position known for that element"}
        x, y = self._px[element_id]
        try:
            res = self.cli.call("click", {"pid": self.pid, "window_id": self.window_id, "x": int(x), "y": int(y),
                                          **({"delivery_mode": "foreground"} if foreground else {})})
            refused = _refusal(res)
            return {"ok": False, "error": refused} if refused else {"ok": True}
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}

    def act(self, a: Action) -> dict:
        try:
            base = {"pid": self.pid, "window_id": self.window_id}
            idx = int(a.id[1:]) if a.id and a.id[1:].isdigit() else None
            tgt = self._target(a.id, idx)
            if a.kind == "navigate":
                return self._navigate(a.url or "")
            if a.kind == "open_app":
                self._launched = True
                p = self.launch(a.value)
                return {"ok": True} if p.ok else {"ok": False, "error": p.reason}
            press_ok = self._pressable.get(a.id, True) if a.id else True
            if a.kind in ("click", "check") and a.id and a.id in getattr(self, "_px", {}) and (a.id.startswith("t") or not press_ok):
                x, y = self._px[a.id]  # words read off the screenshot, or a row/cell with no press action: click where it is
                res = self.cli.call("click", {**base, "x": int(x), "y": int(y)})
            elif a.kind in ("click", "check"):
                res = self.cli.call("click", {**base, **tgt})
            elif a.kind == "type":
                res = self.cli.call("set_value", {**base, **tgt, "value": a.value or ""})
            elif a.kind == "select":
                res = self.cli.call("set_value", {**base, **tgt, "value": a.value or ""})
            elif a.kind == "key":
                res = self.cli.call("press_key", {"pid": self.pid, "key": (a.key or "return").lower(), **(tgt if idx is not None else {})})
            elif a.kind == "scroll":
                self.cli.call("scroll", {"pid": self.pid, "direction": "up" if (a.dy or 0) < 0 else "down", "amount": 5})
            elif a.kind == "wait":
                time.sleep((a.dy or 500) / 1000)
            else:
                return {"ok": False, "error": f"unsupported on the desktop: {a.kind}"}
            refused = _refusal(locals().get("res"))
            return {"ok": False, "error": refused} if refused else {"ok": True}
        except RuntimeError as e:
            return {"ok": False, "error": str(e)[:200]}

    def _target(self, element_id: str | None, idx: int | None) -> dict:
        """How to address an element: the driver refuses a bare element_index. Use the element's token, or the index with its snapshot id."""
        tok = getattr(self, "_tokens", {}).get(element_id or "")
        if tok:
            return {"element_token": tok}
        sid = getattr(self, "_sid", None)
        return {"element_index": idx, **({"snapshot_id": sid} if sid else {})} if idx is not None else {}

    def _navigate(self, url: str) -> dict:
        if not self.browser:
            return {"ok": False, "error": "navigate needs a browser window"}
        try:
            if self._addr is not None:
                idx = int(self._addr.id[1:])
                self.cli.call("set_value", {"pid": self.pid, "window_id": self.window_id, "element_index": idx, "value": url})
                self.cli.call("press_key", {"pid": self.pid, "key": "return", "element_index": idx})
            else:
                self.cli.call("hotkey", {"pid": self.pid, "window_id": self.window_id, "keys": ["cmd", "l"]})
                self.cli.call("type_text", {"pid": self.pid, "text": url})
                self.cli.call("press_key", {"pid": self.pid, "key": "return"})
            time.sleep(0.8)  # let the page start loading; the next observation is the check
            return {"ok": True}
        except RuntimeError as e:
            return {"ok": False, "error": str(e)[:200]}

    def close(self) -> None:
        pass


def ensure_daemon(cli: CuaCli, wait: float = 8.0, popen=subprocess.Popen) -> dict:
    """Start the Cua Driver daemon if it is not running (no terminal needed), and return its status. Safe to call repeatedly."""
    st = cli.status()
    if st["state"] != "daemon_down" or not cli.installed:
        return st
    SOCKET.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([cli.binary, "telemetry", "disable"], capture_output=True, timeout=10)  # off by default in this app
    config.HOME.mkdir(parents=True, exist_ok=True)
    log = open(config.HOME / "cua-driver.log", "ab")
    popen([cli.binary, "serve", "--socket", str(SOCKET)], stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    deadline = time.time() + wait
    while time.time() < deadline:
        st = cli.status()
        if st["state"] != "daemon_down":
            return st
        time.sleep(0.3)
    return st


def grant_permissions(cli: CuaCli, popen=subprocess.Popen) -> None:
    """Opens macOS's own permission dialogs for Cua Driver (the user has to approve them: the OS offers no way around that)."""
    if cli.installed:
        popen([cli.binary, "permissions", "grant"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def normalise_url(value: str) -> str | None:
    v = (value or "").strip()
    if not v or " " in v:
        return None
    if re.match(r"^[a-z][a-z0-9+.-]*://", v):
        return v
    host = v.split("/")[0]
    return "https://" + v if ("." in host or host.startswith("localhost")) else None


def address_bar(elements: list[Element]) -> Element | None:
    return next((e for e in elements if e.role in ("textbox", "combobox") and _ADDRESS.search(e.label)), None)
