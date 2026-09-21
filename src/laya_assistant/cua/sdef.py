"""What a scriptable app can be told to do, read from the scripting dictionary (.sdef) the app ships. Generic: any scriptable app.

A model that writes AppleScript blind guesses class and property names and gets types wrong (`-1700`, `-1728`). Given the app's real
commands, classes, properties and elements it writes a correct script the first time. The digest is small (a 16k context is shared with
everything else), read straight from the bundle (no Xcode `sdef` tool needed), and cached per app for the life of the process.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

BUDGET = 1800  # characters
# Every scriptable app has these (open, close, count, delete, make...): the model knows them, they only cost space.
_SKIP_SUITES = {"standard suite", "text suite", "type definitions", "internet suite", "text", "standard"}
_cache: dict[str, str] = {}


def _sdef_file(app_path: str) -> Path | None:
    files = sorted(Path(app_path, "Contents/Resources").glob("*.sdef"))
    return files[0] if files else None


def _command(c: ET.Element) -> str:
    direct = c.find("direct-parameter")
    parts = [c.get("name", "?")]
    if direct is not None:
        parts.append(f"<{direct.get('type', 'object')}>")
    for p in c.findall("parameter"):
        parts.append(f"[{p.get('name')} <{p.get('type', '?')}>]" if p.get("optional") == "yes" else f"{p.get('name')} <{p.get('type', '?')}>")
    return " ".join(parts)


def _class(c: ET.Element) -> str:
    props = [p.get("name") for p in c.findall("property")][:10]
    elems = [e.get("type") for e in c.findall("element")][:6]
    out = c.get("name", "?")
    if c.get("inherits"):
        out += f" (is a {c.get('inherits')})"
    if props:
        out += " props: " + ", ".join(props)
    if elems:
        out += " | contains: " + ", ".join(elems)
    return out


def digest_xml(name: str, xml_text: str, budget: int = BUDGET) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""
    # A class that many properties, elements and parameters refer to (track, note, tab, message) is what scripts are about; one nothing
    # else mentions (an AirPlay device, an encoder) is not. Rank by that, so the budget goes to the useful ones first.
    refs: dict[str, int] = {}
    for el in root.iter():
        t = el.get("type")
        if t:
            refs[t] = refs.get(t, 0) + 1
    cmds, classes = [], []
    for suite in root.iter("suite"):
        if (suite.get("name") or "").lower() in _SKIP_SUITES:
            continue
        cmds += [_command(c) for c in suite.findall("command")]
        classes += [(refs.get(c.get("name"), 0), _class(c)) for c in suite.findall("class") if c.get("name") and c.get("name") != "application"]
    if not cmds and not classes:
        return ""
    classes = [c for _, c in sorted(classes, key=lambda x: -x[0])]
    lines = [f'{name} scripting dictionary (write `tell application "{name}"`; names below are exact):']
    if cmds:
        lines.append("Commands: " + "; ".join(cmds))
    text = "\n".join(lines)
    for c in classes:  # most-referenced first, each whole, until the budget is spent
        if len(text) + len(c) + 3 > budget:
            break
        text += "\n- " + c
    return text[:budget]


def digest(name: str, app_path: str | None) -> str:
    """The dictionary digest for an installed app, or "" when it has none. Never raises: this only helps the model."""
    if not app_path:
        return ""
    if app_path in _cache:
        return _cache[app_path]
    text = ""
    try:
        f = _sdef_file(app_path)
        if f:
            text = digest_xml(name, f.read_text(errors="replace"))
    except OSError:
        pass
    _cache[app_path] = text
    return text
