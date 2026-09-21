"""Shared types for the computer-use loop (browser and desktop use the same ones)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Element:
    id: str
    role: str  # button | link | textbox | checkbox | radio | select | tab | menuitem | ...
    label: str
    value: str = ""
    type: str = ""  # html input type ("password", "email", ...) when known
    checked: bool = False
    disabled: bool = False
    required: bool = False
    href: str = ""
    in_view: bool = True
    actions: tuple = ()  # desktop: AX actions the element exposes
    frame: tuple | None = None  # desktop: (x, y, w, h) in the screenshot's pixels, when known
    token: str = ""  # desktop: the driver's element_token (a bare index is refused by the driver)
    source: str = "ax"  # "ax" (accessibility tree) | "ocr" (words read from the screenshot)

    def short(self) -> str:
        v = f" = {self.value[:24]!r}" if self.value else ""
        return f"{self.role}: {self.label[:50]}{v}".strip()


@dataclass
class Observation:
    source: str  # "browser" | "desktop"
    title: str
    url: str | None
    elements: list[Element]
    text: str = ""
    alerts: list[str] = field(default_factory=list)
    dialogs: int = 0
    app: str = ""
    ms: float = 0.0
    screenshot: bytes | None = None
    content: str = ""  # everything the window is showing as text (static text, values, OCR words): a change here is a change, even if no control moved

    @property
    def host(self) -> str:
        from urllib.parse import urlparse

        return (urlparse(self.url).hostname or "") if self.url else ""

    @property
    def fingerprint(self) -> str:
        parts = [self.url or self.app, self.title, str(len(self.elements)), str(self.dialogs), self.content]
        parts += [f"{e.id}:{e.label[:20]}:{e.value[:20]}:{int(e.checked)}" for e in self.elements[:60]]
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]

    def by_id(self, element_id: str) -> Element | None:
        return next((e for e in self.elements if e.id == element_id), None)

    def summary(self, max_elements: int = 25) -> str:
        head = f"{self.title} ({self.url or self.app})" + (f" [dialog open]" if self.dialogs else "")
        lines = [head] + [f"- {e.id} {e.short()}" for e in self.elements[:max_elements]]
        if self.alerts:
            lines.append("alerts: " + "; ".join(self.alerts))
        return "\n".join(lines)


@dataclass
class StepPlan:
    """What the LLM wants, in words. `do` is one of the DO_KINDS."""
    do: str
    target: str = ""
    value: str | None = None

    def text(self) -> str:
        return f"{self.do} {self.target}" + (f" = {self.value!r}" if self.value is not None else "")


DO_KINDS = ("click", "type", "select", "check", "press_enter", "scroll", "navigate", "open_app", "wait", "done")


@dataclass
class Candidate:
    """One concrete action the application built for Laya to score. The model can only pick from these."""
    cid: str  # the element id, or "scroll" / "navigate"
    kind: str
    element: Element | None
    value: str | None
    emb: float = 0.0  # embedding similarity of the element to the step's target text

    def label(self) -> str:
        return self.element.short() if self.element else self.kind


@dataclass
class Action:
    kind: str
    id: str | None = None
    value: str | None = None
    key: str | None = None
    dy: int | None = None
    url: str | None = None
