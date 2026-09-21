"""The driver interface. A driver observes a surface and performs one Action on it."""
from __future__ import annotations

from typing import Protocol

from .types import Action, Observation


class Driver(Protocol):
    name: str

    def observe(self) -> Observation: ...

    def act(self, action: Action) -> dict:
        """Perform the action. Returns {"ok": bool, "error": str|None, ...}."""
        ...

    def close(self) -> None: ...


def elements_from_js(raw: list[dict]) -> list:
    from .types import Element

    return [Element(id=e["id"], role=e.get("role", ""), label=e.get("label", ""), value=e.get("value", ""), type=e.get("type", ""),
                    checked=bool(e.get("checked")), disabled=bool(e.get("disabled")), required=bool(e.get("required")),
                    href=e.get("href", ""), in_view=bool(e.get("inViewport", True))) for e in raw]


def observation_from_js(raw: dict, ms: float = 0.0) -> Observation:
    return Observation(source="browser", title=raw.get("title", ""), url=raw.get("url"), elements=elements_from_js(raw.get("elements", [])),
                       text=raw.get("text", ""), alerts=raw.get("alerts", []), dialogs=int(raw.get("dialogs", 0)), ms=ms)
