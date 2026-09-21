"""Puts the pieces together for the app: one bridge, one domain policy, the driver factory, the LLM tie-break."""
from __future__ import annotations

import re

from ..engine import llm_lock, make_llm
from .. import config
from .bridge import BridgeDriver, BridgeServer
from .desktop import CuaCli, DesktopDriver, ensure_daemon
from .manager import ComputerUse
from .own_browser import BrowserWorker, OwnBrowserDriver
from .planner import make_planner
from .policy import DomainPolicy
from .recorder import StepRecorder


class LazyBrowser:
    """A browser handle for early (while-speaking) commands: truthy only when Chrome is connected AND a tab has been handed over."""

    def __init__(self, bridge: BridgeServer):
        self.bridge = bridge

    def __bool__(self) -> bool:
        return self.bridge.connected and any(t.get("claimed") for t in self.bridge.tabs)

    def act(self, action):
        return BridgeDriver(self.bridge).act(action)


def make_tiebreak(llm=None):
    """When Laya and the embeddings cannot separate the top candidates, ask the LLM which element the step means."""
    model = llm or make_llm(config.FAST_MODEL, temperature=0.0)

    def tiebreak(step, obs, cands):
        options = "\n".join(f"{i + 1}. {c.label()}" for i, c in enumerate(cands))
        prompt = f"Page: {obs.title}\nStep: {step.text()}\nWhich element does the step mean?\n{options}\nAnswer with just the number, or 0 if none fits."
        with llm_lock():
            text = str(model.invoke(prompt).content)
        m = re.search(r"\d", text)
        k = int(m.group()) if m else 0
        return cands[k - 1] if 1 <= k <= len(cands) else None

    return tiebreak


def make_computer_use(predictor, ranker, bridge: BridgeServer | None, cli: CuaCli | None = None, domain: DomainPolicy | None = None,
                      recorder_path=None, worker: BrowserWorker | None = None) -> ComputerUse:
    cli = cli or CuaCli()
    domain = domain or DomainPolicy()
    worker = worker or BrowserWorker()
    own_domain = DomainPolicy(ask_unknown=False)

    def driver_factory(target: str, app: str = ""):
        if target == "browser":
            return OwnBrowserDriver(worker)  # a window the app opens itself: nothing to install, enable or grant
        if target == "my_chrome":
            if bridge is not None and bridge.connected and any(t.get("claimed") for t in bridge.tabs):
                return BridgeDriver(bridge)  # the optional extension: a tab you handed over
            st = ensure_daemon(cli)
            if not st["available"]:
                raise RuntimeError("your own Chrome needs either the Laya Bridge extension or the Cua Driver permission (see the Computer use panel); "
                                   "target='browser' works without either")
            return DesktopDriver("", cli, browser=True)
        st = ensure_daemon(cli)
        if not st["available"]:
            raise RuntimeError(st["detail"])
        return DesktopDriver(app, cli)

    cu = ComputerUse(predictor, ranker, lambda t: make_planner(surface="a Mac app" if t == "desktop" else "a web browser"), driver_factory,
                     domain, StepRecorder(recorder_path), llm_tiebreak=make_tiebreak(), own_domain=own_domain)
    cu.worker = worker
    return cu
