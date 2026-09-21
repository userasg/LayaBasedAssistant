"""The local end of the Laya Bridge extension: a WebSocket server the extension connects to, and a Driver that
sends observe/act requests through it to a tab the user has claimed.

Security model: the socket listens on localhost only and every connection must present a random token (created once,
kept in ~/.laya_assistant/bridge_token, pasted into the extension once). The extension itself refuses any request for a
tab that was not claimed by the user, and the user can release everything with one click.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
import secrets
import threading
import time

import websockets

from .. import config
from .driver import observation_from_js
from .types import Action, Observation

DEFAULT_PORT = 8766
# Defence in depth on top of the token: from a browser, only the Laya Bridge extension itself may connect (a web page cannot).
BRIDGE_ORIGINS = [None, re.compile(r"^chrome-extension://[a-p]{32}$")]


def load_token(path=None) -> str:
    p = path or (config.HOME / "bridge_token")
    if p.exists():
        return p.read_text().strip()
    p.parent.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(24)
    p.write_text(tok)
    p.chmod(0o600)
    return tok


class BridgeServer:
    def __init__(self, port: int = DEFAULT_PORT, token: str | None = None, host: str = "127.0.0.1"):
        self.port, self.host = port, host
        self.token = token or load_token()
        self.tabs: list[dict] = []
        self.error: Exception | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._pending: dict[int, concurrent.futures.Future] = {}
        self._n = 0
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._server = None
        self._thread: threading.Thread | None = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    # -- lifecycle --------------------------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), name="bridge-server", daemon=True)
        self._thread.start()
        self._ready.wait(10)
        if self.error:
            raise self.error

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            self._server = await websockets.serve(self._handle, self.host, self.port, max_size=16 * 1024 * 1024, origins=BRIDGE_ORIGINS)
        except Exception as e:
            self.error = e
            self._ready.set()
            return
        self._ready.set()
        await self._stop.wait()
        self._server.close()
        await self._server.wait_closed()

    def stop(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)

    async def _handle(self, ws) -> None:
        try:
            hello = json.loads(await asyncio.wait_for(ws.recv(), 5))
        except Exception:
            return
        if hello.get("type") != "hello" or not secrets.compare_digest(str(hello.get("token", "")), self.token):
            await ws.close(4401, "bad token")
            return
        self._ws = ws
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "tabs":
                    self.tabs = msg.get("tabs", [])
                elif "id" in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
        finally:
            if self._ws is ws:
                self._ws, self.tabs = None, []
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(ConnectionError("the browser extension disconnected"))
            self._pending.clear()

    # -- requests ----------------------------------------------------------------------------------------

    def request(self, method: str, tab_id: int | None = None, params: dict | None = None, timeout: float = 20.0) -> dict:
        if not self._ws or not self._loop:
            raise RuntimeError("Chrome is not connected: install the Laya Bridge extension and paste the token (see the Computer use panel)")
        self._n += 1
        rid = self._n
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._pending[rid] = fut
        payload = json.dumps({"id": rid, "method": method, "tabId": tab_id, "params": params or {}})
        asyncio.run_coroutine_threadsafe(self._ws.send(payload), self._loop)
        try:
            msg = fut.result(timeout)
        except concurrent.futures.TimeoutError:
            self._pending.pop(rid, None)
            raise RuntimeError(f"the browser did not answer {method} within {timeout:.0f}s")
        if msg.get("error"):
            raise RuntimeError(msg["error"] if msg["error"] != "tab_not_claimed" else "that tab is not one you handed to Laya")
        return msg

    def release_all(self) -> None:
        try:
            self.request("release_all")
        except RuntimeError:
            pass

    def wait_connected(self, timeout: float = 15.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.connected:
                return True
            time.sleep(0.1)
        return False


class BridgeDriver:
    name = "chrome"

    def __init__(self, server: BridgeServer, tab_id: int | None = None):
        self.server = server
        self._tab = tab_id

    @property
    def tab_id(self) -> int:
        if self._tab is not None:
            return self._tab
        claimed = [t for t in self.server.tabs if t.get("claimed")]
        if not claimed:
            raise RuntimeError("no Chrome tab has been handed to Laya: open the tab you want me to use, click the Laya Bridge extension icon, and press 'Let Laya control this tab'")
        return claimed[0]["id"]

    def observe(self) -> Observation:
        t0 = time.perf_counter()
        raw = self.server.request("observe", self.tab_id)["result"]
        return observation_from_js(raw, (time.perf_counter() - t0) * 1000)

    def act(self, action: Action) -> dict:
        if action.kind == "navigate":
            return self.server.request("navigate", self.tab_id, {"url": action.url})["result"]
        if action.kind == "wait":
            time.sleep((action.dy or 500) / 1000)
            return {"ok": True}
        payload = {"kind": action.kind, "id": action.id, "value": action.value, "key": action.key, "dy": action.dy}
        res = self.server.request("act", self.tab_id, payload)["result"]
        time.sleep(0.06)  # let handlers and client-side rendering settle
        return res

    def screenshot(self) -> bytes:
        import base64

        data = self.server.request("screenshot", self.tab_id)["result"]["dataUrl"]
        return base64.b64decode(data.split(",", 1)[1])

    def close(self) -> None:
        pass
