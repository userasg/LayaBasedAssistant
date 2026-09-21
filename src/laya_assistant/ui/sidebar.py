"""The sidebar: what is running, how fast it is, and the settings you may want to change.

Kept short on purpose: two rows of badges and one line about what the assistant will ask before doing, then folded sections.
Expanders that hold widgets keep a fixed label, because a label that changes when you flip a switch would close the section
under your hand."""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from .. import approvals, config
from ..cua import authority
from ..cua.bridge import DEFAULT_PORT
from ..cua.desktop import ensure_daemon, grant_permissions
from ..session import AssistantSession
from ..transcript import clip

LOG_ROWS = 25
BADGE = {"laya": "🟢 Laya", "llm": "🔵 LLM", "code": "⚙️ code", "tool": "🛠 tool", "stt": "🎙 STT"}
_FIRST_TOKEN = re.compile(r"first token (\d+(?:\.\d+)?) ms")


def health_pills(session: AssistantSession, predictor) -> str:
    docker = session.handle.label.startswith("Docker")
    sandbox = "Docker sandbox" if docker else clip(session.handle.label, 26)
    return " ".join([
        f":green-badge[{sandbox}]" if docker else f":orange-badge[{sandbox}]",
        f":blue-badge[Laya · {predictor.device}]",
        f":violet-badge[{config.EXECUTOR_MODEL.replace('-instruct', '')}]",
    ])


def speed_pills(session: AssistantSession) -> str | None:
    log = session.log
    laya, llm, stt = log.stats("laya"), log.stats("llm"), log.stats("stt")
    first = [float(m.group(1)) for r in log.rows if r["step"] == "fast model" and (m := _FIRST_TOKEN.search(r["result"]))]
    pills = []
    if laya["count"]:
        pills.append(f":green-badge[Laya {laya['avg_ms']:.0f} ms · {laya['count']}]")
    if llm["count"]:
        pills.append(f":blue-badge[LLM {llm['avg_ms']:.0f} ms · {llm['count']}]")
    if first:
        pills.append(f":orange-badge[⚡ first word {first[-1]:.0f} ms]")
    if stt["count"]:
        pills.append(f":gray-badge[🎙 {stt['avg_ms']:.0f} ms]")
    return " ".join(pills) or None


def autonomy_summary() -> str:
    a = approvals.current
    asks = {"risky": "risky commands", "always_ask": "every command", "never_ask": None}[a.host_commands]
    items = [x for x in ("plans" if a.plan else None, asks, "irreversible steps", "new sites" if a.new_sites else None) if x]
    return "Asks you first: " + ", ".join(items)


def _apply(field: str, key: str):
    def callback():
        setattr(approvals.current, field, st.session_state[key])
    return callback


def render_autonomy() -> None:
    """Keyed widgets that write to the shared settings from a callback. (Unkeyed ones with `value=` taken from the settings get a new
    identity every time the setting changes, which drops the very click that changed it.)"""
    st.caption("Irreversible actions (send, pay, delete, post) and dangerous commands always stop for you. Everything else is up to you:")
    for key, field in (("ap_plan", "plan"), ("ap_host", "host_commands"), ("ap_sites", "new_sites"), ("ap_windows", "windows")):
        st.session_state.setdefault(key, getattr(approvals.current, field))
    labels = {"risky": "Only risky commands", "always_ask": "Every command", "never_ask": "None (dangerous ones are still refused)"}
    st.toggle("Hold plans for my approval", key="ap_plan", on_change=_apply("plan", "ap_plan"))
    st.radio("Ask before running a shell command on my Mac:", list(labels), format_func=labels.get, key="ap_host", on_change=_apply("host_commands", "ap_host"))
    st.toggle("Ask on the first visit to a new site (my own Chrome)", key="ap_sites", on_change=_apply("new_sites", "ap_sites"))
    st.toggle("Show app windows while working (hides this chat until done)", key="ap_windows", on_change=_apply("windows", "ap_windows"),
              help="On: the app comes to the front and this chat is hidden until the task ends: most reliable. Off: everything happens in the background and you stay here; "
                   "an app whose window is on another desktop cannot be driven then.")


def render_voice_settings() -> None:
    st.toggle("Send automatically when I stop speaking", value=False, key="auto_send",
              help="Off: what you say lands in a box you can edit, then you press Send.")
    st.caption("Speech is transcribed on this Mac (Whisper on the GPU). A lone word is never sent automatically.")


def render_computer_panel(session: AssistantSession, bridge, domain, cli) -> None:
    if st.button("⏹ STOP everything", type="primary", width="stretch", help="Halts a running computer-use goal before its next step and releases every Chrome tab."):
        session.stop_computer()
        if bridge is not None:
            bridge.release_all()
        st.toast("Stopped.")
    st.success("**Browser: ready.** I open my own browser window (it keeps its own logins). Nothing to install or switch on.")
    st.caption("**Scripts: ready.** I can open apps and pages, make Notes and run shell commands on your Mac. Safe ones run at once; others ask you.")
    stt = cli.status()
    if stt["state"] == "daemon_down":  # the app starts the driver itself: no terminal
        stt = ensure_daemon(cli)
    st.markdown("**Mac apps and your own Chrome** (Cua Driver)")
    if stt["available"]:
        st.success("Ready: I can click and type in your Mac apps.")
    elif stt["state"] == "needs_permission":
        st.warning("Only needed for clicking inside other apps. macOS has to allow it, once.")
        if st.button("Grant permissions", width="stretch"):
            grant_permissions(cli)
        st.caption("macOS opens its own dialogs. Switch on **CuaDriver** under Privacy & Security → Accessibility and Screen Recording, then press Refresh.")
    elif stt["state"] == "not_installed":
        st.error("Cua Driver is not installed: see docs/laya_assistant_computer_use.md")
    else:
        st.info(stt["detail"][:300])
    if st.button("Refresh", width="stretch"):
        st.rerun()

    approved = sorted(h for h in domain.approved if h not in ("localhost", "127.0.0.1", "::1"))
    if approved:
        st.caption("Sites approved this session: " + ", ".join(approved))
    cu = session.computer
    if cu is not None and cu.rec.rows:
        rows = [r for r in cu.rec.rows if r.get("agreement") is not None]
        if rows:
            st.caption(f"Laya agreed with the final choice on {sum(r['agreement'] for r in rows)}/{len(rows)} ambiguous steps")
        tiers = [r["tier"] for r in cu.rec.rows if "tier" in r]
        st.caption(f"{len(tiers)} steps recorded · {tiers.count('code')} by code, {tiers.count('cache')} remembered, "
                   f"{tiers.count('laya')} Laya, {tiers.count('llm')} LLM, {tiers.count('human')} you")
        st.caption(authority.summary_line(cu.rec.rows))

    with st.expander("Advanced (optional): Chrome extension for precise tab control"):
        ext = Path(__file__).parents[1] / "cua" / "extension"
        st.caption("Not needed. It gives per-tab control and DOM-level precision. Chrome → chrome://extensions → Developer mode → Load unpacked:")
        st.code(str(ext), language="text")
        if bridge is None:
            st.warning(f"Port {DEFAULT_PORT} is busy.")
        elif bridge.connected:
            claimed = [t for t in bridge.tabs if t.get("claimed")]
            st.success(f"Connected · {len(claimed)} tab(s) handed over")
            if st.button("Release all tabs", width="stretch"):
                bridge.release_all()
                st.rerun()
        else:
            st.caption("Then paste this token into the extension:")
            st.code(bridge.token, language="text")


def render(session: AssistantSession, predictor, bridge, domain, cli) -> None:
    with st.sidebar:
        if st.button("＋ New chat", key="new_chat", width="stretch"):
            session.close()
            for k in ("session", "last_voice", "dictation", "view"):
                st.session_state.pop(k, None)
            st.rerun()
        st.markdown(health_pills(session, predictor))
        speed = speed_pills(session)
        st.markdown(speed) if speed else st.caption("Speed shows up here as you use it.")
        st.caption(autonomy_summary())
        with st.expander("🎚 Autonomy"):
            render_autonomy()
        with st.expander("🎙 Voice"):
            render_voice_settings()
        with st.expander("🖥 Computer use"):
            render_computer_panel(session, bridge, domain, cli)
        with st.expander(f"📈 Activity (last {LOG_ROWS})"):
            rows = [{"system": BADGE.get(r["system"], r["system"]), "step": r["step"], "result": r["result"][:48], "ms": r["ms"]}
                    for r in session.log.rows[-LOG_ROWS:]][::-1]
            st.dataframe(rows, hide_index=True, width="stretch", height=300, column_config={"ms": st.column_config.NumberColumn("ms", format="%.0f")})
        with st.expander("⚙️ Models & paths"):
            st.caption(f"🟢 Laya `{config.LAYA_CHECKPOINT.split('/')[-1]}` on `{predictor.device}`")
            st.caption(f"🧠 `{config.EXECUTOR_MODEL}` · ⚡ `{config.FAST_MODEL}` · 👁 `{config.VISION_MODEL}`")
            st.caption(f"🎙 `{config.STT_MODEL.split('/')[-1]}` · 📦 {session.handle.label}")
            st.caption(f"Memory `{config.HOME / 'memories'}`")
            st.caption(f"Log `{session.log.path}`")
