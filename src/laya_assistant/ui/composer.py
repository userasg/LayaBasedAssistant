"""The composer: the box you type in, the mic, attachments, and the review card for what you said.

Everything but the chat box itself goes in Streamlit's bottom block, so it stays pinned above the box however long the chat gets
(before, the mic sat at the end of the page and scrolled away)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from .. import config
from ..session import AssistantSession
from ..voice_server import VoiceServer

_live_voice = components.declare_component("live_voice", path=str(Path(__file__).parents[1] / "voice_component"))


@st.cache_resource(show_spinner=False)
def get_voice_server(_transcriber, _predictor, _make_quick):
    """Local WebSocket the mic component streams audio to. Started once per process."""
    server = VoiceServer(_transcriber, _predictor, port=config.VOICE_PORT, quick=_make_quick())
    server.start()
    return server


@dataclass
class Submission:
    text: str
    uploads: list[tuple[str, bytes]] = field(default_factory=list)


def route_speech(heard: str, done: list, locked: bool, auto_send: bool) -> str:
    """What to do with transcribed speech: "send" it now, park it in the "review" card, or do "nothing".
    Speech is only ever sent unreviewed when auto-send is on AND nothing is running or waiting AND it is not a lone word (a lone word is
    more likely noise than a request). While a question waits for your OK it is always reviewed: Whisper hears "yes" and "no" in room
    noise, and noise must not be able to approve a command."""
    if not heard:
        return "nothing"
    if auto_send and not locked and (len(heard.split()) >= 2 or done):
        return "send"
    return "review"


def _heard(session: AssistantSession, voice: dict, locked: bool, auto_send: bool) -> str | None:
    """What the mic component reported: log it, announce anything already done while you spoke, and either send it or park it in the
    review box. Returns text to send now, or None."""
    st.session_state.last_voice = voice["nonce"]
    heard = (voice.get("text") or "").strip()
    session.log.add("stt", "live dictation", heard[:60] or "(nothing heard)", None, float(voice.get("ms") or 0))
    done = voice.get("done") or []
    for clause in done:
        session.log.add("code", "while speaking", f"ran early: {clause}", None, 0.0)
    heard = (voice.get("remaining") if done else heard) or ""
    note = ("\n\n[Already done while the user was speaking: " + "; ".join(done) + ". Do not repeat it.]") if done else ""
    if done:
        st.toast("⚡ Already done while you spoke: " + "; ".join(done))
    action = route_speech(heard, done, locked, auto_send)
    if action == "send":
        return heard + note
    if action == "review":
        st.session_state["dictation"] = heard
    elif not done:
        st.warning("I did not hear any speech. Check the microphone and try again.")
    return None  # (nothing left to send: everything you said was a quick command and it already ran)


def render(session: AssistantSession, transcriber, predictor, make_quick) -> Submission | None:
    """Draw the composer. Returns what to send, or None. (An answer typed while an approval is pending is handled here too.)"""
    awaiting = session.pending is not None and not session.busy  # a question is waiting: what you say answers it (after you review it)
    locked = session.busy or session.pending is not None
    if st.session_state.pop("clear_dictation", False):
        st.session_state["dictation"] = ""  # before the text area exists on this run
    auto_send = st.session_state.get("auto_send", False)
    text = None
    uploads: list = []
    gen = st.session_state.setdefault("upload_gen", 0)

    with getattr(st, "_bottom", None) or st.container():
        if st.session_state.get("dictation"):
            with st.container(border=True, key="dictation_card"):
                st.caption("🎙 This is what I heard. Fix anything, then send." + (" It will answer the question above." if awaiting else ""))
                edited = st.text_area("What I heard", key="dictation", height=80, label_visibility="collapsed", disabled=session.busy)
                c1, c2, _ = st.columns([1, 1, 5])
                if c1.button("Send", type="primary", disabled=session.busy or not edited.strip(), width="stretch"):
                    text = edited.strip()
                    st.session_state["clear_dictation"] = True
                if c2.button("Discard", disabled=session.busy, width="stretch"):
                    st.session_state["clear_dictation"] = True
                    st.rerun()

        mic, attach = st.columns([7, 1], vertical_alignment="top")  # the attach button stays level with the Talk button while the transcript grows below it
        try:
            get_voice_server(transcriber, predictor, make_quick)
            with mic:
                voice = _live_voice(port=config.VOICE_PORT, disabled=session.busy, key="voice", default=None)
        except OSError as e:
            voice = None
            mic.warning(f"Live voice is unavailable (port {config.VOICE_PORT} busy?): {e}")
        with attach.popover("📎", help="Attach files or images for the assistant", width="stretch"):
            uploads = st.file_uploader("Files or images for the assistant", accept_multiple_files=True, key=f"uploads_{gen}", disabled=locked) or []
        if uploads:
            st.caption("📎 " + ", ".join(f.name for f in uploads) + ": sent with your next message")

    if voice and voice.get("nonce") != st.session_state.get("last_voice"):
        text = _heard(session, voice, locked, auto_send) or text  # never auto-sent while a question waits: noise must not be able to say "yes"

    # The text box works while an approval is pending: type yes / no, or tell me what to change.
    if session.busy:
        placeholder = "Working… press Stop to interrupt"
    elif session.pending:
        placeholder = "Type yes / no, or tell me what to change..."
    else:
        placeholder = "Ask anything..."
    typed = st.chat_input(placeholder, disabled=session.busy)
    if typed and session.pending and not session.busy:
        session.answer_pending(typed)
        st.rerun()
    text = typed or text
    if text and awaiting:  # the dictation card's Send: the spoken words answer the pending question
        session.answer_pending(text)
        st.rerun()
    if text and not locked:
        st.session_state["upload_gen"] = gen + 1  # a fresh uploader: the files just sent must not ride along with the next message
        return Submission(text, [(f.name, f.getvalue()) for f in uploads])
    return None
