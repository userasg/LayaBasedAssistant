"""Streamlit UI for the Laya Assistant: layout only. All decisions live in session.py / cortex.py; what each part of the page looks
like lives in ui/ (chat, composer, sidebar, files, style).

Run:  uv run streamlit run src/laya_assistant/app.py --server.fileWatcherType none

Design rules (each one fixes a problem seen in real use):
  * Turns run on a worker thread (session.submit); the page only polls the feed. A click, toggle or mic event can
    never abort a turn, and nothing holds the GPU lock while the page is being drawn.
  * One answer per request. The model's narration and tool calls are folded under "Show work", not drawn as a wall of bubbles.
  * The live card says what is happening right now, ticks the plan off, and has a Stop button.
  * Review-before-send is the default for voice: a mis-heard transcript is fixed with the keyboard, not sent.
  * The composer (mic, attach, review card) is pinned above the chat box; the page stays small however long the chat gets.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `laya_assistant` imports work under `streamlit run`

import streamlit as st

from laya_assistant import config
from laya_assistant.cua.bridge import DEFAULT_PORT, BridgeServer
from laya_assistant.cua.desktop import CuaCli
from laya_assistant.cua.menu import Ranker
from laya_assistant.cua.policy import DomainPolicy
from laya_assistant.cua.wiring import LazyBrowser, make_computer_use
from laya_assistant.engine import check_models, load_laya
from laya_assistant.session import AssistantSession
from laya_assistant.stream_intent import AppIndex, QuickExecutor
from laya_assistant.stt import Transcriber
from laya_assistant.ui import chat, composer, files, sidebar, style

CHAT, FILES = "chat", "files"


@st.cache_resource(show_spinner="Loading Laya (about 20 s the first time)...")
def get_predictor():
    return load_laya()


@st.cache_resource(show_spinner="Loading speech-to-text (Whisper on the GPU)...")
def get_transcriber():
    t = Transcriber()
    t.warm()
    return t


@st.cache_resource(show_spinner="Loading the element matcher...")
def get_ranker():
    r = Ranker()
    r.warm()
    return r


@st.cache_resource(show_spinner=False)
def get_bridge():
    """The local end of the Laya Bridge Chrome extension. Started once per process."""
    b = BridgeServer(port=DEFAULT_PORT)
    try:
        b.start()
    except OSError:
        return None
    return b


@st.cache_resource(show_spinner=False)
def get_domain():
    return DomainPolicy()  # one policy for the whole process: sites you approve stay approved


def main():
    st.set_page_config(page_title="Laya Assistant", page_icon="🎙", layout="centered")
    style.inject()

    try:
        missing = check_models()
    except Exception:
        st.error("Ollama is not running. Start it (`ollama serve`) and reload.")
        st.stop()
    if missing:
        st.error("Missing Ollama models: " + ", ".join(f"`ollama pull {m}`" for m in missing))
        st.stop()

    predictor = get_predictor()
    transcriber = get_transcriber()
    bridge, domain, cli = get_bridge(), get_domain(), CuaCli()
    if "session" not in st.session_state:
        with st.spinner("Starting the sandbox and warming the models..."):
            cu = make_computer_use(predictor, get_ranker(), bridge, cli, domain, recorder_path=config.HOME / "cua_steps.jsonl")
            st.session_state.session = AssistantSession.create(predictor, computer=cu)
        st.session_state.last_voice = None
    session: AssistantSession = st.session_state.session

    sidebar.render(session, predictor, bridge, domain, cli)

    head, nav = st.columns([4, 1], vertical_alignment="center")
    head.title("Laya Assistant")
    view = st.session_state.setdefault("view", CHAT)
    if nav.button("📁 Files" if view == CHAT else "← Chat", key="nav", width="stretch"):
        st.session_state["view"] = FILES if view == CHAT else CHAT
        st.rerun()

    chosen = None
    if view == FILES:
        files.render(session)
    elif chat.history(session) == 0 and not session.busy and session.pending is None:
        chosen = chat.render_empty()

    # ---- the running turn: polled, never blocking the page ----
    def live():
        if session.busy:
            (chat.render_live_compact if view == FILES else chat.render_live)(session)
        elif session.turn_open:
            session.turn_open = False
            if session.pending:
                st.session_state["view"] = CHAT  # a question is waiting: bring the person to it
            st.rerun()  # the turn finished: redraw the history (and any approval card) from the saved state

    st.fragment(run_every=0.4 if session.busy else None)(live)()

    if session.pending and not session.busy:
        if view == FILES:
            st.warning("The assistant is waiting for your OK. Go back to the chat, or type yes / no below.")
        else:
            chat.render_approval(session)

    # ---- composer: mic, attach, review card, chat box ----
    def make_quick():
        return QuickExecutor(predictor, AppIndex(), domain, browser=LazyBrowser(bridge) if bridge is not None else None)

    sent = composer.render(session, transcriber, predictor, make_quick)
    if sent is None and chosen:
        sent = composer.Submission(chosen)
    if sent and not (session.busy or session.pending is not None):
        session.submit(sent.text, sent.uploads)
        st.session_state["view"] = CHAT
        st.rerun()


if __name__ == "__main__":
    main()
