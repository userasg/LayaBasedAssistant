"""The Files view: the folder you and the assistant share, and the sandbox's scratch space.

A table of what is there and ONE file open at a time (the old panel read and drew a preview for every file on every rerun)."""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from ..session import AssistantSession


@st.cache_data(show_spinner=False, max_entries=64)
def _file_bytes(session_dir: str, path: str, size: int) -> bytes:
    """Fetched once per (file, size): a download button needs the bytes, and re-reading every sandbox file on every
    rerun (one `docker exec` each) made the page slower with each file the agent created."""
    return st.session_state.session.download(path)


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def ago(ts: float) -> str:
    s = max(0, time.time() - ts)
    if s < 90:
        return "just now"
    if s < 5400:
        return f"{s / 60:.0f} min ago"
    if s < 172800:
        return f"{s / 3600:.0f} h ago"
    return time.strftime("%d %b", time.localtime(ts))


def _preview_and_actions(session: AssistantSession, rel: str) -> None:
    shared = session.shared
    pv = shared.preview(rel)
    with st.container(border=True):
        st.markdown(f"**{Path(rel).name}**")
        if pv.kind == "text":
            st.code(pv.text, language=None)
        elif pv.kind == "csv":
            st.dataframe(pv.rows[1:], width="stretch") if len(pv.rows) > 1 else st.write(pv.rows)
        elif pv.kind == "image":
            st.image(pv.data)
        else:
            st.caption("No preview for this file type.")
        c = st.columns(4, vertical_alignment="bottom")
        c[0].download_button("Download", (shared.root / rel).read_bytes(), file_name=Path(rel).name, key=f"dl_sh_{rel}", width="stretch")
        if c[1].button("Show in Finder", key=f"fin_{rel}", width="stretch"):
            shared.open_in_finder(rel)
        new = c[2].text_input("Rename to", value=Path(rel).name, key=f"rn_{rel}", label_visibility="collapsed")
        if new != Path(rel).name and c[2].button("Rename", key=f"rnb_{rel}", width="stretch"):
            try:
                shared.rename(rel, new)
                st.rerun()
            except (FileExistsError, ValueError) as e:
                st.error(str(e))
        if c[3].button("Move to trash", key=f"tr_{rel}", width="stretch"):
            shared.trash(rel)
            st.rerun()


def render(session: AssistantSession) -> None:
    shared = session.shared
    st.caption(f"`{shared.root}` is shared with the assistant (it sees it as `/workspace/shared`). Deleting moves a file to a recoverable trash.")
    up = st.file_uploader("Drop files here", accept_multiple_files=True, key="shared_up")
    for f in up or []:
        key = f"saved_{f.name}_{f.size}"
        if not st.session_state.get(key):
            shared.save_upload(f.name, f.getvalue())
            st.session_state[key] = True
    files = shared.list()
    if not files:
        st.info("The folder is empty. Ask the assistant to make something, or drop a file above.")
    else:
        st.dataframe([{"File": f.rel, "Size": human_size(f.size), "Changed": ago(f.mtime)} for f in files[:100]], hide_index=True, width="stretch",
                     height=min(35 * (len(files) + 1) + 3, 280))
        pick = st.selectbox("Open a file", [f.rel for f in files], index=None, placeholder="Pick a file to preview it…", label_visibility="collapsed")
        if pick:
            _preview_and_actions(session, pick)
    trash = shared.list_trash()
    if trash:
        with st.expander(f"Trash ({len(trash)})"):
            for t in trash[-10:]:
                if st.button(f"Restore {t.name.split('__', 1)[-1]}", key=f"rs_{t.name}"):
                    shared.restore(t.name)
                    st.rerun()
    with st.expander("Sandbox scratch space"):
        st.caption("The agent's private working area (deleted with the session). Anything worth keeping goes in the shared folder.")
        rows = session.list_files_sized()[:15]
        if not rows:
            st.caption("Nothing here yet.")
        for path, size in rows:
            st.download_button(f"{path.replace('/workspace/', '')} · {human_size(size)}", _file_bytes(str(session.session_dir), path, size),
                               file_name=path.rsplit("/", 1)[-1], key=f"dl_sb_{path}_{size}")
