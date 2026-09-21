"""Page styling. Colours are Streamlit's own theme plus neutral translucent greys, so light and dark mode both work.

Nothing dynamic is ever put into HTML: the only markup is this static stylesheet. Text that comes from the model or from tools (commands,
search queries, page text) always goes through Streamlit's markdown or code elements."""
import streamlit as st

ACCENT = "#ff4b4b"

CSS = f"""
/* page: a comfortable reading column, room at the bottom for the pinned composer */
[data-testid="stMainBlockContainer"] {{ max-width: 820px; padding-top: 4.2rem; padding-bottom: 11rem; }}
[data-testid="stAppDeployButton"] {{ display: none; }}
h1 {{ font-size: 1.65rem !important; font-weight: 650 !important; letter-spacing: -0.01em; padding: 0 0 .15rem 0 !important; }}

/* sidebar: quiet, compact */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ margin-bottom: .3rem; }}
[data-testid="stSidebar"] [data-testid="stExpander"] {{ border-color: rgba(128,128,128,.22); }}
[data-testid="stSidebar"] h3 {{ padding-top: .2rem; }}

/* the conversation */
[data-testid="stChatMessage"] {{ padding: .45rem .25rem; }}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{ margin-bottom: .45rem; }}
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {{ opacity: .78; }}
[data-testid="stChatMessage"] [data-testid="stExpander"] {{ border-color: rgba(128,128,128,.22); margin-top: .25rem; }}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary {{ padding: .35rem .75rem; font-size: .85rem; }}

/* the live card: a slim accent rule, and a pulsing dot in front of the headline */
.st-key-live {{ border-left: 3px solid {ACCENT}55; padding-left: .8rem; }}
[class*="st-key-live_head"] p {{ margin: 0; }}
[class*="st-key-live_head"] p::before {{
  content: ""; display: inline-block; width: .55rem; height: .55rem; margin-right: .5rem; border-radius: 50%;
  background: {ACCENT}; animation: laya-pulse 1.1s ease-in-out infinite;
}}
.st-key-live_head_stopping p::before {{ animation-duration: .45s; }}
@keyframes laya-pulse {{ 50% {{ opacity: .2; }} }}

/* the approval card stands out: this is the one place the assistant is waiting for you */
.st-key-approval {{ border-color: {ACCENT}88 !important; background: {ACCENT}0d; }}

/* the composer: pinned above the chat box */
[data-testid="stBottom"] > div {{ padding-top: .25rem; }}
.st-key-dictation_card {{ margin-bottom: .35rem; }}

/* example prompts on the empty screen */
.st-key-examples button {{ justify-content: flex-start; min-height: 3.4rem; height: auto; padding: .55rem .9rem; }}
.st-key-examples button > div {{ justify-content: flex-start; width: 100%; }}
.st-key-examples button p {{ white-space: normal; text-align: left; font-size: .92rem; }}
"""


def inject() -> None:
    st.html(f"<style>{CSS}</style>")
