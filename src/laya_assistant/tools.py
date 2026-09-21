"""Tools the agent can call: web search and image description (local vision model)."""
import base64

from ddgs import DDGS
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from . import config
from .engine import llm_lock, make_llm


def internet_search(query: str, max_results: int = 3) -> str:
    """Search the internet. Returns title, url and snippet for each result."""
    results = DDGS().text(query, max_results=max_results)
    return "\n\n".join(f"{r['title']} ({r['href']}): {r['body']}" for r in results)


def make_describe_image(backend):
    """A `describe_image` tool bound to the agent's own filesystem (the sandbox), so it can look at
    uploaded images and screenshots the agent saved there. Runs the local vision model (qwen3-vl)."""

    @tool
    def describe_image(path: str, question: str = "Describe this image in detail.") -> str:
        """Look at an image file in the workspace (for example /workspace/uploads/photo.png) and answer a question about it."""
        got = backend.download_files([path])[0]
        if got.error or got.content is None:
            return f"Could not read {path}: {got.error or 'empty'}"
        b64 = base64.b64encode(got.content).decode()
        msg = HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ])
        with llm_lock():
            return str(make_llm(config.VISION_MODEL, temperature=0.1).invoke([msg]).content)

    return describe_image
