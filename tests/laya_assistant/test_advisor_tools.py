"""Advisor graph and tools against real Laya, real Docker and the real vision model."""
import io
import json
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from laya_assistant import advisor, tools
from laya_assistant.decisions import DecisionLog
from laya_assistant.sandbox import DockerSandbox


def test_advisor_is_a_compiled_subagent_returning_json(laya_model):
    log = DecisionLog()
    sub = advisor.build_advisor(laya_model, log)
    assert sub["name"] == "laya-advisor" and "runnable" in sub and sub["description"]
    sub["runnable"].invoke({"messages": [HumanMessage(content="write a python script")]})  # warm
    t0 = time.perf_counter()
    out = sub["runnable"].invoke({"messages": [HumanMessage(content="write a python function and test it")]})
    ms = (time.perf_counter() - t0) * 1000
    reply = json.loads(out["messages"][-1].content)
    assert reply["subagent"] in {"researcher", "coder", "reviewer", "none"}
    assert 0.0 <= reply["confidence"] <= 1.0 and set(reply["probabilities"]) == {"researcher", "coder", "reviewer", "none"}
    print(f"advisor: {ms:.0f} ms")
    assert ms < 250
    assert any(r["step"] == "advisor" and r["system"] == "laya" for r in log.rows)


def test_advisor_accuracy_is_measured_and_printed(laya_model):
    cases = [("write a python function that sorts a list and test it", "coder"),
             ("find recent articles about local speech recognition", "researcher"),
             ("look up the latest MLX release notes", "researcher"),
             ("fix the failing unit test", "coder"),
             ("check whether the code we wrote is correct", "reviewer"),
             ("review the report for mistakes", "reviewer"),
             ("hello", "none"), ("what is 2 plus 2", "none")]
    sub = advisor.build_advisor(laya_model, DecisionLog())
    got = [(json.loads(sub["runnable"].invoke({"messages": [HumanMessage(content=t)]})["messages"][-1].content), want)
           for t, want in cases]
    hits = sum(g["subagent"] == want for g, want in got)
    print(f"advisor accuracy: {hits}/{len(cases)}", [(g["subagent"], round(g["confidence"], 2), w) for g, w in got])
    # advisory only: the executor weighs it with its confidence; require better than chance (0.25) with margin
    assert hits >= 3


@pytest.fixture(scope="module")
def box():
    b = DockerSandbox()
    yield b
    b.stop()


def make_png(text="STOP"):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (400, 200), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 120)
    except OSError:
        font = ImageFont.load_default()
    d.text((20, 30), text, fill="red", font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_describe_image_reads_from_the_sandbox_and_sees_the_text(box):
    box.upload_files([("/workspace/uploads/sign.png", make_png("STOP"))])
    describe = tools.make_describe_image(box)
    out = describe.invoke({"path": "/workspace/uploads/sign.png", "question": "What word is written in this image?"})
    print("vision says:", out)
    assert "stop" in out.lower()


def test_describe_image_reports_a_missing_file_plainly(box):
    out = tools.make_describe_image(box).invoke({"path": "/workspace/nope.png"})
    assert "Could not read" in out


def test_internet_search_returns_titles_and_urls():
    out = tools.internet_search("python programming language", max_results=2)
    assert "http" in out
