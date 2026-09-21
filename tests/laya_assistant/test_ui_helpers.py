"""The small pure helpers behind the chat and files views. The important property: text that comes from the model or from a tool
(commands, search results, page text) can never turn into a link, an image, emphasis or HTML in the chat."""
import re

from laya_assistant.transcript import Step
from laya_assistant.ui import chat, files


def test_model_and_tool_text_is_shown_as_text_never_as_markup():
    hostile = "[click](http://evil.example) ![x](http://evil.example/p.png) <b>hi</b> **bold** _it_ `tick` # head | col ~gone~"
    shown = chat.plain(hostile)
    for ch in "[]!<>*_`#|~":
        assert not re.search(rf"(?<!\\){re.escape(ch)}", shown), ch  # every special character is escaped
    assert "http://evil.example" in shown  # the text itself is kept


def test_code_spans_cannot_be_broken_out_of():
    assert "`" not in chat.code("a`b`c")[1:-1] and chat.code("x").startswith("`") and chat.code("x").endswith("`")


def test_the_plan_marks_progress_and_the_step_list_puts_results_in_code():
    md = chat.plan_md([{"content": "search", "status": "completed"}, {"content": "read *it*", "status": "in_progress"}, {"content": "note", "status": "pending"}])
    assert md.splitlines() == ["✅ search  ", "🔄 **read \\*it\\***  ", "⬜ note"]
    steps = [Step("🔎", "Searching the web", "Searched the web", "q", "![x](http://evil.example/p.png)", failed=True),
             Step("💬", "I will look.", "I will look.", kind="say")]
    out = chat.steps_md(steps)
    assert "**Searched the web** ⚠️ `q`" in out and "`![x](http://evil.example/p.png)`" in out  # inside a code span: inert
    assert "> I will look." in out


def test_sizes_and_ages_read_like_a_person_would_say_them():
    assert [files.human_size(n) for n in (0, 999, 1024, 1536, 5 * 1024 * 1024)] == ["0 B", "999 B", "1.0 KB", "1.5 KB", "5.0 MB"]
    import time

    assert files.ago(time.time() - 5) == "just now" and files.ago(time.time() - 600) == "10 min ago" and files.ago(time.time() - 7200) == "2 h ago"
