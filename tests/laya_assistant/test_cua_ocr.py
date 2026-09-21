"""Words read off a window's screenshot, put back on the controls they sit inside. Real Apple Vision OCR on a real rendered image."""
import io

import pytest

from laya_assistant.cua import ocr
from laya_assistant.cua.desktop import NEEDS_TEXT, _frame, enrich_with_screenshot
from laya_assistant.cua.types import Element


def box(text, x, y, w=80, h=16):
    return ocr.TextBox(text, x, y, w, h)


def test_the_words_inside_a_frame_are_joined_in_reading_order():
    boxes = [box("Future", 60, 10), box("Odd", 10, 10), box("elsewhere", 400, 300), box("Playlist", 10, 30)]
    assert ocr.label_from_boxes((0, 0, 200, 50), boxes) == "Odd Future Playlist"
    assert ocr.label_from_boxes((0, 100, 200, 50), boxes) == ""


def test_unnamed_rows_get_their_words_nameless_ones_are_dropped_and_uncovered_words_become_click_targets():
    rows = [Element("e1", "option", "", frame=(0, 0, 200, 30), actions=("AXPress",)), Element("e2", "option", "", frame=(0, 40, 200, 30), actions=("AXPress",)),
            Element("e3", "button", "Play", frame=(300, 0, 40, 30), actions=("AXPress",))]
    boxes = [box("Odd Future", 8, 6), box("Welcome back", 400, 200, 120, 20), box("Play", 305, 6, 30)]
    out = enrich_with_screenshot(rows, boxes)
    by = {e.id: e for e in out}
    assert by["e1"].label == "Odd Future" and by["e1"].source == "ocr"
    assert "e2" not in by  # no name from anywhere: useless as a target
    assert by["e3"].label == "Play" and not any(e.label == "Play" and e.role == "text" for e in out)  # a covered word is not offered twice
    text = [e for e in out if e.role == "text"]
    assert [e.label for e in text] == ["Welcome back"] and text[0].id.startswith("t") and text[0].source == "ocr"


def test_frames_are_mapped_from_screen_points_into_the_screenshots_pixels():
    state = {"window_bounds": {"x": 500, "y": 200, "width": 1000, "height": 600}, "screenshot_width": 500}
    f = _frame({"frame": {"x": 600, "y": 300, "w": 100, "h": 40}}, state)
    assert f == (50.0, 50.0, 50.0, 20.0)  # (600-500)*0.5, (300-200)*0.5 ...
    assert _frame({"frame": {"x": 1, "y": 1, "w": 1, "h": 1}}, {}) is None


def test_real_ocr_reads_real_pixels():
    pytest.importorskip("Vision")
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (600, 200), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 44)
    d.text((20, 20), "Odd Future", fill="black", font=font)
    d.text((20, 110), "Recently Added", fill="black", font=font)
    buf = io.BytesIO(); img.save(buf, "PNG")
    boxes = ocr.read_text(buf.getvalue(), 600, 200)
    texts = " | ".join(b.text for b in boxes)
    assert "Odd Future" in texts and "Recently Added" in texts, texts
    top = next(b for b in boxes if "Odd" in b.text)
    assert top.y < 100 and 10 < top.x < 60  # and where they are is right (top-left origin)
    assert ocr.label_from_boxes((0, 0, 600, 100), boxes) == "Odd Future"


def test_roles_that_get_words_from_the_screenshot():
    assert {"option", "button", "group"} <= NEEDS_TEXT and "textbox" not in NEEDS_TEXT
