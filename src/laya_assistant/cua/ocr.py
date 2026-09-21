"""Reading the words on a window's screenshot, on this Mac, for free (Apple's Vision framework; a few hundred ms).

Why: many apps draw list rows, cards, tiles and web content whose text never appears in the accessibility tree (measured: Music's sidebar
rows, Notes' note list and Finder's file rows are all actionable elements with NO label and no labelled child). The loop cannot pick "the
Odd Future playlist" from a list of unnamed rows. The screenshot has the words; this puts them back on the elements they sit inside.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextBox:
    text: str
    x: float  # pixels in the screenshot, origin top-left
    y: float
    w: float
    h: float
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def read_text(png: bytes, width: int, height: int, min_conf: float = 0.3) -> list[TextBox]:
    import Foundation
    import Vision

    data = Foundation.NSData.dataWithBytes_length_(png, len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        return []
    boxes = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        c = cand[0]
        if c.confidence() < min_conf or not str(c.string()).strip():
            continue
        bb = obs.boundingBox()  # normalised, origin bottom-left
        boxes.append(TextBox(str(c.string()).strip(), bb.origin.x * width, (1 - bb.origin.y - bb.size.height) * height,
                             bb.size.width * width, bb.size.height * height, float(c.confidence())))
    return boxes


def label_from_boxes(frame: tuple[float, float, float, float], boxes: list[TextBox]) -> str:
    """The words inside an element's frame, in reading order (top to bottom, then left to right)."""
    x, y, w, h = frame
    inside = [b for b in boxes if x - 2 <= b.cx <= x + w + 2 and y - 2 <= b.cy <= y + h + 2]
    inside.sort(key=lambda b: (round(b.cy / max(h / 3, 6)), b.x))
    return " ".join(b.text for b in inside)
