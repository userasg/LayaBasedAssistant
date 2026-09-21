"""Every step is recorded: the menu, what each system said, what was done, and whether it worked.

This is the training data for making Laya trustworthy at choosing (a fine-tune on these rows), and the source of the
promotion statistics in authority.py. Nothing here changes behaviour."""
from __future__ import annotations

import json
import time
from pathlib import Path


class StepRecorder:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.rows: list[dict] = []

    def add(self, **row) -> dict:
        row["ts"] = round(time.time(), 3)
        self.rows.append(row)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row
