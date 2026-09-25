"""Command labels and manifest reading -- deliberately free of TensorFlow so
everything that only needs labels (dispatch, benchmark harness, pipeline) can
run on the Pi with just tflite-runtime installed.

The classifier predicts a *command* label ("intent/slot", e.g.
"media_control/next", "play_music/playlist_jazz", "set_timer/5min"), so the
downstream state machine gets the fine-grained action directly instead of
just "media_control". All reject slots (offvocab/nearmiss/silence/noise)
collapse to a single "reject" label -- confusing them with each other is
harmless, only reject-vs-command matters.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

REJECT = "reject"


def command_label(intent: str, slot: str) -> str:
    if intent == REJECT:
        return REJECT
    return f"{intent}/{slot}"


def intent_of(label: str) -> str:
    return label.split("/", 1)[0]


def slot_of(label: str) -> str:
    return label.split("/", 1)[1] if "/" in label else ""


@dataclass
class ManifestRow:
    filepath: str
    intent: str
    slot: str

    @property
    def label(self) -> str:
        return command_label(self.intent, self.slot)


def read_manifest(manifest_path: str, data_root: str) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ManifestRow(
                    filepath=os.path.join(data_root, r["filepath"]),
                    intent=r["intent"],
                    slot=r["slot"],
                )
            )
    return rows


def build_label_list(rows: list[ManifestRow]) -> list[str]:
    return sorted({r.label for r in rows})
