"""Streaming energy-based endpointer: turns a stream of fixed-size mic frames
into utterance clips.

Speech is detected as energy a margin above a slowly-adapting noise floor
(frozen while speech is active, so a long utterance doesn't drag the floor up
with it). Each utterance is cut with ~100 ms of context on both sides, the
same padding audio.trim_to_speech applies to the recordings in the training
set, so live clips look like training clips.

Constants are placeholders until tuned against the real mic (HANDOFF.md):
notably ONSET_MARGIN_DB against the dual-fan noise floor.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import audio
from .ambient_volume import FRAME_MS, FRAME_SAMPLES, frame_dbfs

ONSET_MARGIN_DB = 12.0    # frame is "active" this far above the floor
HOLD_MARGIN_DB = 8.0      # ...and stays active above this lower bar (hysteresis)
MIN_DBFS = -55.0          # never trigger below this, however quiet the floor is
FLOOR_ALPHA = 0.05        # floor adaptation rate on non-active frames
CALIBRATION_FRAMES = 10   # initial frames used to seed the floor (~300 ms)

PREROLL_MS = 100
TAIL_MS = 100
MIN_SPEECH_MS = 90        # consecutive active frames needed to open an utterance
HANGOVER_MS = 500         # silence that ends an utterance
MIN_UTTERANCE_MS = 200
MAX_UTTERANCE_SAMPLES = audio.CLIP_SAMPLES


def _frames(ms: int) -> int:
    return max(1, ms // FRAME_MS)


class Endpointer:
    def __init__(self):
        self._floor: float | None = None
        self._calib: list[float] = []
        self._preroll: deque[np.ndarray] = deque(maxlen=_frames(PREROLL_MS))
        self._pending: list[np.ndarray] = []  # active frames before the utterance opens
        self._utt: list[np.ndarray] | None = None
        self._silent_run = 0

    @property
    def floor_dbfs(self) -> float | None:
        return self._floor

    def _threshold(self, margin: float) -> float:
        return max(self._floor + margin, MIN_DBFS)

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed one FRAME_SAMPLES frame; returns a finished utterance or None."""
        assert len(frame) == FRAME_SAMPLES, f"expected {FRAME_SAMPLES} samples, got {len(frame)}"
        level = frame_dbfs(frame)

        if self._floor is None:
            self._calib.append(level)
            self._preroll.append(frame)
            if len(self._calib) >= CALIBRATION_FRAMES:
                self._floor = float(np.mean(self._calib))
            return None

        in_speech = self._utt is not None
        active = level > self._threshold(HOLD_MARGIN_DB if in_speech else ONSET_MARGIN_DB)

        if not active:
            self._floor += FLOOR_ALPHA * (level - self._floor)

        if not in_speech:
            if active:
                self._pending.append(frame)
                if len(self._pending) >= _frames(MIN_SPEECH_MS):
                    self._utt = list(self._preroll) + self._pending
                    self._pending = []
                    self._silent_run = 0
            else:
                self._pending = []
                self._preroll.append(frame)
            return None

        self._utt.append(frame)
        self._silent_run = 0 if active else self._silent_run + 1
        if self._silent_run >= _frames(HANGOVER_MS):
            return self._finish(tail_frames=_frames(TAIL_MS))
        if sum(len(f) for f in self._utt) >= MAX_UTTERANCE_SAMPLES:
            return self._finish(tail_frames=0)
        return None

    def flush(self) -> np.ndarray | None:
        """End of stream: emit an utterance still in progress, if any."""
        if self._utt is None:
            return None
        return self._finish(tail_frames=min(self._silent_run, _frames(TAIL_MS)))

    def _finish(self, tail_frames: int) -> np.ndarray | None:
        frames = self._utt
        drop = max(0, self._silent_run - tail_frames)
        if drop:
            frames = frames[: len(frames) - drop]
        self._utt = None
        self._silent_run = 0
        self._preroll.clear()
        clip = np.concatenate(frames)
        if len(clip) < MIN_UTTERANCE_MS * audio.TARGET_SR // 1000:
            return None
        return clip[:MAX_UTTERANCE_SAMPLES].astype(np.float32)
