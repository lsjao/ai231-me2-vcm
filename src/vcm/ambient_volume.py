"""Ambient auto-volume: adapts playback volume to room noise level.

This is assignment feature #4 (renamed from the handoff's planned
`ambient_and_beatsync_skeleton.py` -- beat-sync, #3, is cut entirely and
confirmed dead, so there's no beatsync half left to keep a "skeleton" name
for; this is the real module, not a skeleton).

PLACEHOLDER CONSTANTS: SAMPLE_RATE, FRAME_MS, and the noise-floor -> volume
curve below are all guesses made without a real mic. HANDOFF.md flags dual-
fan noise near the mic as an untested risk -- the moment mic + Pi are
together, measure the actual ambient noise floor (with and without the fans
running) and recalibrate NOISE_FLOOR_QUIET_DBFS / NOISE_FLOOR_LOUD_DBFS
below. Everything else (the tracker, hysteresis, rate limiting) should not
need to change.

Tested here against synthetic white noise only, not real ambient audio --
same caveat as the classifier: synthetic signal proves the algorithm logic
works, not that the calibration is right for a real room.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000  # placeholder, matches vcm.audio.TARGET_SR
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

# noise floor tracker: rises slowly (won't be fooled by a brief loud sound),
# falls quickly (recognizes a quieter room right away)
FLOOR_ATTACK = 0.3   # weight applied when the frame is quieter than the floor
FLOOR_RELEASE = 0.02  # weight applied when the frame is louder than the floor

DBFS_FLOOR = -80.0  # clamp for near-silence / digital silence frames

# placeholder calibration curve -- see module docstring
NOISE_FLOOR_QUIET_DBFS = -50.0
NOISE_FLOOR_LOUD_DBFS = -20.0
VOLUME_AT_QUIET = 30
VOLUME_AT_LOUD = 90

MIN_VOLUME_STEP = 5           # hysteresis: ignore suggestions smaller than this
UPDATE_INTERVAL_S = 2.0        # rate limit: re-evaluate at most this often
UPDATE_INTERVAL_FRAMES = max(1, int(UPDATE_INTERVAL_S * 1000 / FRAME_MS))


def frame_dbfs(frame: np.ndarray) -> float:
    """RMS level of a mono float32 frame in [-1, 1], in dBFS."""
    rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
    if rms <= 0.0:
        return DBFS_FLOOR
    return max(DBFS_FLOOR, 20.0 * np.log10(rms))


def volume_for_noise_floor(noise_dbfs: float) -> int:
    """Map an estimated noise floor (dBFS) to a suggested playback volume
    0-100, linearly interpolated between the quiet/loud calibration points
    and clamped outside them."""
    if noise_dbfs <= NOISE_FLOOR_QUIET_DBFS:
        return VOLUME_AT_QUIET
    if noise_dbfs >= NOISE_FLOOR_LOUD_DBFS:
        return VOLUME_AT_LOUD
    span = NOISE_FLOOR_LOUD_DBFS - NOISE_FLOOR_QUIET_DBFS
    frac = (noise_dbfs - NOISE_FLOOR_QUIET_DBFS) / span
    volume = VOLUME_AT_QUIET + frac * (VOLUME_AT_LOUD - VOLUME_AT_QUIET)
    return int(round(volume))


class AmbientNoiseTracker:
    """Asymmetric EMA noise-floor estimator over successive audio frames."""

    def __init__(self, initial_dbfs: float = NOISE_FLOOR_QUIET_DBFS):
        self.floor_dbfs = initial_dbfs

    def update(self, frame: np.ndarray) -> float:
        level = frame_dbfs(frame)
        alpha = FLOOR_ATTACK if level < self.floor_dbfs else FLOOR_RELEASE
        self.floor_dbfs += alpha * (level - self.floor_dbfs)
        return self.floor_dbfs


class AmbientAutoVolume:
    """Feed it consecutive audio frames; periodically get back a suggested
    volume change. Suppressed (paused()) while the state machine has ducked
    for a wake-word command window, so ambient adjustment doesn't fight the
    duck -- caller is responsible for calling pause()/resume() around that,
    same as duck()/unduck() on PlayMusicStateMachine (no wake-word detector
    exists yet to wire this automatically, see HANDOFF.md).
    """

    def __init__(self, current_volume: int = 50):
        self.tracker = AmbientNoiseTracker()
        self.last_applied_volume = current_volume
        self._frames_since_update = 0
        self._paused = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def update(self, frame: np.ndarray) -> int | None:
        """Call once per FRAME_SAMPLES-sized chunk of audio. Returns a new
        suggested volume if the tracker has settled on a materially
        different level, else None (nothing to change yet)."""
        self.tracker.update(frame)
        if self._paused:
            return None

        self._frames_since_update += 1
        if self._frames_since_update < UPDATE_INTERVAL_FRAMES:
            return None
        self._frames_since_update = 0

        suggested = volume_for_noise_floor(self.tracker.floor_dbfs)
        if abs(suggested - self.last_applied_volume) < MIN_VOLUME_STEP:
            return None

        self.last_applied_volume = suggested
        return suggested
