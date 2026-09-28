"""Mic capture at whatever rate the device actually supports, transparently
resampled to audio.TARGET_SR.

Found the hard way: a cheap USB mic that only supports 48000 Hz capture,
not TARGET_SR (16000), even though PortAudio happily reports it as a valid
input device. Assuming a mic supports the model's rate natively is not
safe -- pi_check caught this before it silently broke the live demo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from . import audio


@dataclass
class CaptureRate:
    device_sr: int
    resample: bool

    def to_target(self, wav: np.ndarray) -> np.ndarray:
        if not self.resample:
            return wav.astype(np.float32)
        return audio.resample_integer_ratio(wav, self.device_sr, audio.TARGET_SR)


def pick_capture_rate(device: int | str | None = None) -> CaptureRate:
    """TARGET_SR if the device supports it directly (no resampling needed,
    the common case); otherwise the device's own default rate, decimated
    down after capture."""
    try:
        sd.check_input_settings(device=device, samplerate=audio.TARGET_SR, channels=1)
        return CaptureRate(audio.TARGET_SR, resample=False)
    except Exception:
        pass

    native_sr = int(sd.query_devices(device, "input")["default_samplerate"])
    sd.check_input_settings(device=device, samplerate=native_sr, channels=1)
    if native_sr % audio.TARGET_SR != 0:
        raise RuntimeError(
            f"input device only supports {native_sr} Hz, which isn't an integer "
            f"multiple of {audio.TARGET_SR} Hz -- can't resample without librosa"
        )
    return CaptureRate(native_sr, resample=True)


def record_seconds(seconds: float, device: int | str | None = None) -> np.ndarray:
    """One-shot recording, resampled to TARGET_SR mono float32."""
    rate = pick_capture_rate(device)
    n = int(rate.device_sr * seconds)
    rec = sd.rec(n, samplerate=rate.device_sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return rate.to_target(rec[:, 0])
