import numpy as np
import pytest

from vcm.capture import CaptureRate, pick_capture_rate


class FakeSd:
    """Stand-in for sounddevice, scripted per test."""

    def __init__(self, ok_rates, default_rate):
        self.ok_rates = ok_rates
        self.default_rate = default_rate

    def check_input_settings(self, device=None, samplerate=None, channels=None):
        if samplerate not in self.ok_rates:
            raise Exception(f"Invalid sample rate ({samplerate})")

    def query_devices(self, device, kind):
        return {"default_samplerate": self.default_rate}


def test_picks_target_sr_directly_when_the_device_supports_it(monkeypatch):
    import vcm.capture as capture_mod

    monkeypatch.setattr(capture_mod, "sd", FakeSd(ok_rates={16000}, default_rate=16000))
    rate = pick_capture_rate()
    assert rate == CaptureRate(16000, resample=False)


def test_falls_back_to_device_native_rate_when_target_sr_unsupported(monkeypatch):
    import vcm.capture as capture_mod

    monkeypatch.setattr(capture_mod, "sd", FakeSd(ok_rates={48000}, default_rate=48000))
    rate = pick_capture_rate()
    assert rate == CaptureRate(48000, resample=True)


def test_raises_a_clear_error_for_a_non_integer_native_rate(monkeypatch):
    import vcm.capture as capture_mod

    monkeypatch.setattr(capture_mod, "sd", FakeSd(ok_rates={44100}, default_rate=44100))
    with pytest.raises(RuntimeError, match="44100"):
        pick_capture_rate()


def test_capture_rate_to_target_is_a_passthrough_when_no_resample_needed():
    rate = CaptureRate(16000, resample=False)
    wav = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.array_equal(rate.to_target(wav), wav)


def test_capture_rate_to_target_resamples_when_needed():
    rate = CaptureRate(48000, resample=True)
    wav = np.zeros(48000, dtype=np.float32)
    out = rate.to_target(wav)
    assert abs(len(out) - 16000) <= 1
