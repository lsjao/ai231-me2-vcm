import librosa
import numpy as np
import pytest

from vcm import audio
from vcm.audio import resample_integer_ratio


def tone(freq, seconds, sr):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_same_rate_is_a_no_op():
    wav = tone(300, 0.5, 16000)
    out = resample_integer_ratio(wav, 16000, 16000)
    assert np.array_equal(out, wav.astype(np.float32))


def test_rejects_non_integer_ratios():
    with pytest.raises(ValueError):
        resample_integer_ratio(np.zeros(100, dtype=np.float32), 44100, 16000)


def test_48k_to_16k_matches_librosa_on_a_tone():
    wav = tone(440, 1.0, 48000)
    ours = resample_integer_ratio(wav, 48000, 16000)
    ref = librosa.resample(wav, orig_sr=48000, target_sr=16000)
    n = min(len(ours), len(ref))
    # skip filter warm-up/tail at the edges, compare the steady-state middle
    assert np.allclose(ours[50:n-50], ref[50:n-50], atol=0.03)


def test_output_length_matches_the_ratio():
    wav = np.zeros(48000 * 2, dtype=np.float32)  # 2s at 48kHz
    out = resample_integer_ratio(wav, 48000, 16000)
    assert abs(len(out) - 16000 * 2) <= 1


def test_attenuates_a_tone_above_the_new_nyquist():
    # 12kHz exceeds the new Nyquist (8kHz at 16kHz output) -- it must be
    # killed before decimating from 48kHz, or it aliases into the audible band
    wav = tone(12000, 0.2, 48000)
    out = resample_integer_ratio(wav, 48000, 16000)
    in_rms = np.sqrt(np.mean(wav**2))
    out_rms = np.sqrt(np.mean(out[20:-20] ** 2))
    assert out_rms < 0.1 * in_rms


def test_feeds_the_real_feature_pipeline_without_nans():
    wav = tone(300, audio.CLIP_SECONDS + 0.1, 48000)
    down = resample_integer_ratio(wav, 48000, audio.TARGET_SR)
    clip = audio.fix_length(down, audio.CLIP_SAMPLES)
    feat = audio.waveform_to_features(clip)
    assert feat.shape == audio.feature_shape()
    assert np.isfinite(feat).all()
