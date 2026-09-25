import glob

import librosa
import numpy as np
import pytest

from vcm import audio

SR = audio.TARGET_SR


def librosa_logmel(wav):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_fft=audio.N_FFT, hop_length=audio.HOP_LENGTH,
        n_mels=audio.N_MELS, fmin=audio.FMIN, fmax=audio.FMAX, power=2.0,
    )
    return ((librosa.power_to_db(mel, ref=np.max) + 40.0) / 40.0).astype(np.float32)


def dataset_clips(n=6):
    files = sorted(glob.glob("../dataset/**/*.wav", recursive=True))
    step = max(1, len(files) // n)
    return files[::step][:n]


@pytest.mark.parametrize("path", dataset_clips())
def test_numpy_logmel_matches_librosa_on_real_clips(path):
    wav = audio.load_waveform(path)
    ours, ref = audio.logmel(wav), librosa_logmel(wav)
    assert ours.shape == ref.shape
    assert np.allclose(ours, ref, atol=2e-3), np.abs(ours - ref).max()


def test_numpy_logmel_matches_librosa_on_noise_and_tones():
    rng = np.random.default_rng(0)
    t = np.arange(audio.CLIP_SAMPLES) / SR
    for wav in (
        (rng.standard_normal(audio.CLIP_SAMPLES) * 0.1).astype(np.float32),
        (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
    ):
        assert np.allclose(audio.logmel(wav), librosa_logmel(wav), atol=2e-3)


def test_feature_shape_matches_model_input():
    assert audio.feature_shape() == (audio.N_MELS, 301, 1)
    assert audio.waveform_to_features(np.zeros(audio.CLIP_SAMPLES, dtype=np.float32)).shape == (40, 301, 1)


def test_silence_does_not_produce_nans():
    feat = audio.logmel(np.zeros(audio.CLIP_SAMPLES, dtype=np.float32))
    assert np.isfinite(feat).all()
