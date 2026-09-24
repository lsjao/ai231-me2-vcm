"""Audio loading and log-mel feature extraction.

Fixed preprocessing constants here must match whatever the Pi-side capture
pipeline uses at inference time -- see models/training_config.json, which
train.py writes out so the deployment code has no excuse to drift from this.
"""

from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 16000
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_SECONDS)

N_MELS = 40
N_FFT = 400          # 25ms at 16kHz
HOP_LENGTH = 160      # 10ms at 16kHz
FMIN = 50
FMAX = 7600


def load_waveform(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load a WAV file as mono float32 at target_sr, pad/trim to CLIP_SAMPLES."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return fix_length(wav, CLIP_SAMPLES)


def fix_length(wav: np.ndarray, n_samples: int) -> np.ndarray:
    if len(wav) >= n_samples:
        return wav[:n_samples]
    pad = n_samples - len(wav)
    return np.pad(wav, (0, pad), mode="constant")


def logmel(wav: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Log-mel spectrogram, shape (n_mels, n_frames), float32."""
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    logmel_db = librosa.power_to_db(mel, ref=np.max)
    # normalize to roughly [-1, 1] per-clip so clip loudness doesn't dominate
    logmel_db = (logmel_db + 40.0) / 40.0
    return logmel_db.astype(np.float32)


def feature_shape() -> tuple[int, int, int]:
    """(n_mels, n_frames, 1) shape the model expects, computed from a dummy clip."""
    dummy = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    n_frames = logmel(dummy).shape[1]
    return (N_MELS, n_frames, 1)


def waveform_to_features(wav: np.ndarray) -> np.ndarray:
    feat = logmel(wav)
    return feat[..., np.newaxis]  # (n_mels, n_frames, 1)
