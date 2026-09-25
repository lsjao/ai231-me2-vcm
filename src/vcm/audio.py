"""Audio loading and log-mel feature extraction.

Fixed preprocessing constants here must match whatever the Pi-side capture
pipeline uses at inference time -- see models/training_config.json, which
train.py writes out so the deployment code has no excuse to drift from this.

The log-mel is plain numpy (numerically checked against librosa in
tests/test_audio.py) so the Pi needs neither librosa nor numba. librosa is
only imported lazily, for resampling files in load_waveform.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
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
        wav = resample(wav, sr, target_sr)
    return fix_length(wav, CLIP_SAMPLES)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import librosa
    except ImportError as e:
        raise RuntimeError(
            f"resampling {orig_sr}->{target_sr} Hz needs librosa (pip install librosa); "
            "the live mic path records at the target rate and never needs it"
        ) from e
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)


def fix_length(wav: np.ndarray, n_samples: int) -> np.ndarray:
    if len(wav) >= n_samples:
        return wav[:n_samples]
    pad = n_samples - len(wav)
    return np.pad(wav, (0, pad), mode="constant")


def speech_level_ok(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    frame_ms: int = 20,
    min_peak_dbfs: float = -38.0,
    min_contrast_db: float = 20.0,
    min_active_ms: int = 150,
) -> bool:
    """Is there real speech in this take, or just room noise that happened to
    wiggle? Needs the loudest frame to be absolutely loud, well above the
    take's own floor, AND that loudness to be sustained (a click or a chair
    creak is a few frames; even a short word like "next" is 150ms+).
    (trim_to_speech alone is relative, so it happily 'finds' speech in a
    silent room.)"""
    frame = int(sr * frame_ms / 1000)
    n_frames = len(wav) // frame
    if n_frames == 0:
        return False
    frames = wav[: n_frames * frame].astype(np.float64).reshape(n_frames, frame)
    db = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(frames**2, axis=1)), 1e-10))
    floor = np.percentile(db, 10)
    if db.max() < min_peak_dbfs or db.max() - floor < min_contrast_db:
        return False
    active = db >= max(floor + 15.0, min_peak_dbfs - 6.0)
    return int(active.sum()) >= max(1, min_active_ms // frame_ms)


def trim_to_speech(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    frame_ms: int = 20,
    margin_db: float = 12.0,
    pad_ms: int = 100,
    min_dbfs: float = -55.0,
    min_active_ms: int = 60,
) -> np.ndarray | None:
    """Trim leading/trailing silence with an energy gate set relative to the
    clip's own quietest frames, keeping `pad_ms` of margin. Returns None if
    no speech-like energy is found. Recorded clips and the live VAD both go
    through this so training and inference see speech at the same offset
    (synthetic clips start at t=0; real recordings otherwise wouldn't).
    """
    frame = int(sr * frame_ms / 1000)
    n_frames = len(wav) // frame
    if n_frames == 0:
        return None
    frames = wav[: n_frames * frame].astype(np.float64).reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))

    thresh = max(np.percentile(db, 10) + margin_db, min_dbfs)
    active = np.flatnonzero(db > thresh)
    if active.size < max(1, min_active_ms // frame_ms):
        return None

    pad = int(sr * pad_ms / 1000)
    start = max(0, int(active[0]) * frame - pad)
    end = min(len(wav), (int(active[-1]) + 1) * frame + pad)
    return wav[start:end]


# -- log-mel (numpy port of librosa's melspectrogram + power_to_db) ----------

def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    """Slaney mel scale (librosa's default, htk=False)."""
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_part = min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep
    return np.where(f >= min_log_hz, log_part, f / f_sp)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_part = min_log_hz * np.exp(logstep * (m - min_log_mel))
    return np.where(m >= min_log_mel, log_part, f_sp * m)


@lru_cache(maxsize=None)
def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    fft_freqs = np.linspace(0, sr / 2.0, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]  # Slaney area norm
    return weights.astype(np.float32)


@lru_cache(maxsize=None)
def _hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)  # periodic


def _power_spectrogram(wav: np.ndarray) -> np.ndarray:
    padded = np.pad(wav.astype(np.float32), N_FFT // 2, mode="constant")  # librosa center=True
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(padded[idx] * _hann(N_FFT), axis=1)
    return (spec.real**2 + spec.imag**2).T  # (1 + n_fft/2, n_frames)


def logmel(wav: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Log-mel spectrogram, shape (n_mels, n_frames), float32."""
    mel = _mel_filterbank(sr, N_FFT, N_MELS, float(FMIN), float(FMAX)) @ _power_spectrogram(wav)
    amin = 1e-10
    db = 10.0 * np.log10(np.maximum(amin, mel))
    db -= 10.0 * np.log10(np.maximum(amin, mel.max()))
    db = np.maximum(db, db.max() - 80.0)  # top_db=80, as power_to_db
    # normalize to roughly [-1, 1] per-clip so clip loudness doesn't dominate
    return ((db + 40.0) / 40.0).astype(np.float32)


def feature_shape() -> tuple[int, int, int]:
    """(n_mels, n_frames, 1) shape the model expects, computed from a dummy clip."""
    dummy = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    n_frames = logmel(dummy).shape[1]
    return (N_MELS, n_frames, 1)


def waveform_to_features(wav: np.ndarray) -> np.ndarray:
    feat = logmel(wav)
    return feat[..., np.newaxis]  # (n_mels, n_frames, 1)
