"""Manifest loading, train/val split, and tf.data pipeline with augmentation.

Labels are command labels ("intent/slot"), defined in labels.py. The
synthetic set is small and TTS-only; real recordings live in a separate
root (see record_dataset.py) and are merged in at train time.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import tensorflow as tf

from . import audio
from .labels import ManifestRow

AUGMENT_NOISE_STD = 0.01
AUGMENT_MAX_SHIFT_FRAC = 0.1
AUGMENT_GAIN_DB_RANGE = 6.0


def split_rows(
    rows: list[ManifestRow], val_fraction: float = 0.2, seed: int = 1337
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """Per-label split: each label with >=2 clips gets at least one val clip;
    labels with a single clip stay in train. (sklearn's stratified split
    refuses when there are more labels than val slots, which is the normal
    case with ~35 command labels and a small dataset.)"""
    by_label: dict[str, list[ManifestRow]] = defaultdict(list)
    for r in rows:
        by_label[r.label].append(r)

    rng = random.Random(seed)
    train_rows: list[ManifestRow] = []
    val_rows: list[ManifestRow] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction)) if len(group) >= 2 else 0
        val_rows.extend(group[:n_val])
        train_rows.extend(group[n_val:])
    return train_rows, val_rows


def _augment_waveform(wav: np.ndarray) -> np.ndarray:
    n = len(wav)

    # random gain
    gain_db = np.random.uniform(-AUGMENT_GAIN_DB_RANGE, AUGMENT_GAIN_DB_RANGE)
    wav = wav * (10.0 ** (gain_db / 20.0))

    # random circular time shift
    max_shift = int(n * AUGMENT_MAX_SHIFT_FRAC)
    if max_shift > 0:
        shift = np.random.randint(-max_shift, max_shift + 1)
        wav = np.roll(wav, shift)

    # additive gaussian noise
    wav = wav + np.random.normal(0.0, AUGMENT_NOISE_STD, size=n).astype(np.float32)

    return np.clip(wav, -1.0, 1.0).astype(np.float32)


def _load_features(filepath: bytes, augment: bool) -> np.ndarray:
    wav = audio.load_waveform(filepath.decode("utf-8"))
    if augment:
        wav = _augment_waveform(wav)
    return audio.waveform_to_features(wav)


def make_dataset(
    rows: list[ManifestRow],
    label_to_idx: dict[str, int],
    batch_size: int,
    augment: bool,
    shuffle: bool,
) -> tf.data.Dataset:
    filepaths = [r.filepath for r in rows]
    labels = [label_to_idx[r.label] for r in rows]

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(rows), reshuffle_each_iteration=True)

    feat_shape = audio.feature_shape()

    def _map_fn(filepath, label):
        feat = tf.numpy_function(
            func=lambda fp: _load_features(fp, augment),
            inp=[filepath],
            Tout=tf.float32,
        )
        feat.set_shape(feat_shape)
        return feat, label

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def class_weights(rows: list[ManifestRow], label_to_idx: dict[str, int]) -> dict[int, float]:
    counts = np.zeros(len(label_to_idx), dtype=np.float64)
    for r in rows:
        counts[label_to_idx[r.label]] += 1
    total = counts.sum()
    n_classes = len(label_to_idx)
    weights = total / (n_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}
