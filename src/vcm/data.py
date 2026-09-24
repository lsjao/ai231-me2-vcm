"""Manifest loading, train/val split, and tf.data pipeline with augmentation.

Dataset is tiny (a couple hundred synthetic clips) and entirely TTS so far --
augmentation here is a partial stand-in for the real-voice gap called out in
HANDOFF.md, not a fix for it. Retrain once real recordings land in dataset/.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from . import audio

AUGMENT_NOISE_STD = 0.01
AUGMENT_MAX_SHIFT_FRAC = 0.1
AUGMENT_GAIN_DB_RANGE = 6.0


@dataclass
class ManifestRow:
    filepath: str
    intent: str


def read_manifest(manifest_path: str, data_root: str) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ManifestRow(
                    filepath=os.path.join(data_root, r["filepath"]),
                    intent=r["intent"],
                )
            )
    return rows


def build_label_list(rows: list[ManifestRow]) -> list[str]:
    return sorted({r.intent for r in rows})


def split_rows(
    rows: list[ManifestRow], val_fraction: float = 0.2, seed: int = 1337
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    labels = [r.intent for r in rows]
    train_rows, val_rows = train_test_split(
        rows,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
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
    labels = [label_to_idx[r.intent] for r in rows]

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
        counts[label_to_idx[r.intent]] += 1
    total = counts.sum()
    n_classes = len(label_to_idx)
    weights = total / (n_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}
