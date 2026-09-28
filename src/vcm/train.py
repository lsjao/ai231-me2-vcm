"""Train the from-scratch CRNN command classifier and export for the Pi.

Usage:  run.cmd vcm.train        (all paths default to the project root)

Real recordings in data_real/ are merged in automatically when present.
--extra-data (repeatable) instead names other roots that have a manifest.csv,
e.g. a pooled classmate dataset (this replaces the automatic data_real).

Writes to output-dir:
    vcm_crnn.keras            -- full Keras model
    vcm_crnn.tflite           -- dynamic-range quantized TFLite export
    labels.json               -- index -> command label ("intent/slot", or "reject")
    training_config.json      -- feature extraction params, must match Pi-side capture
    eval_report.txt           -- per-command report, per-intent rollup, confusion matrix
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .paths import root_path
from . import audio, data, labels as label_utils, model as model_lib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=root_path(), help="root dir manifest filepaths are relative to")
    p.add_argument("--manifest", default="manifest.csv")
    p.add_argument("--extra-data", action="append", default=[],
                   help="extra data root containing manifest.csv (repeatable)")
    p.add_argument("--output-dir", default=root_path("models"))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def intent_rollup(y_true: list[int], y_pred: list[int], idx_to_label: dict[int, str]) -> str:
    """Per-intent accuracy where a prediction counts if it names the right
    *intent*, even with the wrong slot -- separates 'heard media_control but
    picked the wrong action' from 'didn't recognize a command at all'."""
    stats: dict[str, list[int]] = {}
    for t, p in zip(y_true, y_pred):
        intent = label_utils.intent_of(idx_to_label[t])
        s = stats.setdefault(intent, [0, 0, 0])  # n, command-correct, intent-correct
        s[0] += 1
        s[1] += int(t == p)
        s[2] += int(label_utils.intent_of(idx_to_label[p]) == intent)
    lines = ["per-intent rollup (val): intent  n  command_acc  intent_acc"]
    for intent in sorted(stats):
        n, cmd_ok, int_ok = stats[intent]
        lines.append(f"  {intent:<16} {n:>4}  {cmd_ok / n:.2f}  {int_ok / n:.2f}")
    total = len(y_true)
    if total:
        cmd = sum(t == p for t, p in zip(y_true, y_pred)) / total
        lines.append(f"  overall command accuracy {cmd:.2f} over {total} val clips")
    return "\n".join(lines) + "\n"


# commands most likely to be confused with the wake phrase if it's not
# phonetically distinct enough, or if the wake window logic has a bug
WAKE_WATCH_LABELS = [
    "media_control/stop",
    "media_control/volume_down",
    "media_control/next",
    "ask_time/none",
    "reject",
]


def wake_confusion_report(y_true: list[int], y_pred: list[int], idx_to_label: dict[int, str]) -> str:
    """Wake-class precision/recall, plus specific confusion counts against
    the commands most likely to be mixed up with it (see WAKE_WATCH_LABELS)."""
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    wake_label = next((lbl for lbl in label_to_idx if label_utils.intent_of(lbl) == "wake"), None)
    if wake_label is None:
        return "(no wake label in this label set)\n"
    wi = label_to_idx[wake_label]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == wi and p == wi)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == wi and p != wi)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != wi and p == wi)
    n_wake = tp + fn
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / n_wake if n_wake else float("nan")

    lines = [f"wake-class report ({wake_label}): n={n_wake} precision={precision:.2f} recall={recall:.2f}"]
    for watch in WAKE_WATCH_LABELS:
        wi2 = label_to_idx.get(watch)
        if wi2 is None:
            continue
        wake_as_watch = sum(1 for t, p in zip(y_true, y_pred) if t == wi and p == wi2)
        watch_as_wake = sum(1 for t, p in zip(y_true, y_pred) if t == wi2 and p == wi)
        lines.append(
            f"  {wake_label} -> {watch}: {wake_as_watch}   |   {watch} -> {wake_label}: {watch_as_wake}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    manifest_path = os.path.join(args.data_root, args.manifest)
    rows = label_utils.read_manifest(manifest_path, args.data_root)
    extra_roots = list(args.extra_data)
    real_root = root_path("data_real")
    if not extra_roots and os.path.exists(os.path.join(real_root, "manifest.csv")):
        extra_roots = [real_root]  # your recordings, picked up automatically
    for extra_root in extra_roots:
        extra = label_utils.read_manifest(os.path.join(extra_root, "manifest.csv"), extra_root)
        print(f"extra data {extra_root}: {len(extra)} clips")
        rows.extend(extra)
    labels = label_utils.build_label_list(rows)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    train_rows, val_rows = data.split_rows(rows, args.val_fraction, args.seed)
    print(f"total={len(rows)} train={len(train_rows)} val={len(val_rows)} classes={len(labels)}")

    train_ds = data.make_dataset(
        train_rows, label_to_idx, args.batch_size, augment=True, shuffle=True
    )
    val_ds = data.make_dataset(
        val_rows, label_to_idx, args.batch_size, augment=False, shuffle=False
    )

    weights = data.class_weights(train_rows, label_to_idx)

    input_shape = audio.feature_shape()
    model = model_lib.build_crnn(input_shape, len(labels))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        # val_accuracy is too coarse to use as the stopping signal once the
        # val set is this small (94 clips -> ~1% per correct/incorrect
        # prediction) -- it was plateauing on an early fluke while train
        # loss was still visibly improving. val_loss is smoother.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )

    # evaluation report on held-out split
    y_true, y_pred = [], []
    for feats, lbls in val_ds:
        probs = model.predict(feats, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(lbls.numpy().tolist())

    all_idx = list(range(len(labels)))
    target_names = [idx_to_label[i] for i in all_idx]
    report = classification_report(
        y_true, y_pred, labels=all_idx, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_idx).tolist()
    report += "\n" + intent_rollup(y_true, y_pred, idx_to_label)
    report += "\n" + wake_confusion_report(y_true, y_pred, idx_to_label)
    print(report)

    os.makedirs(args.output_dir, exist_ok=True)

    model.save(os.path.join(args.output_dir, "vcm_crnn.keras"))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(args.output_dir, "vcm_crnn.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"wrote {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")

    with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
        json.dump(idx_to_label, f, indent=2)

    config = {
        "target_sr": audio.TARGET_SR,
        "clip_seconds": audio.CLIP_SECONDS,
        "n_mels": audio.N_MELS,
        "n_fft": audio.N_FFT,
        "hop_length": audio.HOP_LENGTH,
        "fmin": audio.FMIN,
        "fmax": audio.FMAX,
        "input_shape": list(input_shape),
    }
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(args.output_dir, "eval_report.txt"), "w") as f:
        f.write(report)
        f.write("\nconfusion_matrix (rows=true, cols=pred, label order matches labels.json):\n")
        f.write("\n".join(json.dumps(row) for row in cm))

    print(f"artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
