"""Train the from-scratch CRNN intent classifier and export for the Pi.

Usage:
    python -m vcm.train --data-root . --manifest manifest.csv --output-dir ../models

Writes to output-dir:
    vcm_crnn.keras            -- full Keras model
    vcm_crnn.tflite           -- dynamic-range quantized TFLite export
    labels.json               -- index -> intent label
    training_config.json      -- feature extraction params, must match Pi-side capture
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from . import audio, data, model as model_lib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=".", help="root dir manifest filepaths are relative to")
    p.add_argument("--manifest", default="manifest.csv")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    manifest_path = os.path.join(args.data_root, args.manifest)
    rows = data.read_manifest(manifest_path, args.data_root)
    labels = data.build_label_list(rows)
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
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
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

    target_names = [idx_to_label[i] for i in range(len(labels))]
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
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
        f.write(json.dumps(cm, indent=2))

    print(f"artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
