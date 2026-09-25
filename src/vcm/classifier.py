"""Trained TFLite command classifier, shared by the benchmark harness and the
live pipeline. Reuses vcm.audio's feature extraction so preprocessing exactly
matches training -- do not duplicate that logic elsewhere."""

from __future__ import annotations

import json
import os
import time

import numpy as np

RUNTIME = "ai_edge_litert"
try:
    # Google's current standalone runtime (successor to tflite_runtime); newest
    # op-version support, which matters because the model is converted with TF 2.20
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter

        RUNTIME = "tflite_runtime"
    except ImportError:
        # dev machine: full tensorflow
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
        RUNTIME = "tensorflow"

from . import audio


class Classifier:
    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "labels.json")) as f:
            self.idx_to_label: dict[int, str] = {int(k): v for k, v in json.load(f).items()}
        self.interpreter = Interpreter(model_path=os.path.join(model_dir, "vcm_crnn.tflite"))
        self.interpreter.allocate_tensors()
        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]
        # first inference pays one-off init cost (seconds on a Pi); pay it now, not on the user's first command
        self.predict_probs(np.zeros(audio.CLIP_SAMPLES, dtype=np.float32))

    def predict_probs(self, wav: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (per-label probabilities, latency_ms). Latency covers
        feature extraction + inference, not audio capture."""
        t0 = time.perf_counter()
        feat = audio.waveform_to_features(audio.fix_length(wav, audio.CLIP_SAMPLES))
        x = feat[np.newaxis, ...].astype(np.float32)
        self.interpreter.set_tensor(self._input["index"], x)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self._output["index"])[0]
        return probs, (time.perf_counter() - t0) * 1000.0

    def predict(self, wav: np.ndarray) -> tuple[str, float, float]:
        """Returns (command_label, confidence, latency_ms)."""
        probs, latency_ms = self.predict_probs(wav)
        idx = int(np.argmax(probs))
        return self.idx_to_label[idx], float(probs[idx]), latency_ms
