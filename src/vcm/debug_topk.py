"""Throwaway debug tool: record one 3s clip, print the top-5 class
probabilities instead of just the winner. For diagnosing close-call
confusions (e.g. volume_up vs volume_down) that a bare top-1 log can't show.

Usage (from src/): python -m vcm.debug_topk [--model-dir ../models]
"""
from __future__ import annotations

import argparse

import numpy as np

from .capture import record_seconds
from .classifier import Classifier
from .dispatch import EspeakSpeaker
from .paths import root_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=root_path("models"))
    p.add_argument("--device", default=None)
    p.add_argument("--chunks", type=int, default=6)
    args = p.parse_args()

    clf = Classifier(args.model_dir)
    speaker = EspeakSpeaker()
    print(f"Recording {args.chunks} x 3s chunks. The Pi will say \"go\" right before each one --"
          " start speaking the moment you hear it, not before.")
    for n in range(args.chunks):
        speaker.say("go")  # blocks until spoken -- real audible sync cue, not chat-message timing
        wav = record_seconds(3.0, device=args.device)
        probs, latency_ms = clf.predict_probs(wav)
        order = np.argsort(probs)[::-1][:5]
        print(f"--- chunk {n + 1} ({latency_ms:.1f}ms) ---")
        for i in order:
            print(f"  {clf.idx_to_label[int(i)]:<30} {probs[i]:.3f}")


if __name__ == "__main__":
    main()
