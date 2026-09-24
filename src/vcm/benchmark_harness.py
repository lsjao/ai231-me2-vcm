"""Evaluator-logging benchmark harness.

Assignment requirement (professor's own quote): "should be able to respond
to any person" -- the benchmark must include evaluators other than the
project owner, each command said n times per evaluator, with output logs
of transcript/prediction/confidence/latency per attempt.

On "transcript": this project has no ASR/transcription model anywhere in
the pipeline (audio -> intent classifier directly, no cloud, no LLM, per
HANDOFF.md's hard constraints) -- there is no transcription step to log the
output of. "Transcript" here is read as the ground-truth prompt phrase the
evaluator was asked to say, logged alongside the prediction for each
attempt. That's a judgment call, not a certainty; flagged in HANDOFF.md.

Two run modes:
  - "live": records from a real microphone via `sounddevice`. This is the
    mode evaluators actually run on the Pi. `sounddevice` isn't installed
    on this dev machine (no mic here either) -- it's imported lazily so
    the rest of the harness is usable and testable without it.
  - "replay": feeds existing WAV files from the dataset manifest through
    the same logging pipeline instead of a live mic. This exists purely to
    smoke-test the harness itself (CSV writing, classifier wiring, correct/
    incorrect scoring, latency measurement) before a mic exists to do it
    for real -- replayed synthetic clips are NOT a substitute for the real
    evaluator benchmark the assignment requires, and the CSV rows say so
    (source=synthetic_replay) so they can't be mistaken for it later.

Usage:
    # smoke-test against the existing synthetic dataset, no mic needed
    python -m vcm.benchmark_harness --mode replay --evaluator smoketest \
        --data-root .. --manifest ../manifest.csv --reps 1

    # real evaluator session on the Pi, once a mic exists -- addendum's
    # plan: every play_media command, 3x each (see HANDOFF.md)
    python -m vcm.benchmark_harness --mode live --evaluator "Jane Doe" --evaluator-plan
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

try:
    # lean runtime this actually ships with on the Pi (requirements-pi.txt)
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # dev machine: full tensorflow, no tflite_runtime installed
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

from . import audio, data

CSV_FIELDS = [
    "timestamp",
    "evaluator",
    "source",           # "live_mic" or "synthetic_replay"
    "attempt_index",
    "expected_intent",
    "slot",
    "phrase",            # the "transcript" -- see module docstring
    "predicted_intent",
    "confidence",
    "correct",
    "latency_ms",
    "audio_filepath",
]


@dataclass
class Prompt:
    intent: str
    slot: str
    phrase: str


def load_prompts(
    phrase_list_path: str,
    granularity: str = "intent",
    intents: set[str] | None = None,
) -> list[Prompt]:
    """granularity="intent": one phrase per intent (broad coverage, short
    session). "slot": one phrase per (intent, slot) -- e.g. every distinct
    play_media command (play/pause/stop/next/previous/volume_up/volume_down/
    each playlist/whats_playing/each easter egg), not just "media_control"
    once. "phrase": every row, including repeated phrasings of the same
    slot. `intents`, if given, restricts to those intents first.

    The addendum's evaluator plan ("each evaluator says every play_media
    command 3 times") is `granularity="slot", intents={"media_control",
    "play_music"}` -- see HANDOFF.md and the CLI --evaluator-plan flag.
    """
    if granularity not in ("intent", "slot", "phrase"):
        raise ValueError(f"unknown granularity: {granularity!r}")

    prompts: list[Prompt] = []
    seen_keys: set = set()
    with open(phrase_list_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intent, slot = row["intent"], row["slot"]
            if intent == "reject" and slot in ("silence", "noise"):
                continue  # no real phrase to say for these, TTS placeholders
            if intents is not None and intent not in intents:
                continue

            key = intent if granularity == "intent" else (intent, slot)
            if granularity != "phrase" and key in seen_keys:
                continue
            seen_keys.add(key)
            prompts.append(Prompt(intent=intent, slot=slot, phrase=row["phrase"]))
    return prompts


class Classifier:
    """Thin wrapper around the trained TFLite model. Reuses vcm.audio's
    feature extraction so preprocessing exactly matches training -- do not
    duplicate that logic here."""

    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "labels.json")) as f:
            self.idx_to_label = {int(k): v for k, v in json.load(f).items()}
        self.interpreter = Interpreter(
            model_path=os.path.join(model_dir, "vcm_crnn.tflite")
        )
        self.interpreter.allocate_tensors()
        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]

    def predict(self, wav: np.ndarray) -> tuple[str, float, float]:
        """Returns (predicted_intent, confidence, latency_ms). Latency covers
        feature extraction + inference, not audio capture."""
        t0 = time.perf_counter()
        feat = audio.waveform_to_features(wav)
        x = feat[np.newaxis, ...].astype(np.float32)
        self.interpreter.set_tensor(self._input["index"], x)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self._output["index"])[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        idx = int(np.argmax(probs))
        return self.idx_to_label[idx], float(probs[idx]), latency_ms


class LiveMicSource:
    """Records `audio.CLIP_SECONDS` of audio from the default input device."""

    def __init__(self):
        try:
            import sounddevice as sd
        except ImportError as e:
            raise RuntimeError(
                "live mode needs the 'sounddevice' package (pip install sounddevice), "
                "plus a working mic -- not available on this dev machine, "
                "use --mode replay to smoke-test the harness instead"
            ) from e
        self._sd = sd

    def capture(self, prompt: Prompt) -> np.ndarray:
        input(f'Say: "{prompt.phrase}"  (press Enter, then speak)')
        n_samples = int(audio.TARGET_SR * audio.CLIP_SECONDS)
        rec = self._sd.rec(n_samples, samplerate=audio.TARGET_SR, channels=1, dtype="float32")
        self._sd.wait()
        return rec[:, 0]


class ReplaySource:
    """Cycles existing dataset WAVs per intent, for harness smoke-testing
    without a mic. NOT a real evaluator -- see module docstring."""

    def __init__(self, manifest_path: str, data_root: str, seed: int = 0):
        rows = data.read_manifest(manifest_path, data_root)
        self._by_intent: dict[str, list[str]] = {}
        for r in rows:
            self._by_intent.setdefault(r.intent, []).append(r.filepath)
        self._rng = random.Random(seed)
        for paths in self._by_intent.values():
            self._rng.shuffle(paths)
        self._cursor: dict[str, int] = {intent: 0 for intent in self._by_intent}

    def capture(self, prompt: Prompt) -> tuple[np.ndarray, str]:
        paths = self._by_intent.get(prompt.intent)
        if not paths:
            raise KeyError(f"no replay clips available for intent {prompt.intent!r}")
        i = self._cursor[prompt.intent] % len(paths)
        self._cursor[prompt.intent] += 1
        path = paths[i]
        return audio.load_waveform(path), path


def run_session(
    evaluator: str,
    prompts: list[Prompt],
    reps: int,
    mode: str,
    classifier: Classifier,
    output_csv: str,
    audio_out_dir: str | None,
    manifest_path: str | None = None,
    data_root: str | None = None,
) -> list[dict]:
    if mode == "live":
        source = LiveMicSource()
    elif mode == "replay":
        assert manifest_path and data_root
        source = ReplaySource(manifest_path, data_root)
    else:
        raise ValueError(f"unknown mode: {mode}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    if audio_out_dir:
        os.makedirs(audio_out_dir, exist_ok=True)

    write_header = not os.path.exists(output_csv)
    rows: list[dict] = []

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for prompt in prompts:
            for attempt_index in range(1, reps + 1):
                if mode == "live":
                    wav = source.capture(prompt)
                    src_path = None
                    source_label = "live_mic"
                else:
                    wav, src_path = source.capture(prompt)
                    source_label = "synthetic_replay"

                predicted_intent, confidence, latency_ms = classifier.predict(wav)
                correct = predicted_intent == prompt.intent

                audio_filepath = ""
                if audio_out_dir:
                    fname = f"{evaluator}_{prompt.intent}_{prompt.slot}_{attempt_index}.wav"
                    fname = fname.replace(" ", "_")
                    audio_filepath = os.path.join(audio_out_dir, fname)
                    import soundfile as sf

                    sf.write(audio_filepath, wav, audio.TARGET_SR)

                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "evaluator": evaluator,
                    "source": source_label,
                    "attempt_index": attempt_index,
                    "expected_intent": prompt.intent,
                    "slot": prompt.slot,
                    "phrase": prompt.phrase,
                    "predicted_intent": predicted_intent,
                    "confidence": round(confidence, 4),
                    "correct": correct,
                    "latency_ms": round(latency_ms, 2),
                    "audio_filepath": audio_filepath,
                }
                writer.writerow(row)
                f.flush()
                rows.append(row)
                print(
                    f"[{prompt.intent}/{prompt.slot} #{attempt_index}] "
                    f"said={prompt.phrase!r} pred={predicted_intent} "
                    f"conf={confidence:.2f} correct={correct} lat={latency_ms:.1f}ms"
                )

    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    accuracy = sum(1 for r in rows if r["correct"]) / n
    mean_conf = sum(r["confidence"] for r in rows) / n
    mean_latency = sum(r["latency_ms"] for r in rows) / n

    per_intent: dict[str, dict] = {}
    for r in rows:
        intent = r["expected_intent"]
        bucket = per_intent.setdefault(intent, {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(r["correct"])
    for intent, bucket in per_intent.items():
        bucket["accuracy"] = bucket["correct"] / bucket["n"]

    # retries-until-success: for each (evaluator, intent, slot) command,
    # how many attempts until the first correct prediction, if any
    commands: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["evaluator"], r["expected_intent"], r["slot"])
        commands.setdefault(key, []).append(r)

    attempts_to_success = []
    never_succeeded = 0
    for attempts in commands.values():
        attempts.sort(key=lambda r: r["attempt_index"])
        first_success = next((r["attempt_index"] for r in attempts if r["correct"]), None)
        if first_success is None:
            never_succeeded += 1
        else:
            attempts_to_success.append(first_success)

    return {
        "n_attempts": n,
        "accuracy": accuracy,
        "mean_confidence": mean_conf,
        "mean_latency_ms": mean_latency,
        "per_intent": per_intent,
        "commands_tested": len(commands),
        "commands_never_succeeded": never_succeeded,
        "mean_attempts_to_success": (
            sum(attempts_to_success) / len(attempts_to_success) if attempts_to_success else None
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluator", required=True, help="evaluator's name, used in filenames/logs")
    p.add_argument("--mode", choices=["live", "replay"], default="replay")
    p.add_argument("--reps", type=int, default=3, help="times each command is said per evaluator")
    p.add_argument("--phrase-list", default="phrase_list.csv")
    p.add_argument("--granularity", choices=["intent", "slot", "phrase"], default="intent")
    p.add_argument(
        "--intents", default=None,
        help="comma-separated intent allowlist, e.g. media_control,play_music",
    )
    p.add_argument(
        "--evaluator-plan", action="store_true",
        help="shortcut for the addendum's plan: every play_media command, "
        "3x each (equivalent to --granularity slot --intents media_control,play_music --reps 3)",
    )
    p.add_argument("--model-dir", default="../models")
    p.add_argument("--output-csv", default="../benchmark_logs/results.csv")
    p.add_argument("--audio-out-dir", default="../benchmark_logs/audio")
    p.add_argument("--manifest", default="../manifest.csv", help="replay mode only")
    p.add_argument("--data-root", default="..", help="replay mode only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluator_plan:
        args.granularity = "slot"
        args.intents = "media_control,play_music"
        args.reps = 3

    intents = set(args.intents.split(",")) if args.intents else None
    prompts = load_prompts(args.phrase_list, granularity=args.granularity, intents=intents)
    classifier = Classifier(args.model_dir)

    rows = run_session(
        evaluator=args.evaluator,
        prompts=prompts,
        reps=args.reps,
        mode=args.mode,
        classifier=classifier,
        output_csv=args.output_csv,
        audio_out_dir=args.audio_out_dir,
        manifest_path=args.manifest,
        data_root=args.data_root,
    )

    summary = summarize(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
