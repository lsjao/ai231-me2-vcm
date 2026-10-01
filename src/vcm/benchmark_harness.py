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
    # plan: every real command across all intents, 3x each (see HANDOFF.md)
    python -m vcm.benchmark_harness --mode live --evaluator "Jane Doe" --evaluator-plan
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .paths import root_path
from . import audio, labels
from .classifier import Classifier

CSV_FIELDS = [
    "timestamp",
    "evaluator",
    "source",           # "live_mic" or "synthetic_replay"
    "attempt_index",
    "expected_intent",
    "slot",
    "expected_command",  # "intent/slot" (or "reject")
    "phrase",            # the "transcript" -- see module docstring
    "predicted_command",
    "predicted_intent",  # derived from predicted_command
    "confidence",
    "correct",           # predicted_command == expected_command
    "intent_correct",    # right intent, possibly wrong slot
    "latency_ms",
    "audio_filepath",
]


@dataclass
class Prompt:
    intent: str
    slot: str
    phrase: str

    @property
    def command(self) -> str:
        return labels.command_label(self.intent, self.slot)


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

    The evaluator plan ("each evaluator says every real command 3 times",
    per the assignment addendum) is `granularity="slot",
    intents=all_command_intents(phrase_list_path)` -- see HANDOFF.md and the
    CLI --evaluator-plan flag. `intents` is computed from the current
    `phrase_list.csv` rather than hardcoded, so it can't go stale again the
    way the original play_media-only version did after the Sep 28 scope
    expansion to all 7 intents (caught and fixed Sep 30).
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


def all_command_intents(phrase_list_path: str) -> set[str]:
    """Every intent with real phrases in `phrase_list.csv` except `reject`
    (not a "command" evaluators are asked to say correctly) -- computed
    fresh each call so `--evaluator-plan` covers whatever the current scope
    actually is, including `wake`."""
    with open(phrase_list_path, newline="", encoding="utf-8") as f:
        return {row["intent"] for row in csv.DictReader(f)} - {labels.REJECT}


class LiveMicSource:
    """Records `audio.CLIP_SECONDS` of audio from the default input device."""

    def __init__(self):
        import importlib.util

        if importlib.util.find_spec("sounddevice") is None:
            raise RuntimeError(
                "live mode needs the 'sounddevice' package (pip install sounddevice), "
                "plus a working mic -- not available on this dev machine, "
                "use --mode replay to smoke-test the harness instead"
            )

    def capture(self, prompt: Prompt) -> np.ndarray:
        from .capture import record_seconds

        input(f'Say: "{prompt.phrase}"  (press Enter, then speak)')
        return record_seconds(audio.CLIP_SECONDS)


class ReplaySource:
    """Cycles existing dataset WAVs per command, for harness smoke-testing
    without a mic. NOT a real evaluator -- see module docstring."""

    def __init__(self, manifest_path: str, data_root: str, seed: int = 0):
        rows = labels.read_manifest(manifest_path, data_root)
        self._by_command: dict[str, list[str]] = {}
        for r in rows:
            self._by_command.setdefault(r.label, []).append(r.filepath)
        self._rng = random.Random(seed)
        for paths in self._by_command.values():
            self._rng.shuffle(paths)
        self._cursor: dict[str, int] = {label: 0 for label in self._by_command}

    def available(self, prompt: Prompt) -> bool:
        return bool(self._by_command.get(prompt.command))

    def capture(self, prompt: Prompt) -> tuple[np.ndarray, str]:
        paths = self._by_command.get(prompt.command)
        if not paths:
            raise KeyError(f"no replay clips available for command {prompt.command!r}")
        i = self._cursor[prompt.command] % len(paths)
        self._cursor[prompt.command] += 1
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

                predicted_command, confidence, latency_ms = classifier.predict(wav)
                correct = predicted_command == prompt.command
                predicted_intent = labels.intent_of(predicted_command)
                intent_correct = predicted_intent == prompt.intent

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
                    "expected_command": prompt.command,
                    "phrase": prompt.phrase,
                    "predicted_command": predicted_command,
                    "predicted_intent": predicted_intent,
                    "confidence": round(confidence, 4),
                    "correct": correct,
                    "intent_correct": intent_correct,
                    "latency_ms": round(latency_ms, 2),
                    "audio_filepath": audio_filepath,
                }
                writer.writerow(row)
                f.flush()
                rows.append(row)
                print(
                    f"[{prompt.intent}/{prompt.slot} #{attempt_index}] "
                    f"said={prompt.phrase!r} pred={predicted_command} "
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

    intent_flags = [r["intent_correct"] for r in rows if "intent_correct" in r]
    intent_accuracy = sum(intent_flags) / len(intent_flags) if intent_flags else None

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
        "accuracy": accuracy,  # command-level: right intent AND slot
        "intent_accuracy": intent_accuracy,  # right intent, slot ignored
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
    p.add_argument("--phrase-list", default=root_path("phrase_list.csv"))
    p.add_argument("--granularity", choices=["intent", "slot", "phrase"], default="intent")
    p.add_argument(
        "--intents", default=None,
        help="comma-separated intent allowlist, e.g. media_control,play_music",
    )
    p.add_argument(
        "--evaluator-plan", action="store_true",
        help="shortcut for the addendum's plan: every real command (all intents "
        "except reject) 3x each, computed from --phrase-list so it stays correct "
        "as scope changes (equivalent to --granularity slot --reps 3 with "
        "--intents defaulted to every non-reject intent)",
    )
    p.add_argument("--model-dir", default=root_path("models"))
    p.add_argument("--output-csv", default=root_path("benchmark_logs", "results.csv"))
    p.add_argument("--audio-out-dir", default=root_path("benchmark_logs", "audio"))
    p.add_argument("--manifest", default=root_path("manifest.csv"), help="replay mode only")
    p.add_argument("--data-root", default=root_path(), help="replay mode only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluator_plan:
        args.granularity = "slot"
        args.reps = 3
        if args.intents is None:
            args.intents = ",".join(sorted(all_command_intents(args.phrase_list)))

    intents = set(args.intents.split(",")) if args.intents else None
    prompts = load_prompts(args.phrase_list, granularity=args.granularity, intents=intents)
    classifier = Classifier(args.model_dir)

    if args.mode == "replay":
        replay = ReplaySource(args.manifest, args.data_root)
        skipped = [p for p in prompts if not replay.available(p)]
        prompts = [p for p in prompts if replay.available(p)]
        for p in skipped:
            print(f"replay: no clips for {p.command!r}, skipping")

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
