"""Guided real-voice recording session.

Walks through phrase_list.csv in passes (each pass says every phrase once, in
shuffled order, so takes vary naturally instead of repeating one phrase 20x),
records each take, trims it to the speech, and saves it under a *separate*
data root (default `data_real/`) with its own manifest.csv. Real recordings
are kept out of the main dataset/ tree and out of git (bulky, and they're
someone's actual voice); train with `--extra-data data_real`.

Recording protocol this implements (see HANDOFF.md): most takes near the mic,
a few far; some quiet, some with TV/music/fan noise. Run it once per
(condition, distance) combination -- `--reps-per-slot` applies per run:

    python -m vcm.record_dataset --speaker josh --condition quiet --distance near \
        --reps-per-slot 16 --intents media_control,play_music
    python -m vcm.record_dataset --speaker josh --condition tv --distance near \
        --reps-per-slot 6
    python -m vcm.record_dataset --speaker josh --condition quiet --distance far \
        --reps-per-slot 3

Resume-safe: reps already in the manifest for the same (speaker, condition,
distance) are counted, so re-running continues where you stopped.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import soundfile as sf

from .paths import root_path
from . import audio

MANIFEST_FIELDS = ["filepath", "intent", "slot", "phrase", "speaker", "condition", "source"]

SILENCE_PHRASE = "__silence__"
NOISE_PHRASE = "__background_noise__"
NOISE_PROMPT_HINT = "make some background noise (TV, fan, typing, talking, music...)"

MIN_PEAK = 0.02  # takes quieter than this after trimming are assumed to be a miss
MAX_TAKE_RETRIES = 3          # misses on one prompt before moving on
MAX_CONSECUTIVE_MISSES = 6    # misses in a row before assuming the mic is muted / you left


@dataclass(frozen=True)
class PhraseRow:
    intent: str
    slot: str
    phrase: str


@dataclass(frozen=True)
class Task:
    row: PhraseRow
    pass_index: int  # 1-based


class Recorder(Protocol):
    def record(self, seconds: float) -> np.ndarray: ...


class SoundDeviceRecorder:
    def __init__(self, device: int | str | None = None):
        try:
            import sounddevice as sd
        except ImportError as e:
            raise RuntimeError("recording needs `pip install sounddevice`") from e
        self._sd = sd
        self._device = device

    def record(self, seconds: float) -> np.ndarray:
        n = int(audio.TARGET_SR * seconds)
        rec = self._sd.rec(
            n, samplerate=audio.TARGET_SR, channels=1, dtype="float32", device=self._device
        )
        self._sd.wait()
        return rec[:, 0]


def beep() -> None:
    """Short cue that recording starts now, so the first word isn't clipped."""
    import sounddevice as sd

    sr = 22050
    t = np.arange(int(0.12 * sr)) / sr
    tone = (0.2 * np.sin(2 * np.pi * 1000 * t) * np.hanning(len(t))).astype(np.float32)
    sd.play(tone, sr)
    sd.wait()


def load_phrase_rows(phrase_list_path: str) -> list[PhraseRow]:
    with open(phrase_list_path, newline="", encoding="utf-8") as f:
        return [PhraseRow(r["intent"], r["slot"], r["phrase"]) for r in csv.DictReader(f)]


def slugify(phrase: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")


def condition_tag(condition: str, distance: str) -> str:
    return f"{condition}_{distance}"


def read_manifest_rows(manifest_path: str) -> list[dict]:
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def existing_counts(manifest_path: str, speaker: str, tag: str) -> Counter:
    counts: Counter = Counter()
    for r in read_manifest_rows(manifest_path):
        if r["speaker"] == speaker and r["condition"] == tag:
            counts[(r["intent"], r["slot"], r["phrase"])] += 1
    return counts


def targets_per_phrase(
    rows: list[PhraseRow], reps_per_slot: int, noise_reps: int
) -> dict[PhraseRow, int]:
    """Spread each slot's rep budget across its phrasings (2-3 phrasings per
    slot per the protocol), rounding up so every phrasing is covered."""
    phrases_in_slot: Counter = Counter((r.intent, r.slot) for r in rows)
    targets = {}
    for r in rows:
        if r.phrase in (SILENCE_PHRASE, NOISE_PHRASE):
            targets[r] = noise_reps
        else:
            targets[r] = math.ceil(reps_per_slot / phrases_in_slot[(r.intent, r.slot)])
    return targets


def build_tasks(
    rows: list[PhraseRow],
    done: Counter,
    reps_per_slot: int,
    noise_reps: int,
    seed: int = 0,
) -> list[Task]:
    """Passes of one take per still-unfinished phrase, shuffled within a pass."""
    targets = targets_per_phrase(rows, reps_per_slot, noise_reps)
    remaining = {r: max(0, targets[r] - done[(r.intent, r.slot, r.phrase)]) for r in rows}
    rng = random.Random(seed)
    tasks: list[Task] = []
    pass_index = 0
    while any(remaining.values()):
        pass_index += 1
        batch = [r for r in rows if remaining[r] > 0]
        rng.shuffle(batch)
        for r in batch:
            tasks.append(Task(r, pass_index))
            remaining[r] -= 1
    return tasks


def prepare_clip(wav: np.ndarray, row: PhraseRow) -> np.ndarray | None:
    """Trim a raw take to its speech (or keep the full window for silence/
    noise). Returns None if the take should be re-done."""
    if row.phrase == SILENCE_PHRASE:
        return wav
    if row.phrase == NOISE_PHRASE:
        return wav if float(np.max(np.abs(wav))) >= MIN_PEAK else None
    if not audio.speech_level_ok(wav):
        return None
    trimmed = audio.trim_to_speech(wav)
    if trimmed is None or float(np.max(np.abs(trimmed))) < MIN_PEAK:
        return None
    return trimmed


def clip_relpath(row: PhraseRow, speaker: str, tag: str, out_root: str) -> str:
    folder = os.path.join(f"intent={row.intent}", f"slot={row.slot}")
    stem = f"{slugify(row.phrase)}__real_{slugify(speaker)}_{tag}"
    n = 1
    while True:
        rel = os.path.join(folder, f"{stem}_{n:03d}.wav")
        if not os.path.exists(os.path.join(out_root, rel)):
            return rel
        n += 1


def save_take(
    wav: np.ndarray, row: PhraseRow, speaker: str, tag: str, out_root: str
) -> dict:
    rel = clip_relpath(row, speaker, tag, out_root)
    full = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    sf.write(full, wav, audio.TARGET_SR)

    record = {
        "filepath": rel.replace(os.sep, "/"),
        "intent": row.intent,
        "slot": row.slot,
        "phrase": row.phrase,
        "speaker": speaker,
        "condition": tag,
        "source": "real",
    }
    manifest_path = os.path.join(out_root, "manifest.csv")
    write_header = not os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(record)
    return record


def delete_take(record: dict, out_root: str) -> None:
    """Undo a saved take (file + its manifest row)."""
    full = os.path.join(out_root, record["filepath"])
    if os.path.exists(full):
        os.remove(full)
    manifest_path = os.path.join(out_root, "manifest.csv")
    kept = [r for r in read_manifest_rows(manifest_path) if r["filepath"] != record["filepath"]]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(kept)


def prompt_text(row: PhraseRow) -> str:
    if row.phrase == SILENCE_PHRASE:
        return "stay SILENT (room tone only)"
    if row.phrase == NOISE_PHRASE:
        return NOISE_PROMPT_HINT
    return f'say: "{row.phrase}"'


def run_session(
    tasks: list[Task],
    recorder: Recorder,
    ask: Callable[[str], str],
    out_root: str,
    speaker: str,
    tag: str,
    say: Callable[[str], None] = print,
) -> list[dict]:
    """`ask(prompt)` returns "" (record), "r" (redo last take), "s" (skip),
    or "q" (quit). Injected so tests can script a session without a mic."""
    saved: list[dict] = []
    saved_at: list[int] = []  # task index each saved take belongs to, for redo
    i = 0
    misses = 0  # on the current prompt
    streak = 0  # consecutive, across prompts
    try:
        while i < len(tasks):
            task = tasks[i]
            header = (
                f"[pass {task.pass_index} | {i + 1}/{len(tasks)} | "
                f"{task.row.intent}/{task.row.slot}] {prompt_text(task.row)}"
            )
            choice = ask(header).strip().lower()

            if choice == "q":
                break
            if choice == "s":
                i += 1
                misses = 0
                continue
            if choice == "r":
                if saved:
                    delete_take(saved.pop(), out_root)
                    i = saved_at.pop()
                    say("  removed last take, redoing it")
                continue

            wav = recorder.record(audio.CLIP_SECONDS)
            clip = prepare_clip(wav, task.row)
            if clip is None:
                misses += 1
                streak += 1
                if streak >= MAX_CONSECUTIVE_MISSES:
                    say(f"  {streak} misses in a row -- is the mic muted, or did you step away? stopping "
                        "(re-run to resume)")
                    break
                if misses >= MAX_TAKE_RETRIES:
                    say("  still nothing after 3 tries, skipping this one")
                    i += 1
                    misses = 0
                else:
                    say("  didn't catch that (too quiet / no speech), try again")
                continue
            misses = 0
            streak = 0
            saved.append(save_take(clip, task.row, speaker, tag, out_root))
            saved_at.append(i)
            say(f"  saved ({len(clip) / audio.TARGET_SR:.2f}s, peak {np.max(np.abs(clip)):.2f})")
            i += 1
    except KeyboardInterrupt:
        say("\nstopped")
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--speaker", required=True)
    p.add_argument("--condition", default="quiet", help="quiet / tv / fan / music ...")
    p.add_argument("--distance", default="near", help="near (6-12in) / far (2-3ft)")
    p.add_argument("--reps-per-slot", type=int, default=16)
    p.add_argument("--noise-reps", type=int, default=15, help="takes for silence/noise classes")
    p.add_argument("--intents", default=None, help="comma-separated intent allowlist")
    p.add_argument("--slots", default=None, help="comma-separated slot allowlist")
    p.add_argument("--phrase-list", default=root_path("phrase_list.csv"))
    p.add_argument("--out-root", default=root_path("data_real"))
    p.add_argument("--device", default=None, help="sounddevice input device index/name")
    p.add_argument("--auto", action="store_true", help="no Enter needed: prompt, short pause, beep, then it records -- speak right after the beep")
    p.add_argument("--lead-in", type=float, default=1.0, help="seconds between the prompt and the beep in --auto")
    p.add_argument("--no-beep", action="store_true", help="--auto without the start-of-recording beep")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_phrase_rows(args.phrase_list)
    if args.intents:
        allowed = set(args.intents.split(","))
        rows = [r for r in rows if r.intent in allowed]
    if args.slots:
        allowed_slots = set(args.slots.split(","))
        rows = [r for r in rows if r.slot in allowed_slots]

    tag = condition_tag(args.condition, args.distance)
    manifest_path = os.path.join(args.out_root, "manifest.csv")
    done = existing_counts(manifest_path, args.speaker, tag)
    tasks = build_tasks(rows, done, args.reps_per_slot, args.noise_reps, args.seed)
    if not tasks:
        print("nothing left to record for these settings")
        return

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    recorder = SoundDeviceRecorder(device)

    if args.auto:
        def ask(prompt: str) -> str:
            print(prompt)
            time.sleep(args.lead_in)
            if not args.no_beep:
                beep()
                time.sleep(0.1)  # let the beep die out before the mic opens
            return ""
    else:
        def ask(prompt: str) -> str:
            return input(f"{prompt}\n  [Enter]=record  r=redo last  s=skip  q=quit > ")

    print(f"{len(tasks)} takes to go ({args.speaker}, {tag}). Ctrl+C or q to stop; re-run to resume.")
    saved = run_session(tasks, recorder, ask, args.out_root, args.speaker, tag)
    print(f"done. this session saved {len(saved)} takes into {args.out_root}/")


if __name__ == "__main__":
    main()
