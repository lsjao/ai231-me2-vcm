"""Import long recordings (phone voice memos, a friend's .m4a/.mp3/.wav) as
labeled takes.

You read a printed script into ONE audio file, pausing between phrases; this
splits the file into utterances with the same VAD the live pipeline uses and
labels them from the script by order. Two steps per block of phrases:

  1. Print the next block's script (also saved as data_real/pending_block.json):
       run.cmd vcm.import_recording --speaker josh --condition phone --distance near ^
           --intents media_control,play_music --reps-per-slot 16 --block-size 20
     Add --script-file script.txt to also write it to a file you can send to
     a friend. silence/noise prompts are left out (they produce no utterance to
     split) -- record those with record_dataset on the laptop.

  2. Record the block (a phone voice memo is fine), then import it:
       run.cmd vcm.import_recording --file "C:\\path\\to\\block.m4a"

Recording rules: ~1 s of silence at the very start; say each phrase once, then
pause ~1.5 s (silence, no throat-clearing); no pauses *inside* a phrase; keep
to the printed order. The import refuses (and saves nothing) if it hears a
different number of utterances than the script has, since one miscount would
mislabel every take after it -- it prints what it heard so you can see why.

Formats: wav/flac/ogg/mp3 via soundfile; m4a/aac/mp4 via PyAV (pip install av).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict

import numpy as np
import soundfile as sf

from . import audio
from .paths import root_path
from .record_dataset import (
    MIN_PEAK,
    NOISE_PHRASE,
    SILENCE_PHRASE,
    PhraseRow,
    build_tasks,
    condition_tag,
    existing_counts,
    load_phrase_rows,
    save_take,
)
from .vad import segment_utterances

PENDING_NAME = "pending_block.json"


def load_audio_16k(path: str) -> np.ndarray:
    """Any common audio file -> mono float32 at 16 kHz."""
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != audio.TARGET_SR:
            wav = audio.resample(wav, sr, audio.TARGET_SR)
        return wav.astype(np.float32)
    except Exception as first_error:
        try:
            import av
        except ImportError:
            raise RuntimeError(
                f"couldn't read {path} ({first_error}). For m4a/aac run `pip install av`, "
                "or export the recording as wav/mp3."
            ) from first_error
        return _load_with_av(av, path)


def _load_with_av(av, path: str) -> np.ndarray:
    container = av.open(path)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=audio.TARGET_SR)
    chunks: list[np.ndarray] = []
    try:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    finally:
        container.close()
    if not chunks:
        raise RuntimeError(f"no audio found in {path}")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


# -- step 1: plan + print a block ------------------------------------------

def spoken_rows(rows: list[PhraseRow]) -> list[PhraseRow]:
    return [r for r in rows if r.phrase not in (SILENCE_PHRASE, NOISE_PHRASE)]


def plan_block(
    rows: list[PhraseRow], out_root: str, speaker: str, tag: str,
    reps_per_slot: int, block_size: int, seed: int = 0,
) -> list[PhraseRow]:
    done = existing_counts(os.path.join(out_root, "manifest.csv"), speaker, tag)
    tasks = build_tasks(spoken_rows(rows), done, reps_per_slot, noise_reps=0, seed=seed)
    return [t.row for t in tasks[:block_size]]


def pending_path(out_root: str) -> str:
    return os.path.join(out_root, PENDING_NAME)


def write_pending(out_root: str, speaker: str, tag: str, block: list[PhraseRow]) -> None:
    os.makedirs(out_root, exist_ok=True)
    with open(pending_path(out_root), "w", encoding="utf-8") as f:
        json.dump({"speaker": speaker, "tag": tag, "rows": [asdict(r) for r in block]}, f, indent=2)


def read_pending(out_root: str) -> dict | None:
    path = pending_path(out_root)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["rows"] = [PhraseRow(**r) for r in data["rows"]]
    return data


def format_script(speaker: str, tag: str, block: list[PhraseRow]) -> str:
    lines = [
        f"Script: {len(block)} phrases ({speaker}, {tag})",
        "Record ONE audio file:",
        "  - stay silent for about 1 second at the start",
        "  - say each phrase once, then pause ~1.5 seconds (no talking or throat-clearing in the pause)",
        "  - no pauses inside a phrase; keep this order",
        "",
    ]
    lines += [f"{i:>3}. {r.phrase}" for i, r in enumerate(block, start=1)]
    return "\n".join(lines) + "\n"


# -- step 2: import ----------------------------------------------------------

def describe_heard(found: list[tuple[np.ndarray, float]], block: list[PhraseRow]) -> str:
    lines = [f"expected {len(block)} utterances, heard {len(found)}:"]
    for i in range(max(len(found), len(block))):
        heard = (
            f"heard {len(found[i][0]) / audio.TARGET_SR:.2f}s, ended at {found[i][1]:.1f}s"
            if i < len(found) else "-- nothing --"
        )
        want = f'"{block[i].phrase}"' if i < len(block) else "-- extra --"
        lines.append(f"  {i + 1:>3}. {want:<45} {heard}")
    return "\n".join(lines)


def import_block(path: str, out_root: str, say=print) -> int:
    """Returns a process exit code: 0 imported, 2 refused (nothing saved)."""
    pending = read_pending(out_root)
    if pending is None:
        say("no pending block -- run step 1 first (print the script) so I know the phrase order")
        return 2
    block: list[PhraseRow] = pending["rows"]
    speaker, tag = pending["speaker"], pending["tag"]

    found = segment_utterances(load_audio_16k(path))
    if len(found) != len(block):
        say(describe_heard(found, block))
        say(
            "\nNothing was saved. Usual causes: a pause inside a phrase (one phrase heard as two), "
            "no pause between phrases (two heard as one), a missed or repeated phrase, or a "
            "noisy start. Re-record the block (same script), or split the file into smaller blocks."
        )
        return 2

    too_quiet = [i + 1 for i, (clip, _) in enumerate(found) if float(np.max(np.abs(clip))) < MIN_PEAK]
    if too_quiet:
        say(f"utterances {too_quiet} are too quiet (peak < {MIN_PEAK}); nothing saved. Record closer / louder.")
        return 2

    for (clip, _), row in zip(found, block):
        save_take(clip, row, speaker, tag, out_root)
    os.remove(pending_path(out_root))
    say(f"imported {len(block)} takes for {speaker} ({tag}) into {out_root}. "
        "Re-run step 1 for the next block.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--file", help="step 2: audio file for the pending block")
    p.add_argument("--speaker", help="step 1: who is recording")
    p.add_argument("--condition", default="phone", help="step 1: e.g. phone, tv")
    p.add_argument("--distance", default="near")
    p.add_argument("--reps-per-slot", type=int, default=16)
    p.add_argument("--block-size", type=int, default=20)
    p.add_argument("--intents", default=None)
    p.add_argument("--slots", default=None)
    p.add_argument("--script-file", default=None, help="also write the script here (to send to someone)")
    p.add_argument("--phrase-list", default=root_path("phrase_list.csv"))
    p.add_argument("--out-root", default=root_path("data_real"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.file:
        return import_block(args.file, args.out_root)

    if not args.speaker:
        print("step 1 needs --speaker (or use --file to import a recording)")
        return 2
    rows = load_phrase_rows(args.phrase_list)
    if args.intents:
        allowed = set(args.intents.split(","))
        rows = [r for r in rows if r.intent in allowed]
    if args.slots:
        allowed_slots = set(args.slots.split(","))
        rows = [r for r in rows if r.slot in allowed_slots]

    tag = condition_tag(args.condition, args.distance)
    block = plan_block(rows, args.out_root, args.speaker, tag, args.reps_per_slot, args.block_size, args.seed)
    if not block:
        print("nothing left to record for these settings")
        return 0
    write_pending(args.out_root, args.speaker, tag, block)
    script = format_script(args.speaker, tag, block)
    print(script)
    if args.script_file:
        with open(args.script_file, "w", encoding="utf-8") as f:
            f.write(script)
        print(f"(script also written to {args.script_file})")
    counts = Counter(r.intent for r in block)
    print("block covers:", ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
    print('\nWhen recorded:  run.cmd vcm.import_recording --file "<your recording>"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
