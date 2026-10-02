"""One-time migration: pull the class's collated master dataset
(airimonda/ai231-me2-voice-commands on HuggingFace, Option B schema) into a
gitignored external_data_hf/ root (same pattern as external_data/ and
data_real/: its own dataset/intent=X/slot=Y/ + manifest.csv, merged at train
time via --extra-data).

Only the `train` split is pulled. `test` and `holdout` are the class's fixed,
speaker-disjoint evaluation set (see the dataset README) -- pulling them in
here would leak those speakers into our training pool and defeat the
point of a shared, fixed test set.

Reuses import_mark_dataset.py's SIMPLE_MAP / SLOTTED_MAP / OUT_OF_SCOPE_INTENTS
as the single source of truth for the Option B -> our-schema mapping, so this
importer can't silently drift from Mark's. This also finally fills the gap
import_snips_reject.py's docstring complains about: Mark's six hard-negative
intents (ALARM/WEATHER/CALL/MESSAGE/CREATE_REMINDER/LIST_REMINDERS) were
unavailable on disk before; the master dataset has real audio for them.

Usage: python scripts/import_hf_master_dataset.py [--dry-run] [--per-slot-cap N]
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
from collections import Counter, defaultdict

import soundfile as sf

from import_mark_dataset import SIMPLE_MAP, SLOTTED_MAP, OUT_OF_SCOPE_INTENTS

PROJECT_ROOT = r"C:\Users\Josh\Documents\MEng AI\AI 231\ME 2"
EXTERNAL_ROOT = os.path.join(PROJECT_ROOT, "external_data_hf")
OUR_MANIFEST = os.path.join(EXTERNAL_ROOT, "manifest.csv")
HF_DATASET = "airimonda/ai231-me2-voice-commands"
DEFAULT_PER_SLOT_CAP = 60
SEED = 1337
MANIFEST_FIELDS = ["filepath", "intent", "slot", "phrase", "speaker", "condition", "source"]

# COLOR values in the master dataset are capitalized ("Red"/"Blue"/"Green");
# Mark's SLOTTED_MAP keys are lowercase. Normalize both sides to match.
NORM_SLOTTED_MAP = {(cmd, val.lower()): target for (cmd, val), target in SLOTTED_MAP.items()}


def resolve(command: str, slot_value: str) -> tuple[str, str] | None:
    if command in SIMPLE_MAP:
        return SIMPLE_MAP[command]
    key = (command, (slot_value or "").strip().lower())
    if key in NORM_SLOTTED_MAP:
        return NORM_SLOTTED_MAP[key]
    if command in OUT_OF_SCOPE_INTENTS:
        return ("reject", "out_of_scope")
    if command == "OUT_OF_SCOPE":
        # real negative speech (noise / Filipino speech / near-miss / general
        # speech) -- same bucket Snips reject negatives already use
        return ("reject", "nearmiss")
    return None  # shouldn't happen -- the master schema is exactly Option B + OUT_OF_SCOPE


def existing_filepaths() -> set[str]:
    if not os.path.exists(OUR_MANIFEST):
        return set()
    with open(OUR_MANIFEST, newline="", encoding="utf-8") as f:
        return {row["filepath"] for row in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--per-slot-cap", type=int, default=DEFAULT_PER_SLOT_CAP)
    args = p.parse_args()

    from datasets import Audio, load_dataset

    print(f"Streaming {HF_DATASET} [train] ...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    already = existing_filepaths()
    candidates: dict[str, list[dict]] = defaultdict(list)
    skipped_unmapped: Counter = Counter()

    for row in ds:
        mapped = resolve(row["command"], row["slot_value"])
        if mapped is None:
            skipped_unmapped[row["command"]] += 1
            continue
        our_intent, our_slot = mapped

        speaker = f"hf_{row['speaker_id']}"
        condition = "synthetic_clean" if row["is_synthetic"] else "real_speech"
        dst_name = f"{our_intent}__{our_slot}__hf_{os.path.basename(row['file'])}"
        rel_path = f"dataset/intent={our_intent}/slot={our_slot}/{dst_name}"

        if rel_path in already:
            continue

        candidates[f"{our_intent}/{our_slot}"].append({
            "filepath": rel_path,
            "intent": our_intent,
            "slot": our_slot,
            "phrase": row["transcript"],
            "speaker": speaker,
            "condition": condition,
            "source": f"hf_master:{row['source']}",
            "_audio_bytes": row["audio"]["bytes"],
        })

    rng = random.Random(SEED)
    new_rows: list[dict] = []
    counts: Counter = Counter()
    available: Counter = Counter()
    for key, rows in candidates.items():
        available[key] = len(rows)
        rng.shuffle(rows)
        chosen = rows[: args.per_slot_cap]
        new_rows.extend(chosen)
        counts[key] = len(chosen)

    print(f"\nHF master dataset: {len(new_rows)} new rows to add (capped at {args.per_slot_cap}/slot), "
          f"{sum(skipped_unmapped.values())} truly unmapped rows skipped")
    if skipped_unmapped:
        print("  unmapped commands (unexpected, check the dataset schema):", dict(skipped_unmapped))
    print(f"\nPer our (intent/slot), new rows to add (of how many were available):")
    for k in sorted(counts):
        print(f"  {k:40} {counts[k]:>4} / {available[k]}")

    if args.dry_run:
        print("\n--dry-run: no files written, manifest not touched")
        return

    for row in new_rows:
        dst_full = os.path.join(EXTERNAL_ROOT, row["filepath"].replace("/", os.sep))
        os.makedirs(os.path.dirname(dst_full), exist_ok=True)
        wav, sr = sf.read(io.BytesIO(row["_audio_bytes"]), dtype="float32")
        sf.write(dst_full, wav, sr)

    os.makedirs(EXTERNAL_ROOT, exist_ok=True)
    write_header = not os.path.exists(OUR_MANIFEST) or os.path.getsize(OUR_MANIFEST) == 0
    with open(OUR_MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            w.writeheader()
        for row in new_rows:
            w.writerow({k: row[k] for k in MANIFEST_FIELDS})

    print(f"\nDone: wrote {len(new_rows)} files, appended {len(new_rows)} manifest rows "
          f"into {EXTERNAL_ROOT}")


if __name__ == "__main__":
    main()
