"""One-time migration: merge the Snips SLU smart-lighting dataset into
external_data/ (same gitignored root as import_mark_dataset.py). Classifies
each row by keyword matching on its transcription (no pre-built intent/slot
labels exist in this dataset). Safe to re-run -- skips filepaths already in
the manifest.

Usage: python scripts/import_snips_dataset.py [--dry-run]
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import os
import random
import re
from collections import Counter, defaultdict

import soundfile as sf

SNIPS_PATH = r"C:\Users\Josh\Documents\MEng AI\AI 231\mark_dataset\snips_slu_data"
PROJECT_ROOT = r"C:\Users\Josh\Documents\MEng AI\AI 231\ME 2"
EXTERNAL_ROOT = os.path.join(PROJECT_ROOT, "external_data")
OUR_MANIFEST = os.path.join(EXTERNAL_ROOT, "manifest.csv")
MANIFEST_FIELDS = ["filepath", "intent", "slot", "phrase", "speaker", "condition", "source"]
DEFAULT_PER_SLOT_CAP = 60  # same cap as import_mark_dataset.py, for consistency
SEED = 1337

COLOR_WORDS = ["red", "blue", "green", "yellow", "white", "orange", "purple", "pink", "warm", "cool"]


def classify(text: str) -> tuple[str, str] | None:
    """Keyword classification per the spec -- order matters: "off" is
    checked before "on" so "turn off" doesn't get caught by the "on" check
    (turn-*on*), and color/brightness are checked before falling through."""
    t = text.lower()

    if re.search(r"\boff\b", t):
        return ("light_on_off", "off")
    if re.search(r"\bon\b", t):
        return ("light_on_off", "on")

    for color in COLOR_WORDS:
        if re.search(rf"\b{color}\b", t):
            return ("light_dim_color", f"color_{color}")

    if re.search(r"\bbright\w*\b|\bdim\w*\b|\d+\s*%|\bpercent\b|\d+", t):
        return ("light_dim_color", "brightness_other")

    return None


def existing_filepaths() -> set[str]:
    if not os.path.exists(OUR_MANIFEST):
        return set()
    with open(OUR_MANIFEST, newline="", encoding="utf-8") as f:
        return {row["filepath"] for row in csv.DictReader(f)}


def worker_id(worker_field: str, row_index: int) -> str:
    if worker_field:
        try:
            parsed = ast.literal_eval(worker_field)
            if isinstance(parsed, dict) and parsed.get("id"):
                return str(parsed["id"])[:8]  # full UUID is unwieldy in a filename
        except (ValueError, SyntaxError):
            pass
    return str(row_index)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--per-slot-cap", type=int, default=DEFAULT_PER_SLOT_CAP)
    args = p.parse_args()

    from datasets import Audio, load_from_disk

    ds = load_from_disk(SNIPS_PATH)["train"]
    ds = ds.cast_column("audio", Audio(decode=False))

    already = existing_filepaths()
    unmatched = 0
    candidates: dict[str, list[dict]] = defaultdict(list)

    for i, row in enumerate(ds):
        mapped = classify(row["text"])
        if mapped is None:
            unmatched += 1
            continue
        our_intent, our_slot = mapped

        speaker = f"snips_{worker_id(row['worker'], i)}"
        dst_name = f"{our_intent}__{our_slot}__snips_{i}.wav"
        rel_path = f"dataset/intent={our_intent}/slot={our_slot}/{dst_name}"
        if rel_path in already:
            continue

        candidates[f"{our_intent}/{our_slot}"].append({
            "filepath": rel_path,
            "intent": our_intent,
            "slot": our_slot,
            "phrase": row["text"],
            "speaker": speaker,
            "condition": "real_speech",
            "source": "snips_slu",
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

    print(f"Snips: {len(new_rows)} new rows to add (capped at {args.per_slot_cap}/slot), "
          f"{unmatched} rows didn't match any keyword rule (skipped)")
    print("\nPer our (intent/slot), new rows to add (of how many were available):")
    for k in sorted(counts):
        print(f"  {k:35} {counts[k]:>4} / {available[k]}")

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

    print(f"\nDone: wrote {len(new_rows)} files, appended {len(new_rows)} manifest rows into {EXTERNAL_ROOT}")


if __name__ == "__main__":
    main()
