"""One-time migration: import Snips SLU rows that import_snips_dataset.py's
keyword classifier couldn't map to any of our commands (3,472 of 5,886) as
reject-class negatives -- real, naturally-occurring smart-home-adjacent
speech that isn't one of our commands, filling the reject-data gap left by
Mark's out-of-scope categories being unavailable on disk (see
daily_log_report.md, 2026-09-29). Same gitignored external_data/ root as the
other two importers. Safe to re-run -- skips filepaths already in the
manifest.

Usage: python scripts/import_snips_reject.py [--dry-run] [--cap N]
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import os
import random
from collections import Counter

import soundfile as sf

from import_snips_dataset import SNIPS_PATH, classify, worker_id  # reuse, single source of truth

PROJECT_ROOT = r"C:\Users\Josh\Documents\MEng AI\AI 231\ME 2"
EXTERNAL_ROOT = os.path.join(PROJECT_ROOT, "external_data")
OUR_MANIFEST = os.path.join(EXTERNAL_ROOT, "manifest.csv")
MANIFEST_FIELDS = ["filepath", "intent", "slot", "phrase", "speaker", "condition", "source"]
DEFAULT_CAP = 300  # roughly in scale with our better-resourced real intents, not the full 3,472
SEED = 1337


def existing_filepaths() -> set[str]:
    if not os.path.exists(OUR_MANIFEST):
        return set()
    with open(OUR_MANIFEST, newline="", encoding="utf-8") as f:
        return {row["filepath"] for row in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cap", type=int, default=DEFAULT_CAP)
    args = p.parse_args()

    from datasets import Audio, load_from_disk

    ds = load_from_disk(SNIPS_PATH)["train"]
    ds = ds.cast_column("audio", Audio(decode=False))

    already = existing_filepaths()
    candidates: list[dict] = []
    matched = 0

    for i, row in enumerate(ds):
        if classify(row["text"]) is not None:
            matched += 1
            continue  # already imported as a real command by import_snips_dataset.py

        speaker = f"snips_{worker_id(row['worker'], i)}"
        rel_path = f"dataset/intent=reject/slot=near_domain/reject__near_domain__snips_{i}.wav"
        if rel_path in already:
            continue

        candidates.append({
            "filepath": rel_path,
            "intent": "reject",
            "slot": "near_domain",
            "phrase": row["text"],
            "speaker": speaker,
            "condition": "real_speech",
            "source": "snips_slu_reject",
            "_audio_bytes": row["audio"]["bytes"],
        })

    rng = random.Random(SEED)
    rng.shuffle(candidates)
    new_rows = candidates[: args.cap]

    print(f"Snips reject candidates: {len(candidates)} unmatched rows available "
          f"({matched} were already claimed by a command), capped to {len(new_rows)}")

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
          f"(intent=reject) into {EXTERNAL_ROOT}")


if __name__ == "__main__":
    main()
