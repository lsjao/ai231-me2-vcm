"""One-time migration: pull martinnavs/ai231-fil-supplemental-data (shared by
a classmate on 2026-10-02) into the same gitignored external_data_hf/ root.

This is Filipino-accented synthetic voice data (zero-shot TTS cloned from
real Filipino reference speakers, e.g. "ilonggo1"), built specifically to
address the accent gap another classmate (Anthony Navarez) measured and
posted in the class group chat: real Filipino speech is only ~5% of the
main master dataset's train/test but ~45% of holdout, and models trained on
the public train split scored 17-27% on that one real Filipino speaker's
clips vs 88-96% on holdout's synthetic voices.

Same schema as the main master dataset (command/slot_value match Option B
exactly), so this reuses import_hf_master_dataset.py's resolve() directly --
single source of truth, can't drift from how we map everything else.

Usage: python scripts/import_hf_fil_supplemental.py [--dry-run] [--per-slot-cap N]
"""

from __future__ import annotations

import argparse
import io
import os
from collections import Counter, defaultdict
import random

import soundfile as sf

from import_hf_master_dataset import EXTERNAL_ROOT, MANIFEST_FIELDS, OUR_MANIFEST, existing_filepaths, resolve

HF_DATASET = "martinnavs/ai231-fil-supplemental-data"
DEFAULT_PER_SLOT_CAP = 60
SEED = 1337


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

        speaker = f"hf_fil_{row['speaker_id']}"
        dst_name = f"{our_intent}__{our_slot}__hf_fil_{os.path.basename(row['file'])}"
        rel_path = f"dataset/intent={our_intent}/slot={our_slot}/{dst_name}"
        if rel_path in already:
            continue

        candidates[f"{our_intent}/{our_slot}"].append({
            "filepath": rel_path,
            "intent": our_intent,
            "slot": our_slot,
            "phrase": row["transcript"],
            "speaker": speaker,
            "condition": "synthetic_clean",
            "source": f"hf_fil_supplemental:{row['source']}:{row.get('voice_accent','')}",
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

    print(f"\n{len(new_rows)} new Filipino-supplemental rows to add (capped at {args.per_slot_cap}/slot), "
          f"{sum(skipped_unmapped.values())} unmapped rows skipped")
    if skipped_unmapped:
        print("  unmapped commands:", dict(skipped_unmapped))
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
        import csv
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            w.writeheader()
        for row in new_rows:
            w.writerow({k: row[k] for k in MANIFEST_FIELDS})

    print(f"\nDone: wrote {len(new_rows)} files, appended {len(new_rows)} manifest rows into {EXTERNAL_ROOT}")


if __name__ == "__main__":
    main()
