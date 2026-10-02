"""One-time migration: pull the class master dataset's `synthetic_negatives`
config (added 2026-10-02, after our first import_hf_master_dataset.py pull)
into the same gitignored external_data_hf/ root.

These are purpose-built OUT_OF_SCOPE clips for training and measuring
misfires (non-command audio heard as a command) -- exactly the weakness our
own eval found (false-accept rate ~50% even after the class-weight fix).
Five kinds, equal shares: noise_only, babble, reversed, truncated,
near_silence (see the dataset README).

Only the `train` split (1,000 clips) is pulled -- its `test` split (250
clips) is built from the master test split's own audio and is part of the
class's fixed evaluation set, not ours to train on.

Usage: python scripts/import_hf_synthetic_negatives.py [--dry-run]
"""

from __future__ import annotations

import csv
import io
import os

import soundfile as sf

from import_hf_master_dataset import HF_DATASET, EXTERNAL_ROOT, OUR_MANIFEST, MANIFEST_FIELDS, existing_filepaths


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from datasets import Audio, load_dataset

    print(f"Streaming {HF_DATASET} [synthetic_negatives/train] ...")
    ds = load_dataset(HF_DATASET, "synthetic_negatives", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    already = existing_filepaths()
    new_rows: list[dict] = []
    from collections import Counter
    counts: Counter = Counter()

    for row in ds:
        neg_kind = row["neg_kind"]
        dst_name = f"reject__{neg_kind}__hf_{os.path.basename(row['file'])}"
        rel_path = f"dataset/intent=reject/slot={neg_kind}/{dst_name}"
        if rel_path in already:
            continue

        new_rows.append({
            "filepath": rel_path,
            "intent": "reject",
            "slot": neg_kind,
            "phrase": row["transcript"],
            "speaker": f"hf_{row['speaker_id']}",
            "condition": "synthetic_clean" if row["is_synthetic"] else "real_speech",
            "source": f"hf_master:synthetic_negative:{row['source']}",
            "_audio_bytes": row["audio"]["bytes"],
        })
        counts[neg_kind] += 1

    print(f"\n{len(new_rows)} new synthetic-negative rows to add")
    for k in sorted(counts):
        print(f"  reject/{k:14} {counts[k]:>4}")

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
