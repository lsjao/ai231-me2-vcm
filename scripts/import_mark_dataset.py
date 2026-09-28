"""One-time migration: merge Mark's OptionB spoken-command dataset into a
gitignored external_data/ root (same pattern as data_real/: its own
dataset/intent=X/slot=Y/ + manifest.csv, merged at train time via
--extra-data). Copies (never moves) source WAVs. Randomly subsamples to
PER_SLOT_CAP per (intent, slot) -- Mark's data has ~590/slot, far more than
needed to meaningfully help and slow enough to bloat storage and per-epoch
training time for no real benefit. Safe to re-run -- skips anything already
in the manifest.

Usage: python scripts/import_mark_dataset.py [--dry-run] [--per-slot-cap N]
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict

MARK_ROOT = r"C:\Users\Josh\Documents\MEng AI\AI 231\mark_dataset\MEX2\OptionB"
PROJECT_ROOT = r"C:\Users\Josh\Documents\MEng AI\AI 231\ME 2"
MARK_MANIFEST = os.path.join(MARK_ROOT, "manifest.csv")
EXTERNAL_ROOT = os.path.join(PROJECT_ROOT, "external_data")
OUR_MANIFEST = os.path.join(EXTERNAL_ROOT, "manifest.csv")
DEFAULT_PER_SLOT_CAP = 60
SEED = 1337

# intent with no slot values -> (our_intent, our_slot)
SIMPLE_MAP = {
    "LIGHT_ON": ("light_on_off", "on"),
    "LIGHT_OFF": ("light_on_off", "off"),
    "TIME": ("ask_time", "none"),
    "PLAY_MUSIC": ("media_control", "play"),
    "PAUSE": ("media_control", "pause"),
    "STOP": ("media_control", "stop"),
    "NEXT": ("media_control", "next"),
    "VOLUME_UP": ("media_control", "volume_up"),
    "VOLUME_DOWN": ("media_control", "volume_down"),
}

# (intent, slot_value) -> (our_intent, our_slot) -- current schema (brightness
# 20/60/100, Celsius temperature), matches Mark's own folder names exactly
SLOTTED_MAP = {
    ("BRIGHTNESS", "100 percent"): ("light_dim_color", "brightness_100"),
    ("BRIGHTNESS", "20 percent"): ("light_dim_color", "brightness_20"),
    ("BRIGHTNESS", "60 percent"): ("light_dim_color", "brightness_60"),
    ("COLOR", "red"): ("light_dim_color", "color_red"),
    ("COLOR", "blue"): ("light_dim_color", "color_blue"),
    ("COLOR", "green"): ("light_dim_color", "color_green"),
    ("COLOR", "yellow"): ("light_dim_color", "color_yellow"),
    ("TIMER", "1 minute"): ("set_timer", "1min"),
    ("TIMER", "10 seconds"): ("set_timer", "10sec"),
    ("TIMER", "30 seconds"): ("set_timer", "30sec"),
    ("TEMPERATURE", "18 degrees"): ("set_temperature", "18"),
    ("TEMPERATURE", "22 degrees"): ("set_temperature", "22"),
    ("TEMPERATURE", "26 degrees"): ("set_temperature", "26"),
}

# deliberate hard negatives, not discarded (covers REMINDER -> Mark's actual
# CREATE_REMINDER/LIST_REMINDERS labels)
OUT_OF_SCOPE_INTENTS = {"ALARM", "WEATHER", "CALL", "MESSAGE", "CREATE_REMINDER", "LIST_REMINDERS"}

MANIFEST_FIELDS = ["filepath", "intent", "slot", "phrase", "speaker", "condition", "source"]


def resolve(intent: str, slot_value: str) -> tuple[str, str] | None:
    if intent in SIMPLE_MAP:
        return SIMPLE_MAP[intent]
    key = (intent, slot_value)
    if key in SLOTTED_MAP:
        return SLOTTED_MAP[key]
    if intent in OUT_OF_SCOPE_INTENTS:
        return ("reject", "out_of_scope")
    return None  # genuinely out of scope (shouldn't happen given labels.json)


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

    already = existing_filepaths()
    candidates: dict[str, list[dict]] = defaultdict(list)
    skipped_unmapped: Counter = Counter()
    missing_source = 0

    with open(MARK_MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapped = resolve(row["intent"], row["slot_value"])
            if mapped is None:
                skipped_unmapped[row["intent"]] += 1
                continue
            our_intent, our_slot = mapped

            condition = "clean" if row["variant_id"] == "clean" else "noisy"
            speaker = f"mark_{row['speaker']}"
            dst_name = f"{our_intent}__{our_slot}__{speaker}_{row['phrase_id']}_{condition}.wav"
            rel_path = f"dataset/intent={our_intent}/slot={our_slot}/{dst_name}"

            if rel_path in already:
                continue

            src_path = os.path.join(MARK_ROOT, row["path"].replace("/", os.sep))
            if not os.path.exists(src_path):
                # Mark's README notes 844 files were excluded/flagged after his manifest
                # was written -- his manifest wasn't fully cleaned up to match. Real data
                # gap in the source, not something to paper over.
                missing_source += 1
                continue

            candidates[f"{our_intent}/{our_slot}"].append({
                "filepath": rel_path,
                "intent": our_intent,
                "slot": our_slot,
                "phrase": row["transcript"],
                "speaker": speaker,
                "condition": condition,
                "source": "mark_repo",
                "_src_path": src_path,
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

    print(f"Mark's dataset: {len(new_rows)} new rows to add (capped at {args.per_slot_cap}/slot), "
          f"{sum(skipped_unmapped.values())} truly unmapped rows skipped, "
          f"{missing_source} rows skipped (source file missing on disk)")
    if skipped_unmapped:
        print("  unmapped intents (unexpected, check labels.json):", dict(skipped_unmapped))
    print(f"\nPer our (intent/slot), new rows to add (of how many were available):")
    for k in sorted(counts):
        print(f"  {k:40} {counts[k]:>4} / {available[k]}")

    if args.dry_run:
        print("\n--dry-run: no files copied, manifest not touched")
        return

    for i, row in enumerate(new_rows):
        dst_full = os.path.join(EXTERNAL_ROOT, row["filepath"].replace("/", os.sep))
        os.makedirs(os.path.dirname(dst_full), exist_ok=True)
        try:
            shutil.copy2(row["_src_path"], dst_full)
        except OSError as e:
            print(f"COPY FAILED at row {i}: src={row['_src_path']!r} "
                  f"(exists={os.path.exists(row['_src_path'])}) dst={dst_full!r} "
                  f"(dir exists={os.path.isdir(os.path.dirname(dst_full))}) -- {e}")
            raise

    os.makedirs(EXTERNAL_ROOT, exist_ok=True)
    write_header = not os.path.exists(OUR_MANIFEST) or os.path.getsize(OUR_MANIFEST) == 0
    with open(OUR_MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            w.writeheader()
        for row in new_rows:
            w.writerow({k: row[k] for k in MANIFEST_FIELDS})

    print(f"\nDone: copied {len(new_rows)} files, appended {len(new_rows)} manifest rows "
          f"into {EXTERNAL_ROOT}")


if __name__ == "__main__":
    main()
