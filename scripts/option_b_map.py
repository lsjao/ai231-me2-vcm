"""Maps our internal command labels to the class's Option B schema (the 19
agreed-upon commands, or OUT_OF_SCOPE) for the shared benchmark/validator.

Per the 2026-10-02 class chat (Ailene <-> D discussion): internal class
counts can differ -- we use 13 intents / ~47 fine labels, D used 31, Mark
~31/32 -- as long as each student can provide a mapping from their own
labels back to the 19 agreed intents for cross-class comparison. This is
that mapping, for us.

Built by inverting SIMPLE_MAP/SLOTTED_MAP from import_mark_dataset.py (the
same tables scripts/import_hf_master_dataset.py uses to go the other way),
so this can't silently drift from how we actually import training data.

Dev-side reporting tool only -- deliberately NOT in src/vcm/, which has to
stay deployable to the Pi without scripts/'s dev-only dependencies.

Usage: python scripts/option_b_map.py [--labels-json ../models/labels.json]
    Prints every label in the model's labels.json mapped to its Option B
    command (or "NO OPTION B EQUIVALENT" for our own creative additions),
    as a sanity check that the mapping covers everything the model predicts.
"""

from __future__ import annotations

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from import_mark_dataset import SIMPLE_MAP, SLOTTED_MAP  # noqa: E402 -- single source of truth

from vcm import labels as label_utils  # noqa: E402

# Reverse of SIMPLE_MAP: (our_intent, our_slot) -> Option B command.
_FROM_SIMPLE: dict[tuple[str, str], str] = {v: k for k, v in SIMPLE_MAP.items()}

# Reverse of SLOTTED_MAP: (our_intent, our_slot) -> Option B command (drops
# the exact slot_value text -- 19-level ("intent accuracy") reporting only
# needs which command, not which value; command-level (93) accuracy is a
# separate, finer question this mapping doesn't answer).
_FROM_SLOTTED: dict[tuple[str, str], str] = {v: cmd for (cmd, _val), v in SLOTTED_MAP.items()}

# Our own extensions beyond Option B's exact schema (3 colors, fixed
# brightness steps) still belong to the same command category for
# intent-level reporting, even though Option B's dataset never generated
# them under these exact values.
_EXTRA: dict[tuple[str, str], str] = {
    ("light_dim_color", "brightness_other"): "BRIGHTNESS",
    ("light_dim_color", "color_pink"): "COLOR",
    ("light_dim_color", "color_white"): "COLOR",
    ("light_dim_color", "color_cool"): "COLOR",
}

_TABLE: dict[tuple[str, str], str] = {**_FROM_SIMPLE, **_FROM_SLOTTED, **_EXTRA}


def to_option_b(label: str) -> str | None:
    """Our command label ("intent/slot", "reject", or "wake/...") -> the
    Option B command it corresponds to ("LIGHT_ON", "TEMPERATURE", ...,
    or "OUT_OF_SCOPE"), or None if it has no Option B equivalent: our own
    creative additions (play_music's specific playlists/easter eggs all
    collapse to PLAY_MUSIC; media_control/previous and the wake label don't
    exist in Option B at all).
    """
    if label == label_utils.REJECT:
        return "OUT_OF_SCOPE"
    intent, slot = label_utils.intent_of(label), label_utils.slot_of(label)
    if intent == "wake":
        return None
    if intent == "play_music":
        return "PLAY_MUSIC"
    if intent == "media_control" and slot == "previous":
        return None
    return _TABLE.get((intent, slot))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--labels-json", default=os.path.join(PROJECT_ROOT, "models", "labels.json"))
    args = p.parse_args()

    with open(args.labels_json) as f:
        idx_to_label = json.load(f)

    option_b_commands = {
        "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF", "PAUSE", "STOP", "NEXT",
        "VOLUME_UP", "VOLUME_DOWN", "CALL", "MESSAGE", "LIST_REMINDERS", "TIMER", "ALARM",
        "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
    }
    covered: set[str] = set()
    no_equivalent: list[str] = []

    for label in sorted(idx_to_label.values()):
        mapped = to_option_b(label)
        tag = mapped if mapped else "NO OPTION B EQUIVALENT"
        print(f"  {label:35} -> {tag}")
        if mapped and mapped != "OUT_OF_SCOPE":
            covered.add(mapped)
        elif mapped is None:
            no_equivalent.append(label)

    missing = option_b_commands - covered
    print(f"\n{len(covered)}/19 Option B commands covered by this model's label set")
    if missing:
        print(f"  MISSING: {sorted(missing)}")
    if no_equivalent:
        print(f"\n{len(no_equivalent)} of our labels have no Option B equivalent (expected -- our own extras):")
        print(f"  {no_equivalent}")


if __name__ == "__main__":
    main()
