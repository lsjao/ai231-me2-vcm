"""Evaluate a trained model against the class's shared, fixed master-dataset
`test` split (airimonda/ai231-me2-voice-commands) -- per the 2026-10-01 class
meeting, this fixed test set (not our own internal random val split, which
draws from our own thinner, unevenly-sourced data pool) is what's supposed to
decide which model to use. `holdout` is the separate, smaller set reserved
for the live Pi demo and is intentionally not touched here.

Reuses import_hf_master_dataset.py's resolve() so "what counts as which of
our labels" can't drift between the training-data importer and this eval.

Usage: python scripts/eval_hf_master_test.py [--model-dir ../models] [--limit N]
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from collections import Counter, defaultdict

import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from import_hf_master_dataset import HF_DATASET, resolve  # noqa: E402 -- single source of truth for the mapping

from vcm import audio  # noqa: E402
from vcm import labels as label_utils  # noqa: E402
from vcm.classifier import Classifier  # noqa: E402


def intent_rollup(y_true: list[str], y_pred: list[str]) -> str:
    stats: dict[str, list[int]] = {}
    for t, p in zip(y_true, y_pred):
        intent = label_utils.intent_of(t)
        s = stats.setdefault(intent, [0, 0, 0])  # n, command-correct, intent-correct
        s[0] += 1
        s[1] += int(t == p)
        s[2] += int(label_utils.intent_of(p) == intent)
    lines = ["per-intent rollup (master test set): intent  n  command_acc  intent_acc"]
    for intent in sorted(stats):
        n, cmd_ok, int_ok = stats[intent]
        lines.append(f"  {intent:<16} {n:>4}  {cmd_ok / n:.2f}  {int_ok / n:.2f}")
    return "\n".join(lines)


def reject_confusion(y_true: list[str], y_pred: list[str]) -> str:
    """False-accept (real negative predicted as a command) and false-reject
    (real command predicted as reject) rates -- the two numbers that matter
    most for a voice assistant's reject threshold."""
    tp = fn = fp = tn = 0
    fp_examples: Counter = Counter()
    fn_examples: Counter = Counter()
    for t, p in zip(y_true, y_pred):
        t_rej, p_rej = (t == "reject"), (p == "reject")
        if t_rej and p_rej:
            tp += 1
        elif t_rej and not p_rej:
            fn += 1  # false accept: real negative let through as a command
            fp_examples[p] += 1
        elif not t_rej and p_rej:
            fp += 1  # false reject: real command wrongly rejected
            fn_examples[t] += 1
        else:
            tn += 1
    n_reject = tp + fn
    n_command = fp + tn
    lines = [
        f"reject-class report: n_reject={n_reject} n_command={n_command}",
        f"  false-accept rate (reject -> command): {fn / n_reject:.2%}" if n_reject else "  (no reject rows)",
        f"  false-reject rate (command -> reject): {fp / n_command:.2%}" if n_command else "  (no command rows)",
    ]
    if fp_examples:
        lines.append(f"  reject mistaken for: {fp_examples.most_common(5)}")
    if fn_examples:
        lines.append(f"  commands wrongly rejected, most often: {fn_examples.most_common(5)}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=os.path.join(PROJECT_ROOT, "models"))
    p.add_argument("--split", default="test", choices=["test", "holdout"],
                   help="which fixed class split to evaluate against")
    p.add_argument("--limit", type=int, default=None, help="only evaluate the first N rows (debugging)")
    args = p.parse_args()

    from datasets import Audio, load_dataset

    clf = Classifier(args.model_dir)
    known_labels = set(clf.idx_to_label.values())
    print(f"model has {len(known_labels)} labels: {sorted(known_labels)}\n")

    print(f"Streaming {HF_DATASET} [{args.split}] ...")
    ds = load_dataset(HF_DATASET, split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    y_true: list[str] = []
    y_pred: list[str] = []
    skipped_unmapped: Counter = Counter()
    skipped_unknown_label: Counter = Counter()
    n = 0

    for row in ds:
        if args.limit and n >= args.limit:
            break
        mapped = resolve(row["command"], row["slot_value"])
        if mapped is None:
            skipped_unmapped[row["command"]] += 1
            continue
        our_intent, our_slot = mapped
        true_label = label_utils.command_label(our_intent, our_slot)
        if true_label not in known_labels:
            skipped_unknown_label[true_label] += 1
            continue

        wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != audio.TARGET_SR:
            wav = audio.resample(wav, sr, audio.TARGET_SR)
        wav = audio.fix_length(wav, audio.CLIP_SAMPLES)

        pred_label, _confidence, _latency_ms = clf.predict(wav)
        y_true.append(true_label)
        y_pred.append(pred_label)

        n += 1
        if n % 500 == 0:
            print(f"  ...{n} clips evaluated")

    print(f"\nEvaluated {n} clips from the master test set")
    print(f"  skipped (no home in our schema): {sum(skipped_unmapped.values())} -- {dict(skipped_unmapped)}")
    print(f"  skipped (our schema has no training data for this label): "
          f"{sum(skipped_unknown_label.values())} -- {dict(skipped_unknown_label)}")

    if n == 0:
        print("\nnothing evaluated, stopping")
        return

    overall = sum(t == p for t, p in zip(y_true, y_pred)) / n
    print(f"\noverall command accuracy: {overall:.2%} over {n} clips\n")
    print(intent_rollup(y_true, y_pred))
    print()
    print(reject_confusion(y_true, y_pred))


if __name__ == "__main__":
    main()
