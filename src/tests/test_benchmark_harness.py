import csv

import numpy as np
import pytest
import soundfile as sf

from vcm.benchmark_harness import (
    Prompt,
    ReplaySource,
    load_prompts,
    run_session,
    summarize,
)


def test_load_prompts_intent_granularity_skips_silence_and_noise_placeholders(tmp_path):
    csv_path = tmp_path / "phrase_list.csv"
    csv_path.write_text(
        "intent,slot,phrase\n"
        "ask_time,none,what time is it\n"
        "ask_time,none,tell me the time\n"
        "reject,silence,__silence__\n"
        "reject,noise,__background_noise__\n"
        "reject,offvocab,i think i left my keys somewhere\n",
        encoding="utf-8",
    )
    prompts = load_prompts(str(csv_path), granularity="intent")
    assert len(prompts) == 2
    intents = {p.intent for p in prompts}
    assert intents == {"ask_time", "reject"}
    reject_prompt = next(p for p in prompts if p.intent == "reject")
    assert reject_prompt.slot == "offvocab"


def test_load_prompts_phrase_granularity_keeps_every_row_except_silence_noise(tmp_path):
    csv_path = tmp_path / "phrase_list.csv"
    csv_path.write_text(
        "intent,slot,phrase\n"
        "ask_time,none,what time is it\n"
        "ask_time,none,tell me the time\n"
        "reject,silence,__silence__\n",
        encoding="utf-8",
    )
    prompts = load_prompts(str(csv_path), granularity="phrase")
    assert len(prompts) == 2


def test_load_prompts_slot_granularity_gives_one_phrase_per_slot(tmp_path):
    csv_path = tmp_path / "phrase_list.csv"
    csv_path.write_text(
        "intent,slot,phrase\n"
        "media_control,pause,pause\n"
        "media_control,pause,pause the music\n"
        "media_control,next,next\n"
        "media_control,next,skip this song\n",
        encoding="utf-8",
    )
    prompts = load_prompts(str(csv_path), granularity="slot")
    assert len(prompts) == 2
    slots = {p.slot for p in prompts}
    assert slots == {"pause", "next"}


def test_load_prompts_intents_filter_restricts_to_given_intents(tmp_path):
    csv_path = tmp_path / "phrase_list.csv"
    csv_path.write_text(
        "intent,slot,phrase\n"
        "ask_time,none,what time is it\n"
        "media_control,play,play\n"
        "play_music,playlist_jazz,play some jazz\n",
        encoding="utf-8",
    )
    prompts = load_prompts(
        str(csv_path), granularity="intent", intents={"media_control", "play_music"}
    )
    intents = {p.intent for p in prompts}
    assert intents == {"media_control", "play_music"}


def test_load_prompts_rejects_unknown_granularity(tmp_path):
    csv_path = tmp_path / "phrase_list.csv"
    csv_path.write_text("intent,slot,phrase\nask_time,none,what time is it\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompts(str(csv_path), granularity="nonsense")


def _write_wav(path, seconds=1.0, sr=22050):
    n = int(seconds * sr)
    sf.write(str(path), np.zeros(n, dtype=np.float32), sr)


def _make_tiny_dataset(tmp_path):
    data_root = tmp_path
    (data_root / "dataset").mkdir()
    manifest_rows = []
    for intent, count in [("ask_time", 2), ("set_timer", 1)]:
        for i in range(count):
            fname = f"dataset/{intent}_{i}.wav"
            _write_wav(data_root / fname)
            manifest_rows.append((fname, intent))

    manifest_path = data_root / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "intent", "slot", "phrase", "speaker", "condition", "source"])
        for fname, intent in manifest_rows:
            w.writerow([fname, intent, "none", "phrase", "voice_us", "synthetic_clean", "tts"])

    return str(manifest_path), str(data_root)


def test_replay_source_cycles_without_exhausting(tmp_path):
    manifest_path, data_root = _make_tiny_dataset(tmp_path)
    source = ReplaySource(manifest_path, data_root, seed=0)
    prompt = Prompt(intent="ask_time", slot="none", phrase="what time is it")
    seen_paths = [source.capture(prompt)[1] for _ in range(5)]
    assert len(set(seen_paths)) == 2  # only 2 clips exist, cycles through them


def test_replay_source_raises_for_unknown_intent(tmp_path):
    manifest_path, data_root = _make_tiny_dataset(tmp_path)
    source = ReplaySource(manifest_path, data_root, seed=0)
    prompt = Prompt(intent="nonexistent_intent", slot="none", phrase="x")
    with pytest.raises(KeyError):
        source.capture(prompt)


class FakeClassifier:
    """Always predicts from a fixed cycle of (intent, confidence, latency_ms)
    tuples, ignoring the actual waveform -- isolates run_session's CSV/
    logging logic from real model behavior."""

    def __init__(self, predictions):
        self._predictions = list(predictions)
        self._i = 0

    def predict(self, wav):
        pred = self._predictions[self._i % len(self._predictions)]
        self._i += 1
        return pred


def test_run_session_writes_expected_csv_rows(tmp_path):
    manifest_path, data_root = _make_tiny_dataset(tmp_path)
    prompts = [Prompt(intent="ask_time", slot="none", phrase="what time is it")]
    classifier = FakeClassifier([("ask_time/none", 0.9, 1.0), ("set_timer/none", 0.4, 2.0)])
    output_csv = tmp_path / "results.csv"

    rows = run_session(
        evaluator="tester",
        prompts=prompts,
        reps=2,
        mode="replay",
        classifier=classifier,
        output_csv=str(output_csv),
        audio_out_dir=None,
        manifest_path=manifest_path,
        data_root=data_root,
    )

    assert len(rows) == 2
    assert rows[0]["predicted_intent"] == "ask_time"
    assert rows[0]["correct"] is True
    assert rows[1]["predicted_intent"] == "set_timer"
    assert rows[1]["correct"] is False
    assert rows[0]["source"] == "synthetic_replay"
    assert rows[0]["expected_command"] == "ask_time/none"
    assert rows[0]["predicted_command"] == "ask_time/none"
    assert rows[1]["predicted_command"] == "set_timer/none"
    assert rows[0]["intent_correct"] is True
    assert rows[1]["intent_correct"] is False

    with open(output_csv, newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 2
    assert written[0]["expected_intent"] == "ask_time"


def test_right_intent_wrong_slot_is_wrong_command_but_right_intent(tmp_path):
    manifest_path, data_root = _make_tiny_dataset(tmp_path)
    prompts = [Prompt(intent="ask_time", slot="none", phrase="what time is it")]
    rows = run_session(
        evaluator="tester",
        prompts=prompts,
        reps=1,
        mode="replay",
        classifier=FakeClassifier([("ask_time/other_slot", 0.8, 1.0)]),
        output_csv=str(tmp_path / "results.csv"),
        audio_out_dir=None,
        manifest_path=manifest_path,
        data_root=data_root,
    )
    assert rows[0]["correct"] is False
    assert rows[0]["intent_correct"] is True


def test_replay_source_reports_which_commands_have_clips(tmp_path):
    manifest_path, data_root = _make_tiny_dataset(tmp_path)
    source = ReplaySource(manifest_path, data_root)
    assert source.available(Prompt("ask_time", "none", "x"))
    assert not source.available(Prompt("ask_time", "some_other_slot", "x"))
    assert not source.available(Prompt("wake", "kuya_jukebox", "kuya jukebox"))


def test_summarize_reports_intent_accuracy_separately_from_command_accuracy():
    rows = [
        {**_row("a", "media_control", "next", 1, False), "intent_correct": True},
        {**_row("a", "media_control", "pause", 1, True), "intent_correct": True},
        {**_row("a", "set_timer", "5min", 1, False), "intent_correct": False},
    ]
    summary = summarize(rows)
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["intent_accuracy"] == pytest.approx(2 / 3)


def test_run_session_appends_without_duplicating_header(tmp_path):
    manifest_path, data_root = _make_tiny_dataset(tmp_path)
    prompts = [Prompt(intent="ask_time", slot="none", phrase="what time is it")]
    output_csv = tmp_path / "results.csv"

    run_session(
        evaluator="tester1",
        prompts=prompts,
        reps=1,
        mode="replay",
        classifier=FakeClassifier([("ask_time/none", 0.9, 1.0)]),
        output_csv=str(output_csv),
        audio_out_dir=None,
        manifest_path=manifest_path,
        data_root=data_root,
    )
    run_session(
        evaluator="tester2",
        prompts=prompts,
        reps=1,
        mode="replay",
        classifier=FakeClassifier([("ask_time/none", 0.9, 1.0)]),
        output_csv=str(output_csv),
        audio_out_dir=None,
        manifest_path=manifest_path,
        data_root=data_root,
    )

    with open(output_csv, encoding="utf-8") as f:
        lines = f.readlines()
    header_lines = [l for l in lines if l.startswith("timestamp,")]
    assert len(header_lines) == 1
    assert len(lines) == 3  # 1 header + 2 data rows


def _row(evaluator, expected_intent, slot, attempt_index, correct, confidence=0.5, latency_ms=1.0):
    return {
        "evaluator": evaluator,
        "expected_intent": expected_intent,
        "slot": slot,
        "attempt_index": attempt_index,
        "correct": correct,
        "confidence": confidence,
        "latency_ms": latency_ms,
    }


def test_summarize_computes_accuracy_and_per_intent_breakdown():
    rows = [
        _row("a", "ask_time", "none", 1, True, confidence=0.8, latency_ms=10.0),
        _row("a", "ask_time", "none", 2, False, confidence=0.4, latency_ms=20.0),
        _row("a", "set_timer", "1min", 1, True, confidence=0.9, latency_ms=5.0),
    ]
    summary = summarize(rows)
    assert summary["n_attempts"] == 3
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["mean_confidence"] == pytest.approx((0.8 + 0.4 + 0.9) / 3)
    assert summary["per_intent"]["ask_time"]["accuracy"] == pytest.approx(0.5)
    assert summary["per_intent"]["set_timer"]["accuracy"] == pytest.approx(1.0)


def test_summarize_of_empty_rows_is_empty_dict():
    assert summarize([]) == {}


def test_summarize_attempts_to_success_counts_first_correct_attempt():
    rows = [
        _row("jane", "media_control", "next", 1, False),
        _row("jane", "media_control", "next", 2, False),
        _row("jane", "media_control", "next", 3, True),
    ]
    summary = summarize(rows)
    assert summary["commands_tested"] == 1
    assert summary["commands_never_succeeded"] == 0
    assert summary["mean_attempts_to_success"] == pytest.approx(3.0)


def test_summarize_tracks_commands_that_never_succeed():
    rows = [
        _row("jane", "media_control", "next", 1, False),
        _row("jane", "media_control", "next", 2, False),
        _row("jane", "media_control", "pause", 1, True),
    ]
    summary = summarize(rows)
    assert summary["commands_tested"] == 2
    assert summary["commands_never_succeeded"] == 1
    assert summary["mean_attempts_to_success"] == pytest.approx(1.0)  # only the pause command


def test_summarize_attempts_to_success_is_none_when_nothing_ever_succeeds():
    rows = [_row("jane", "media_control", "next", 1, False)]
    summary = summarize(rows)
    assert summary["mean_attempts_to_success"] is None
    assert summary["commands_never_succeeded"] == 1


def test_summarize_groups_attempts_to_success_per_evaluator_separately():
    rows = [
        _row("jane", "media_control", "next", 1, True),   # jane succeeds immediately
        _row("bob", "media_control", "next", 1, False),
        _row("bob", "media_control", "next", 2, True),    # bob needs 2 attempts
    ]
    summary = summarize(rows)
    assert summary["commands_tested"] == 2  # (jane, next) and (bob, next) are distinct
    assert summary["mean_attempts_to_success"] == pytest.approx((1 + 2) / 2)
