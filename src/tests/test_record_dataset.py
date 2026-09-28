import csv
import os
from collections import Counter

import numpy as np

from vcm import audio
from vcm.record_dataset import (
    NOISE_PHRASE,
    SILENCE_PHRASE,
    PhraseRow,
    build_tasks,
    condition_tag,
    delete_take,
    existing_counts,
    prepare_clip,
    prompt_text,
    run_session,
    save_take,
    slugify,
    targets_per_phrase,
)

SR = audio.TARGET_SR


def make_take(speech=True, seconds=audio.CLIP_SECONDS, lead=0.5, dur=0.6, seed=0):
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    wav = (rng.standard_normal(n) * 10 ** (-65 / 20)).astype(np.float32)  # room tone
    if speech:
        s = int(SR * lead)
        t = np.arange(int(SR * dur)) / SR
        wav[s : s + len(t)] += (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    return wav


class FakeRecorder:
    def __init__(self, takes):
        self._takes = list(takes)
        self.calls = 0

    def record(self, seconds):
        take = self._takes[min(self.calls, len(self._takes) - 1)]
        self.calls += 1
        return take


def scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it, "q")


ROWS = [
    PhraseRow("media_control", "pause", "pause"),
    PhraseRow("media_control", "pause", "pause the music"),
    PhraseRow("media_control", "play", "play"),
    PhraseRow("reject", "silence", SILENCE_PHRASE),
]


# -- trim_to_speech ------------------------------------------------------

def test_trim_to_speech_cuts_leading_and_trailing_silence():
    wav = make_take(speech=True, lead=0.5, dur=0.6)
    trimmed = audio.trim_to_speech(wav)
    assert trimmed is not None
    # 0.6s of speech plus ~0.1s padding each side, well under the 3s window
    assert 0.6 * SR <= len(trimmed) <= 0.9 * SR


def test_trim_to_speech_returns_none_for_silence():
    assert audio.trim_to_speech(make_take(speech=False)) is None
    assert audio.trim_to_speech(np.zeros(SR, dtype=np.float32)) is None


# -- planning ------------------------------------------------------------

def test_slugify():
    assert slugify("What's playing?") == "what_s_playing"
    assert slugify(SILENCE_PHRASE) == "silence"


def test_targets_split_a_slots_budget_across_its_phrasings():
    targets = targets_per_phrase(ROWS, reps_per_slot=16, noise_reps=5)
    assert targets[ROWS[0]] == 8  # pause has 2 phrasings
    assert targets[ROWS[1]] == 8
    assert targets[ROWS[2]] == 16  # play has 1
    assert targets[ROWS[3]] == 5  # silence uses noise_reps


def test_build_tasks_makes_passes_of_one_take_per_unfinished_phrase():
    tasks = build_tasks(ROWS, Counter(), reps_per_slot=2, noise_reps=1)
    by_phrase = Counter(t.row.phrase for t in tasks)
    assert by_phrase["pause"] == 1
    assert by_phrase["pause the music"] == 1
    assert by_phrase["play"] == 2
    assert by_phrase[SILENCE_PHRASE] == 1
    # first pass covers every phrase once before any phrase repeats
    first_pass = [t for t in tasks if t.pass_index == 1]
    assert len(first_pass) == 4
    assert max(t.pass_index for t in tasks) == 2


def test_build_tasks_resumes_from_existing_counts():
    done = Counter({("media_control", "play", "play"): 2})
    tasks = build_tasks(ROWS, done, reps_per_slot=2, noise_reps=1)
    assert all(t.row.phrase != "play" for t in tasks)


# -- prepare_clip --------------------------------------------------------

def test_prompt_text_reminds_wake_phrase_to_be_said_as_one_flowing_unit():
    text = prompt_text(PhraseRow("wake", "kuya_jukebox", "kuya jukebox"))
    assert "kuya jukebox" in text and "no pause" in text


def test_prompt_text_ordinary_phrase_has_no_extra_hint():
    text = prompt_text(ROWS[0])
    assert text == 'say: "pause"'


def test_prepare_clip_trims_speech_and_rejects_silent_takes():
    speech_row = ROWS[0]
    assert prepare_clip(make_take(speech=True), speech_row) is not None
    assert prepare_clip(make_take(speech=False), speech_row) is None


def test_prepare_clip_keeps_full_window_for_silence_class():
    wav = make_take(speech=False)
    assert prepare_clip(wav, ROWS[3]) is wav


def test_prepare_clip_noise_class_needs_some_signal():
    noise_row = PhraseRow("reject", "noise", NOISE_PHRASE)
    assert prepare_clip(make_take(speech=False), noise_row) is None
    assert prepare_clip(make_take(speech=True), noise_row) is not None


# -- saving / session ----------------------------------------------------

def test_save_take_writes_wav_and_manifest_row(tmp_path):
    out = str(tmp_path)
    rec = save_take(make_take(), ROWS[0], "josh", "quiet_near", out)
    assert rec["filepath"] == "intent=media_control/slot=pause/pause__real_josh_quiet_near_001.wav"
    assert os.path.exists(os.path.join(out, rec["filepath"]))
    with open(tmp_path / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["source"] == "real"
    assert rows[0]["condition"] == "quiet_near"

    rec2 = save_take(make_take(), ROWS[0], "josh", "quiet_near", out)
    assert rec2["filepath"].endswith("_002.wav")  # no overwrite


def test_delete_take_removes_file_and_row(tmp_path):
    out = str(tmp_path)
    keep = save_take(make_take(), ROWS[0], "josh", "quiet_near", out)
    gone = save_take(make_take(), ROWS[1], "josh", "quiet_near", out)
    delete_take(gone, out)
    assert not os.path.exists(os.path.join(out, gone["filepath"]))
    with open(tmp_path / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["filepath"] for r in rows] == [keep["filepath"]]


def test_run_session_saves_retries_missed_takes_and_supports_redo(tmp_path):
    out = str(tmp_path)
    tag = condition_tag("quiet", "near")
    tasks = build_tasks(ROWS[:1], Counter(), reps_per_slot=2, noise_reps=1)  # 2 pause takes
    assert len(tasks) == 2

    # take 1 is silence (miss -> retried), then good; then redo it; then good again; then good
    recorder = FakeRecorder([make_take(False), make_take(True), make_take(True), make_take(True), make_take(True)])
    ask = scripted(["", "", "r", "", ""])
    messages = []
    saved = run_session(tasks, recorder, ask, out, "josh", tag, say=messages.append)

    assert any("didn't catch" in m for m in messages)
    assert any("redoing" in m for m in messages)
    counts = existing_counts(os.path.join(out, "manifest.csv"), "josh", tag)
    assert counts[("media_control", "pause", "pause")] == 2  # redo removed one, then re-recorded
    assert len(saved) == 2


def test_run_session_quit_stops_early(tmp_path):
    tasks = build_tasks(ROWS[:1], Counter(), reps_per_slot=4, noise_reps=1)
    recorder = FakeRecorder([make_take(True)])
    saved = run_session(tasks, recorder, scripted(["", "q"]), str(tmp_path), "josh", "quiet_near", say=lambda m: None)
    assert len(saved) == 1


def test_existing_counts_only_counts_matching_speaker_and_condition(tmp_path):
    out = str(tmp_path)
    save_take(make_take(), ROWS[0], "josh", "quiet_near", out)
    save_take(make_take(), ROWS[0], "josh", "tv_near", out)
    save_take(make_take(), ROWS[0], "sam", "quiet_near", out)
    counts = existing_counts(os.path.join(out, "manifest.csv"), "josh", "quiet_near")
    assert counts[("media_control", "pause", "pause")] == 1


def test_existing_counts_handles_missing_manifest(tmp_path):
    assert existing_counts(str(tmp_path / "nope.csv"), "josh", "quiet_near") == Counter()


# -- silent-room protection ------------------------------------------------

def room_noise(dbfs=-48.0, seconds=audio.CLIP_SECONDS, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SR * seconds)) * 10 ** (dbfs / 20)).astype(np.float32)


def test_room_noise_is_not_mistaken_for_speech():
    assert audio.speech_level_ok(room_noise()) is False
    assert prepare_clip(room_noise(), ROWS[0]) is None


def test_quiet_but_real_speech_still_passes():
    wav = make_take(speech=True)
    assert audio.speech_level_ok(wav) is True


def test_silent_recorder_cannot_loop_forever(tmp_path):
    tasks = build_tasks(ROWS[:1], Counter(), reps_per_slot=4, noise_reps=1)
    recorder = FakeRecorder([room_noise()])
    messages = []
    saved = run_session(tasks, recorder, scripted([""] * 100), str(tmp_path), "josh", "quiet_near",
                        say=messages.append)
    assert saved == []
    assert recorder.calls <= 6  # gave up instead of spinning
    assert any("mic muted" in m for m in messages)


def test_a_missed_prompt_is_skipped_after_three_tries_when_later_takes_work(tmp_path):
    tasks = build_tasks(ROWS[:1], Counter(), reps_per_slot=2, noise_reps=1)  # 2 prompts
    # 3 misses on the first prompt -> skipped; then the second prompt succeeds
    recorder = FakeRecorder([room_noise(), room_noise(), room_noise(), make_take(True)])
    messages = []
    saved = run_session(tasks, recorder, scripted([""] * 10), str(tmp_path), "josh", "quiet_near",
                        say=messages.append)
    assert len(saved) == 1
    assert any("skipping" in m for m in messages)


def test_a_brief_click_in_a_quiet_room_is_not_speech():
    wav = room_noise(-55.0)
    wav[8000:8160] += 0.3  # 10 ms thump
    assert audio.speech_level_ok(wav) is False


def test_a_short_word_worth_of_sound_is_speech():
    wav = room_noise(-55.0)
    t = np.arange(int(SR * 0.3)) / SR
    wav[8000 : 8000 + len(t)] += (0.2 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    assert audio.speech_level_ok(wav) is True
