import csv
import os

import numpy as np
import pytest
import soundfile as sf
from test_vad import burst, noise

from vcm import audio
from vcm.import_recording import (
    describe_heard,
    format_script,
    import_block,
    load_audio_16k,
    pending_path,
    plan_block,
    read_pending,
    write_pending,
)
from vcm.record_dataset import NOISE_PHRASE, SILENCE_PHRASE, PhraseRow, save_take

SR = audio.TARGET_SR

ROWS = [
    PhraseRow("media_control", "pause", "pause"),
    PhraseRow("media_control", "pause", "pause the music"),
    PhraseRow("media_control", "next", "next"),
    PhraseRow("reject", "silence", SILENCE_PHRASE),
    PhraseRow("reject", "noise", NOISE_PHRASE),
]


def recording(n_utterances, amp=0.3, gap=1.5, seed=0):
    parts = [noise(1.0, seed=seed)]
    for i in range(n_utterances):
        parts += [burst(0.6, 250 + 60 * i, amp), noise(gap, seed=seed + i + 1)]
    return np.concatenate(parts)


def write_wav(path, wav, sr=SR):
    sf.write(str(path), wav, sr)
    return str(path)


def pending_block(out_root, rows):
    write_pending(out_root, "josh", "phone_near", rows)


def manifest_rows(out_root):
    with open(os.path.join(out_root, "manifest.csv"), newline="") as f:
        return list(csv.DictReader(f))


def test_plan_block_leaves_out_silence_and_noise_and_respects_block_size():
    block = plan_block(ROWS, "nonexistent_dir", "josh", "phone_near", reps_per_slot=4, block_size=5)
    assert len(block) == 5
    assert all(r.phrase not in (SILENCE_PHRASE, NOISE_PHRASE) for r in block)


def test_plan_block_resumes_after_takes_were_imported(tmp_path):
    out = str(tmp_path)
    first = plan_block(ROWS, out, "josh", "phone_near", reps_per_slot=2, block_size=100)
    for row in first:
        save_take(burst(0.5), row, "josh", "phone_near", out)
    assert plan_block(ROWS, out, "josh", "phone_near", reps_per_slot=2, block_size=100) == []


def test_pending_block_round_trips(tmp_path):
    out = str(tmp_path)
    pending_block(out, ROWS[:3])
    p = read_pending(out)
    assert p["speaker"] == "josh" and p["tag"] == "phone_near" and p["rows"] == ROWS[:3]
    assert read_pending(str(tmp_path / "nothing")) is None


def test_script_lists_numbered_phrases():
    text = format_script("josh", "phone_near", ROWS[:3])
    assert "  1. pause" in text and "  3. next" in text and "1.5 seconds" in text


def test_import_labels_takes_in_script_order(tmp_path):
    out = str(tmp_path / "data_real")
    pending_block(out, ROWS[:3])
    rec = write_wav(tmp_path / "block.wav", recording(3))

    assert import_block(rec, out, say=lambda m: None) == 0

    rows = manifest_rows(out)
    assert [(r["slot"], r["phrase"]) for r in rows] == [("pause", "pause"), ("pause", "pause the music"), ("next", "next")]
    assert all(r["speaker"] == "josh" and r["condition"] == "phone_near" and r["source"] == "real" for r in rows)
    assert all(os.path.exists(os.path.join(out, r["filepath"])) for r in rows)
    assert not os.path.exists(pending_path(out))  # consumed


def test_import_refuses_when_utterance_count_differs_and_saves_nothing(tmp_path):
    out = str(tmp_path / "data_real")
    pending_block(out, ROWS[:3])
    rec = write_wav(tmp_path / "block.wav", recording(2))  # one phrase missed
    messages = []

    assert import_block(rec, out, say=messages.append) == 2

    assert "expected 3 utterances, heard 2" in messages[0]
    assert not os.path.exists(os.path.join(out, "manifest.csv"))
    assert os.path.exists(pending_path(out))  # still there to retry


def test_import_refuses_when_there_are_extra_utterances(tmp_path):
    out = str(tmp_path / "data_real")
    pending_block(out, ROWS[:3])
    assert import_block(write_wav(tmp_path / "b.wav", recording(4)), out, say=lambda m: None) == 2
    assert not os.path.exists(os.path.join(out, "manifest.csv"))


def test_import_needs_a_pending_block(tmp_path):
    messages = []
    assert import_block(write_wav(tmp_path / "b.wav", recording(1)), str(tmp_path / "x"), say=messages.append) == 2
    assert "no pending block" in messages[0]


def test_import_refuses_takes_that_are_too_quiet(tmp_path):
    out = str(tmp_path / "data_real")
    pending_block(out, ROWS[:2])
    rec = write_wav(tmp_path / "quiet.wav", recording(2, amp=0.012))
    messages = []
    assert import_block(rec, out, say=messages.append) == 2
    assert any("too quiet" in m for m in messages)
    assert not os.path.exists(os.path.join(out, "manifest.csv"))


def test_describe_heard_lines_up_prompts_with_what_was_heard():
    from vcm.vad import segment_utterances

    found = segment_utterances(recording(2))
    text = describe_heard(found, ROWS[:3])
    assert '"pause the music"' in text and "-- nothing --" in text


def test_load_audio_resamples_to_16k(tmp_path):
    t = np.arange(44100) / 44100
    path = write_wav(tmp_path / "tone44.wav", (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sr=44100)
    wav = load_audio_16k(path)
    assert wav.dtype == np.float32 and abs(len(wav) - SR) < 50


def test_load_audio_stereo_is_downmixed(tmp_path):
    stereo = np.stack([burst(0.5), burst(0.5)], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, SR)
    assert load_audio_16k(str(path)).ndim == 1


def _encode_with_av(path, wav, codec, rate):
    av = pytest.importorskip("av")
    x = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    out = av.open(str(path), "w")
    try:
        stream = out.add_stream(codec, rate=rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(x[None, :], format="s16", layout="mono")
        frame.sample_rate = rate
        for pkt in stream.encode(frame):
            out.mux(pkt)
        for pkt in stream.encode(None):
            out.mux(pkt)
    finally:
        out.close()
    return str(path)


def test_m4a_recording_can_be_imported(tmp_path):
    out = str(tmp_path / "data_real")
    pending_block(out, ROWS[:3])
    rec = _encode_with_av(tmp_path / "block.m4a", recording(3), "aac", SR)
    assert import_block(rec, out, say=lambda m: None) == 0
    assert len(manifest_rows(out)) == 3


def test_mp3_recording_can_be_imported(tmp_path):
    out = str(tmp_path / "data_real")
    pending_block(out, ROWS[:3])
    try:
        rec = _encode_with_av(tmp_path / "block.mp3", recording(3), "mp3", SR)
    except Exception as e:  # this ffmpeg build has no mp3 encoder
        pytest.skip(f"can't make an mp3 fixture here: {e}")
    assert import_block(rec, out, say=lambda m: None) == 0
    assert len(manifest_rows(out)) == 3
