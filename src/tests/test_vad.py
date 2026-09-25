import numpy as np

from vcm import audio
from vcm.ambient_volume import FRAME_SAMPLES
from vcm.vad import Endpointer

SR = audio.TARGET_SR


def noise(seconds, dbfs=-60.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SR * seconds)) * 10 ** (dbfs / 20)).astype(np.float32)


def burst(seconds=0.6, freq=300.0, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def stream(*parts):
    wav = np.concatenate(parts)
    n = -(-len(wav) // FRAME_SAMPLES) * FRAME_SAMPLES
    return np.pad(wav, (0, n - len(wav)))


def run(wav):
    ep = Endpointer()
    out = []
    for i in range(0, len(wav), FRAME_SAMPLES):
        clip = ep.process(wav[i : i + FRAME_SAMPLES])
        if clip is not None:
            out.append(clip)
    tail = ep.flush()
    if tail is not None:
        out.append(tail)
    return out


def test_single_burst_becomes_one_clip_with_padding():
    clips = run(stream(noise(1.0), burst(0.6), noise(1.5, seed=1)))
    assert len(clips) == 1
    # 0.6s speech + ~0.1s context each side, nowhere near the 3s window
    assert 0.6 * SR < len(clips[0]) < 1.0 * SR


def test_two_bursts_separated_by_silence_are_two_clips():
    clips = run(stream(noise(1.0), burst(0.5), noise(1.2, seed=1), burst(0.5, 500), noise(1.2, seed=2)))
    assert len(clips) == 2


def test_pure_room_tone_produces_nothing():
    assert run(stream(noise(4.0))) == []


def test_a_single_click_is_ignored():
    click = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    click[:50] = 0.8
    assert run(stream(noise(1.0), click, noise(1.5, seed=1))) == []


def test_speech_over_a_noisy_room_is_still_detected():
    loud_room = noise(1.0, dbfs=-40)
    clips = run(stream(loud_room, burst(0.6, amp=0.3), noise(1.5, dbfs=-40, seed=1)))
    assert len(clips) == 1


def test_overlong_speech_is_capped_at_the_clip_window():
    clips = run(stream(noise(1.0), burst(5.0), noise(1.0, seed=1)))
    assert len(clips) >= 1
    assert all(len(c) <= audio.CLIP_SAMPLES for c in clips)


def test_flush_emits_an_utterance_cut_off_by_end_of_stream():
    clips = run(stream(noise(1.0), burst(0.6)))  # stream ends mid-hangover
    assert len(clips) == 1


def test_floor_does_not_creep_up_during_long_speech():
    ep = Endpointer()
    wav = stream(noise(1.0), burst(2.5))
    for i in range(0, len(wav), FRAME_SAMPLES):
        ep.process(wav[i : i + FRAME_SAMPLES])
    assert ep.floor_dbfs < -50  # still tracking the room, not the speech
