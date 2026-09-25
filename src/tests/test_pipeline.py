import random

import numpy as np

from test_vad import burst, noise, stream

from vcm.ambient_volume import FRAME_SAMPLES, AmbientAutoVolume
from vcm.dispatch import Dispatcher
from vcm.pipeline import Pipeline, PipelineConfig, iter_frames, run_wav
from vcm.play_music_state_machine import PlaybackState, PlayMusicStateMachine


class ScriptedClassifier:
    """Returns the next scripted (label, confidence) per utterance."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def predict(self, wav):
        label, conf = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return label, conf, 1.0


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


class FakePlayer:
    def __init__(self):
        self.syncs = 0
        self.finished = False

    def sync(self, sm):
        self.syncs += 1

    def track_finished(self):
        done, self.finished = self.finished, False
        return done


def build(script, playing=True, config=None, player=None, ambient=None):
    sm = PlayMusicStateMachine(rng=random.Random(0), volume=50)
    if playing:
        sm.handle_command("playlist_jazz")
    speaker = FakeSpeaker()
    dispatcher = Dispatcher(sm, speaker, seconds_per_minute=0.01)
    logs = []
    pipe = Pipeline(ScriptedClassifier(script), dispatcher, player, ambient,
                    config=config or PipelineConfig(), log=logs.append)
    return pipe, sm, speaker, logs


def session(*utterances, gap=1.0):
    parts = [noise(1.0)]
    for i, (dur, freq) in enumerate(utterances):
        parts += [burst(dur, freq), noise(gap, seed=i + 1)]
    return stream(*parts)


def types(events):
    return [e["type"] for e in events]


def test_wake_then_command_ducks_then_restores_and_executes():
    pipe, sm, _, _ = build([("wake/hey_pi", 0.9), ("media_control/pause", 0.9)])
    events = run_wav(pipe, session((0.5, 300), (0.5, 400)))
    assert types(events) == ["wake", "command"]
    assert sm.state == PlaybackState.PAUSED
    assert sm.volume == 50  # unducked


def test_volume_is_ducked_while_the_window_is_open():
    pipe, sm, _, _ = build([("wake/hey_pi", 0.9)])
    seen = []
    for frame in iter_frames(session((0.5, 300))):
        pipe.process_frame(frame)
        if pipe.window_open:
            seen.append(sm.volume)
    assert seen and max(seen) < 5


def test_command_without_wake_is_ignored():
    pipe, sm, _, _ = build([("media_control/pause", 0.9)])
    events = run_wav(pipe, session((0.5, 300)))
    assert types(events) == ["ignored_no_wake"]
    assert sm.state == PlaybackState.PLAYING


def test_no_wake_mode_executes_directly():
    pipe, sm, _, _ = build([("media_control/pause", 0.9)], config=PipelineConfig(require_wake=False))
    events = run_wav(pipe, session((0.5, 300)))
    assert types(events) == ["command"]
    assert sm.state == PlaybackState.PAUSED


def test_unused_window_times_out_and_restores_volume():
    pipe, sm, _, _ = build([("wake/hey_pi", 0.9)], config=PipelineConfig(wake_window_s=1.0))
    wav = stream(noise(1.0), burst(0.5), noise(4.0, seed=1))
    events = run_wav(pipe, wav)
    assert types(events) == ["wake", "window_timeout"]
    assert sm.volume == 50 and not pipe.window_open


def test_volume_up_inside_window_is_not_overwritten_by_unduck():
    pipe, sm, _, _ = build([("wake/hey_pi", 0.9), ("media_control/volume_up", 0.9)])
    run_wav(pipe, session((0.5, 300), (0.5, 400)))
    assert sm.volume == 60


def test_low_confidence_is_treated_as_reject():
    pipe, sm, _, _ = build([("wake/hey_pi", 0.2)])
    events = run_wav(pipe, session((0.5, 300)))
    assert types(events) == ["reject"]
    assert events[0]["heard"] == "wake/hey_pi" and events[0]["label"] == "reject"
    assert not pipe.window_open


def test_reject_class_does_nothing_even_inside_a_window():
    pipe, sm, _, _ = build([("wake/hey_pi", 0.9), ("reject", 0.9)])
    events = run_wav(pipe, session((0.5, 300), (0.5, 400)))
    assert types(events) == ["wake", "reject"]
    assert pipe.window_open  # still open, waiting for a real command


def test_non_media_command_speaks_through_the_dispatcher():
    pipe, _, speaker, _ = build([("wake/hey_pi", 0.9), ("set_temperature/70", 0.9)], playing=False)
    run_wav(pipe, session((0.5, 300), (0.5, 400)))
    assert speaker.said == ["Setting the temperature to 70 degrees"]


def test_player_is_synced_on_wake_and_command_and_auto_advances():
    player = FakePlayer()
    pipe, sm, _, _ = build([("wake/hey_pi", 0.9), ("media_control/next", 0.9)], player=player)
    events = run_wav(pipe, session((0.5, 300), (0.5, 400)))
    assert player.syncs >= 3  # duck, unduck, command
    track_before = sm.now_playing.track

    player.finished = True
    ev = pipe.process_frame(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    assert types(ev) == ["auto_next"]
    assert sm.now_playing.track != track_before
    assert "command" in types(events)


def test_auto_advance_does_nothing_when_not_playing():
    player = FakePlayer()
    pipe, _, _, _ = build([("reject", 0.9)], playing=False, player=player)
    player.finished = True
    assert pipe.process_frame(np.zeros(FRAME_SAMPLES, dtype=np.float32)) == []


def test_ambient_volume_holds_during_playback_but_adapts_when_idle():
    loud_room = stream(noise(8.0, dbfs=-20))

    playing, sm_p, _, _ = build([("reject", 0.9)], playing=True, ambient=AmbientAutoVolume(30))
    sm_p.volume = 30
    run_wav(playing, loud_room)
    assert sm_p.volume == 30  # suppressed while music plays (would chase its own loudness)

    idle, sm_i, _, _ = build([("reject", 0.9)], playing=False, ambient=AmbientAutoVolume(30))
    sm_i.volume = 30
    run_wav(idle, loud_room)
    assert sm_i.volume > 30


def test_manual_volume_command_suspends_ambient_adjustment():
    pipe, sm, _, _ = build(
        [("media_control/volume_down", 0.9)],
        playing=False,
        config=PipelineConfig(require_wake=False, manual_volume_hold_s=60.0),
        ambient=AmbientAutoVolume(50),
    )
    # loud room (-20 dBFS); a hot burst 16 dB above it is heard as speech and
    # the command lands before ambient's first 2s update
    wav = stream(noise(0.4, dbfs=-20), burst(0.4, amp=0.9), noise(8.0, dbfs=-20, seed=1))
    events = run_wav(pipe, wav)
    assert types(events) == ["command"]
    assert sm.volume == 40  # the user's change stands; the loud room didn't override it

    # control: same room with no command -> ambient adapts upward as usual
    ctl, sm_ctl, _, _ = build([("reject", 0.9)], playing=False, ambient=AmbientAutoVolume(50))
    run_wav(ctl, stream(noise(8.0, dbfs=-20)))
    assert sm_ctl.volume > 50
