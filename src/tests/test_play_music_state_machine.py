import random

import pytest

from vcm.play_music_state_machine import (
    EASTER_EGGS,
    NoRepeatQueue,
    PlaybackState,
    PlayMusicStateMachine,
)


def make_sm(seed=0):
    return PlayMusicStateMachine(rng=random.Random(seed))


def test_no_repeat_queue_never_repeats_immediately_across_many_wraparounds():
    q = NoRepeatQueue(["a", "b", "c"], rng=random.Random(1))
    seen = [q.current()]
    for _ in range(200):
        seen.append(q.advance())
    for prev, nxt in zip(seen, seen[1:]):
        assert prev != nxt


def test_no_repeat_queue_rejects_empty_playlist():
    with pytest.raises(ValueError):
        NoRepeatQueue([])


def test_play_playlist_sets_state_and_now_playing():
    sm = make_sm()
    resp = sm.handle_command("playlist_jazz")
    assert resp["ok"] is True
    assert sm.state == PlaybackState.PLAYING
    assert sm.now_playing.playlist == "playlist_jazz"
    assert sm.now_playing.track in sm.playlists["playlist_jazz"]


def test_pause_then_resume_keeps_same_track():
    sm = make_sm()
    sm.handle_command("playlist_chill")
    track = sm.now_playing.track
    pause_resp = sm.handle_command("pause")
    assert pause_resp["ok"] is True
    assert sm.state == PlaybackState.PAUSED

    resume_resp = sm.handle_command("play")
    assert resume_resp["ok"] is True
    assert sm.state == PlaybackState.PLAYING
    assert sm.now_playing.track == track


def test_pause_with_nothing_playing_fails_cleanly():
    sm = make_sm()
    resp = sm.handle_command("pause")
    assert resp["ok"] is False


def test_next_advances_without_repeating_current_track():
    sm = make_sm()
    sm.handle_command("playlist_workout")
    first = sm.now_playing.track
    resp = sm.handle_command("next")
    assert resp["ok"] is True
    assert sm.now_playing.track != first


def test_previous_returns_to_prior_track():
    sm = make_sm()
    sm.handle_command("playlist_focus")
    first = sm.now_playing.track
    sm.handle_command("next")
    second = sm.now_playing.track
    assert second != first

    back = sm.handle_command("previous")
    assert back["track"] == first


def test_next_without_anything_queued_fails_cleanly():
    sm = make_sm()
    resp = sm.handle_command("next")
    assert resp["ok"] is False


def test_stop_clears_now_playing_but_bare_play_resumes_same_queue_position():
    sm = make_sm()
    sm.handle_command("playlist_general")
    track = sm.now_playing.track

    stop_resp = sm.handle_command("stop")
    assert stop_resp["ok"] is True
    assert sm.state == PlaybackState.IDLE
    assert sm.now_playing.track is None

    resume_resp = sm.handle_command("play")
    assert resume_resp["ok"] is True
    assert resume_resp["track"] == track


def test_whats_playing_reports_nothing_when_idle():
    sm = make_sm()
    resp = sm.handle_command("whats_playing")
    assert resp["track"] is None


def test_whats_playing_reflects_paused_state():
    sm = make_sm()
    sm.handle_command("playlist_jazz")
    sm.handle_command("pause")
    resp = sm.handle_command("whats_playing")
    assert "Paused" in resp["message"]


def test_easter_eggs_play_immediately_and_are_flagged():
    sm = make_sm()
    for slot, title in EASTER_EGGS.items():
        resp = sm.handle_command(slot)
        assert resp["ok"] is True
        assert resp["track"] == title
        assert resp["is_easter_egg"] is True
        assert sm.state == PlaybackState.PLAYING


def test_next_after_easter_egg_resumes_last_playlist_context():
    sm = make_sm()
    sm.handle_command("playlist_jazz")
    sm.handle_command("easter_good_morning")
    assert sm.now_playing.track == "Good Morning"

    resp = sm.handle_command("next")
    assert resp["ok"] is True
    assert resp["playlist"] == "playlist_jazz"


def test_volume_up_and_down_clamp_to_0_100():
    sm = make_sm()
    sm.volume = 95
    sm.handle_command("volume_up")
    sm.handle_command("volume_up")
    assert sm.volume == 100

    sm.volume = 5
    sm.handle_command("volume_down")
    sm.handle_command("volume_down")
    assert sm.volume == 0


def test_duck_and_unduck_round_trip():
    sm = make_sm()
    sm.volume = 80
    sm.duck()
    assert sm.volume == 4  # round(80 * 0.05)
    sm.unduck()
    assert sm.volume == 80


def test_duck_is_idempotent_while_already_ducked():
    sm = make_sm()
    sm.volume = 80
    sm.duck()
    sm.volume = 1  # simulate external change while ducked, shouldn't matter
    sm.duck()  # should be a no-op since already ducked
    sm.unduck()
    assert sm.volume == 80


def test_unrecognized_command_fails_cleanly():
    sm = make_sm()
    resp = sm.handle_command("not_a_real_command")
    assert resp["ok"] is False
