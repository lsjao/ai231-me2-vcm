import random
import time
from datetime import datetime

from vcm.dispatch import Dispatcher
from vcm.play_music_state_machine import PlaybackState, PlayMusicStateMachine


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


def make(now=datetime(2026, 9, 26, 15, 5), spm=60.0):
    speaker = FakeSpeaker()
    sm = PlayMusicStateMachine(rng=random.Random(0))
    return Dispatcher(sm, speaker, now=lambda: now, seconds_per_minute=spm), speaker, sm


def test_media_commands_route_to_the_state_machine_by_slot():
    d, _, sm = make()
    r = d.handle("play_music/playlist_jazz")
    assert r["ok"] and r["kind"] == "media"
    assert sm.state == PlaybackState.PLAYING
    d.handle("media_control/pause")
    assert sm.state == PlaybackState.PAUSED


def test_transport_commands_speak_their_result():
    # Used to stay quiet on success (to avoid talking over real music) --
    # changed 2026-10-03: with no real music loaded for most of a demo, a
    # silent success is indistinguishable from a silent failure, and
    # evaluators have no other way to tell a transport command worked.
    d, speaker, _ = make()
    d.handle("play_music/playlist_chill")
    assert speaker.said[-1].startswith("Playing")
    d.handle("media_control/next")
    assert speaker.said[-1].startswith("Skipping to")
    d.handle("play_music/whats_playing")
    assert speaker.said[-1].startswith("Playing")


def test_failed_media_command_is_spoken():
    d, speaker, _ = make()
    r = d.handle("media_control/pause")  # nothing playing
    assert r["ok"] is False
    assert speaker.said == ["nothing is playing to pause"]


def test_volume_commands_change_state_machine_volume():
    d, _, sm = make()
    before = sm.volume
    d.handle("media_control/volume_up")
    assert sm.volume > before


def test_reject_and_wake_have_no_side_effects():
    d, speaker, sm = make()
    assert d.handle("reject")["kind"] == "reject"
    assert d.handle("wake/kuya_jukebox")["kind"] == "wake"
    assert speaker.said == [] and sm.state == PlaybackState.IDLE


def test_ask_time_speaks_12_hour_clock():
    d, speaker, _ = make(now=datetime(2026, 9, 26, 15, 5))
    d.handle("ask_time/none")
    assert speaker.said == ["It's 3:05 PM"]
    d2, speaker2, _ = make(now=datetime(2026, 9, 26, 0, 30))
    d2.handle("ask_time/none")
    assert speaker2.said == ["It's 12:30 AM"]


def test_set_timer_starts_a_timer_and_announces_when_it_expires():
    d, speaker, _ = make(spm=0.01)  # 1 "minute" = 10 ms
    r = d.handle("set_timer/1min")
    assert r["minutes"] == 1
    assert speaker.said[0] == "Timer set for 1 minute"
    time.sleep(0.3)
    assert speaker.said[-1] == "Your 1 minute timer is done"


def test_timer_can_be_cancelled_before_it_fires():
    d, speaker, _ = make(spm=5.0)
    d.handle("set_timer/5min")
    assert d.timers.active() == 1
    d.timers.cancel_all()
    assert d.timers.active() == 0
    assert all("done" not in s for s in speaker.said)


def test_temperature_and_light_handlers_update_simulated_devices():
    d, speaker, _ = make()
    d.handle("set_temperature/75")
    assert d.devices.temperature == 75
    d.handle("light_on_off/on")
    assert d.devices.lights_on is True
    d.handle("light_dim_color/brightness_25")
    assert d.devices.brightness == 25
    d.handle("light_on_off/off")
    assert d.devices.lights_on is False
    assert len(speaker.said) == 4


def test_light_dim_color_handles_color_and_brightness_other_slots():
    d, _, _ = make()
    r = d.handle("light_dim_color/color_red")
    assert r["ok"] and d.devices.color == "red" and d.devices.lights_on is True

    before = d.devices.brightness
    r = d.handle("light_dim_color/brightness_other")
    assert r["ok"] and d.devices.brightness == before  # kept, not guessed

    r = d.handle("light_dim_color/not_a_real_slot")
    assert not r["ok"]


def test_bad_slots_and_unknown_intents_fail_cleanly():
    d, _, _ = make()
    assert d.handle("set_timer/soon")["ok"] is False
    assert d.handle("set_temperature/warm")["ok"] is False
    assert d.handle("light_on_off/dim")["ok"] is False
    assert d.handle("teleport/now")["ok"] is False
