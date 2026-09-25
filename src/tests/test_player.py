import random

import numpy as np
import soundfile as sf

from vcm.play_music_state_machine import DEFAULT_PLAYLISTS, PlayMusicStateMachine
from vcm.player import FileSource, Library, Player, ToneSource, slug, volume_to_gain


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.gain = None
        self.finished = False

    def start(self, source):
        self.calls.append(("start", type(source).__name__))

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))

    def stop(self):
        self.calls.append(("stop",))

    def set_gain(self, gain):
        self.gain = gain

    def poll_finished(self):
        done, self.finished = self.finished, False
        return done


def make(library=None):
    sm = PlayMusicStateMachine(rng=random.Random(0))
    backend = FakeBackend()
    return sm, backend, Player(library or Library(), backend)


def kinds(backend):
    return [c[0] for c in backend.calls]


def test_volume_to_gain_is_monotonic_and_bounded():
    assert volume_to_gain(0) == 0.0
    assert volume_to_gain(100) == 1.0
    assert volume_to_gain(30) < volume_to_gain(60)
    assert volume_to_gain(150) == 1.0 and volume_to_gain(-5) == 0.0


def test_idle_does_not_start_playback():
    sm, backend, player = make()
    player.sync(sm)
    assert backend.calls == []


def test_playing_starts_once_and_not_again_on_repeated_sync():
    sm, backend, player = make()
    sm.handle_command("playlist_jazz")
    player.sync(sm)
    player.sync(sm)
    assert backend.calls == [("start", "ToneSource")]  # no music files -> stub tone


def test_pause_then_resume_uses_backend_pause_and_resume_not_a_restart():
    sm, backend, player = make()
    sm.handle_command("playlist_jazz")
    player.sync(sm)
    sm.handle_command("pause")
    player.sync(sm)
    player.sync(sm)  # idempotent while paused
    sm.handle_command("play")
    player.sync(sm)
    assert kinds(backend) == ["start", "pause", "resume"]


def test_stop_stops_the_backend_and_next_play_restarts_it():
    sm, backend, player = make()
    sm.handle_command("playlist_chill")
    player.sync(sm)
    sm.handle_command("stop")
    player.sync(sm)
    sm.handle_command("play")
    player.sync(sm)
    assert kinds(backend) == ["start", "stop", "start"]


def test_next_starts_a_new_track():
    sm, backend, player = make()
    sm.handle_command("playlist_workout")
    player.sync(sm)
    sm.handle_command("next")
    player.sync(sm)
    assert kinds(backend) == ["start", "start"]


def test_easter_egg_plays_immediately():
    sm, backend, player = make()
    sm.handle_command("playlist_jazz")
    player.sync(sm)
    sm.handle_command("easter_good_morning")
    player.sync(sm)
    assert kinds(backend) == ["start", "start"]


def test_gain_follows_volume_including_ducking():
    sm, backend, player = make()
    sm.volume = 80
    player.sync(sm)
    normal = backend.gain
    sm.duck()
    player.sync(sm)
    assert backend.gain < normal * 0.05
    sm.unduck()
    player.sync(sm)
    assert backend.gain == normal


def test_track_finished_is_reported_once():
    _, backend, player = make()
    backend.finished = True
    assert player.track_finished() is True
    assert player.track_finished() is False


# -- library / sources -----------------------------------------------------

def write_wav(path, seconds=0.2, sr=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(sr * seconds), dtype="float32"), sr)


def test_library_falls_back_to_stub_titles_without_files(tmp_path):
    assert Library(str(tmp_path)).playlists() == DEFAULT_PLAYLISTS
    assert Library(None).playlists() == DEFAULT_PLAYLISTS


def test_library_uses_real_file_stems_for_playlists_that_have_files(tmp_path):
    write_wav(tmp_path / "playlist_jazz" / "take_five.wav")
    write_wav(tmp_path / "playlist_jazz" / "so_what.wav")
    lib = Library(str(tmp_path))
    playlists = lib.playlists()
    assert playlists["playlist_jazz"] == ["so_what", "take_five"]
    assert playlists["playlist_chill"] == DEFAULT_PLAYLISTS["playlist_chill"]  # no files -> stubs
    assert lib.path_for("playlist_jazz", "take_five").endswith("take_five.wav")
    assert lib.path_for("playlist_jazz", "missing") is None


def test_library_finds_easter_eggs_by_slugged_title(tmp_path):
    write_wav(tmp_path / "easter" / "good_morning.wav")
    lib = Library(str(tmp_path))
    assert lib.easter_path("Good Morning").endswith("good_morning.wav")
    assert lib.easter_path("No") is None
    assert slug("Good Morning") == "good_morning"


def test_player_plays_real_files_when_present(tmp_path):
    write_wav(tmp_path / "playlist_jazz" / "take_five.wav")
    lib = Library(str(tmp_path))
    sm = PlayMusicStateMachine(playlists=lib.playlists(), rng=random.Random(0))
    backend = FakeBackend()
    player = Player(lib, backend)
    sm.handle_command("playlist_jazz")
    player.sync(sm)
    assert backend.calls == [("start", "FileSource")]


def test_unreadable_file_falls_back_to_a_tone(tmp_path):
    bad = tmp_path / "playlist_jazz" / "broken.wav"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not audio")
    lib = Library(str(tmp_path))
    sm = PlayMusicStateMachine(playlists=lib.playlists(), rng=random.Random(0))
    backend = FakeBackend()
    player = Player(lib, backend)
    sm.handle_command("playlist_jazz")
    player.sync(sm)
    assert backend.calls == [("start", "ToneSource")]


def test_tone_source_is_continuous_and_pitch_differs_per_title():
    a, b = ToneSource("Take Five"), ToneSource("So What")
    first, second = a.read(100), a.read(100)
    assert first.shape == (100, 1) and first.dtype == np.float32
    whole = ToneSource("Take Five").read(200)
    assert np.allclose(np.concatenate([first, second]), whole, atol=1e-5)  # no phase jump
    assert a.freq != b.freq


def test_file_source_reads_blocks_until_exhausted(tmp_path):
    p = tmp_path / "t.wav"
    write_wav(p, seconds=0.1, sr=8000)  # 800 frames
    src = FileSource(str(p))
    assert src.samplerate == 8000 and src.channels == 1
    assert len(src.read(500)) == 500
    assert len(src.read(500)) == 300  # short read = end of track
    assert len(src.read(500)) == 0
    src.close()
