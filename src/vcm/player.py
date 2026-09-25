"""Audio playback driven by the play_media state machine.

`Player.sync(state_machine)` reconciles what *should* be audible (state,
track, volume) with what the backend is actually doing, so the pipeline just
calls it after every command / duck / ambient-volume change instead of
threading play/pause/stop calls through everywhere.

Music comes from local files (no streaming, per the hard constraints):

    music/<playlist_name>/*.mp3|wav|ogg|flac   e.g. music/playlist_jazz/
    music/easter/good_morning.mp3, music/easter/no.mp3

Any playlist with no files falls back to the state machine's stub titles,
played as a distinct sine tone per title, so next/pause/volume/ducking can
be tested on the speaker before any real music is sourced.

Playback goes through sounddevice with volume as a software gain, so
ducking is instantaneous and there's no dependency on ALSA mixers or
external players (mpg123/vlc). MP3 decode relies on libsndfile >= 1.1.
"""

from __future__ import annotations

import glob
import os
import re
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .play_music_state_machine import (
    DEFAULT_PLAYLISTS,
    PlaybackState,
    PlayMusicStateMachine,
)

AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac")
TONE_AMPLITUDE = 0.2


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def volume_to_gain(volume: int) -> float:
    """State machine volume (0-100) -> linear gain. Squared so the low end
    of the range is usable (a linear map is too loud too early)."""
    return (max(0, min(100, volume)) / 100.0) ** 2


class Library:
    def __init__(self, music_dir: str | None = None):
        self.music_dir = music_dir

    def _files(self, folder: str) -> list[str]:
        if not self.music_dir:
            return []
        found: list[str] = []
        for ext in AUDIO_EXTS:
            found.extend(glob.glob(os.path.join(self.music_dir, folder, f"*{ext}")))
        return sorted(found)

    def playlists(self) -> dict[str, list[str]]:
        """Titles per playlist: real file stems where files exist, else the
        state machine's stub titles."""
        out: dict[str, list[str]] = {}
        for name, stub_titles in DEFAULT_PLAYLISTS.items():
            files = self._files(name)
            out[name] = [Path(f).stem for f in files] if files else list(stub_titles)
        return out

    def path_for(self, playlist: str, title: str) -> str | None:
        for f in self._files(playlist):
            if Path(f).stem == title:
                return f
        return None

    def easter_path(self, title: str) -> str | None:
        for f in self._files("easter"):
            if Path(f).stem == slug(title):
                return f
        return None


class Source(Protocol):
    samplerate: int
    channels: int

    def read(self, frames: int) -> np.ndarray: ...
    def close(self) -> None: ...


class ToneSource:
    """Endless sine tone; pitch derived from the title so tracks differ."""

    channels = 1

    def __init__(self, title: str, samplerate: int = 22050):
        self.samplerate = samplerate
        self.freq = 220.0 + (sum(ord(c) for c in title) % 440)
        self._phase = 0

    def read(self, frames: int) -> np.ndarray:
        t = (self._phase + np.arange(frames)) / self.samplerate
        self._phase += frames
        return (TONE_AMPLITUDE * np.sin(2 * np.pi * self.freq * t)).astype(np.float32)[:, None]

    def close(self) -> None:
        pass


class FileSource:
    def __init__(self, path: str):
        import soundfile as sf

        self._f = sf.SoundFile(path)
        self.samplerate = self._f.samplerate
        self.channels = self._f.channels

    def read(self, frames: int) -> np.ndarray:
        return self._f.read(frames, dtype="float32", always_2d=True)

    def close(self) -> None:
        self._f.close()


class Backend(Protocol):
    def start(self, source: Source) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...
    def set_gain(self, gain: float) -> None: ...
    def poll_finished(self) -> bool: ...


class SoundDeviceBackend:
    def __init__(self, device: int | str | None = None, blocksize: int = 2048):
        try:
            import sounddevice as sd
        except ImportError as e:
            raise RuntimeError("playback needs `pip install sounddevice`") from e
        self._sd = sd
        self._device = device
        self._blocksize = blocksize
        self._lock = threading.Lock()
        self._stream = None
        self._source: Source | None = None
        self._paused = False
        self._gain = 0.25
        self._finished = False

    def start(self, source: Source) -> None:
        self.stop()
        with self._lock:
            self._source = source
            self._paused = False
            self._finished = False
        self._stream = self._sd.OutputStream(
            samplerate=source.samplerate,
            channels=source.channels,
            dtype="float32",
            blocksize=self._blocksize,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, _time, _status):
        with self._lock:
            source, paused, gain = self._source, self._paused, self._gain
        if source is None or paused:
            outdata.fill(0)
            return
        chunk = source.read(frames)
        n = len(chunk)
        outdata[:n] = chunk * gain
        if n < frames:
            outdata[n:] = 0
            with self._lock:
                self._finished = True
            raise self._sd.CallbackStop

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if self._source is not None:
                self._source.close()
            self._source = None
            self._paused = False

    def set_gain(self, gain: float) -> None:
        with self._lock:
            self._gain = gain

    def poll_finished(self) -> bool:
        with self._lock:
            done, self._finished = self._finished, False
        return done


class Player:
    def __init__(self, library: Library | None = None, backend: Backend | None = None):
        self.library = library or Library()
        self.backend = backend or SoundDeviceBackend()
        self._current: tuple | None = None  # (playlist, title, is_easter) being played
        self._track_done = False
        self._backend_paused = False

    def _source_for(self, playlist: str | None, title: str, is_easter: bool) -> Source:
        path = self.library.easter_path(title) if is_easter else self.library.path_for(playlist, title)
        if path is not None:
            try:
                return FileSource(path)
            except Exception as e:  # unreadable/unsupported file: fall back to a tone, keep going
                print(f"[player] can't decode {path} ({e}); playing a tone instead")
        return ToneSource(title)

    def sync(self, sm: PlayMusicStateMachine) -> None:
        self.backend.set_gain(volume_to_gain(sm.volume))

        if sm.state == PlaybackState.IDLE or sm.now_playing.track is None:
            if self._current is not None:
                self.backend.stop()
                self._current = None
            self._track_done = False
            self._backend_paused = False
            return

        np_ = sm.now_playing
        wanted = (np_.playlist, np_.track, np_.is_easter_egg)

        if sm.state == PlaybackState.PAUSED:
            if self._current is not None and not self._backend_paused:
                self.backend.pause()
                self._backend_paused = True
            return

        # PLAYING
        if wanted != self._current:
            self.backend.start(self._source_for(*wanted))
            self._current = wanted
            self._track_done = False
            self._backend_paused = False
        elif self._backend_paused:
            self.backend.resume()
            self._backend_paused = False

    def track_finished(self) -> bool:
        """True once, when the current track ran out on its own -- the
        pipeline should respond by dispatching media_control/next."""
        if self.backend.poll_finished():
            self._track_done = True
            return True
        return False

    def close(self) -> None:
        self.backend.stop()
        self._current = None
