"""Play-media playback state machine: playlist selection, transport controls,
volume, easter eggs, and a "what's playing" query.

Input contract: `handle_command` takes a fine-grained command string -- the
media_control/play_music *slot* values from phrase_list.csv (e.g. "play",
"next", "playlist_jazz", "whats_playing", "easter_good_morning"), not the
8-way *intent* the classifier in vcm.train currently predicts. The
classifier can currently only tell you an utterance was media_control or
play_music, not which of the 7+ commands within it -- mapping that down to
these commands is an unresolved slot-classification gap, not something this
module can paper over. See HANDOFF.md.

Track titles are stub placeholders. Real playback (actual audio files,
easter-egg MP3s) is out of scope here -- see README.md's "Still missing"
section for the Good Morning / No song files this module expects to exist
by title once wired to a real player.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from functools import partial

VOLUME_STEP = 10
DUCK_VOLUME_FRACTION = 0.05

DEFAULT_PLAYLISTS: dict[str, list[str]] = {
    "playlist_general": ["Track A", "Track B", "Track C", "Track D"],
    "playlist_jazz": ["Blue in Green", "So What", "Take Five", "Round Midnight"],
    "playlist_workout": ["Pump It", "Eye of the Tiger", "Stronger", "Titanium"],
    "playlist_chill": ["Weightless", "Clair de Lune", "River Flows in You", "Porcelain"],
    "playlist_focus": ["Nuvole Bianche", "Brain Food 1", "Brain Food 2", "Deep Focus"],
}

EASTER_EGGS: dict[str, str] = {
    "easter_good_morning": "Good Morning",
    "easter_stage_fright": "No",
}


class PlaybackState(Enum):
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass
class NowPlaying:
    playlist: str | None = None
    track: str | None = None
    is_easter_egg: bool = False


class NoRepeatQueue:
    """Shuffled cycle over a playlist's tracks with no immediate repeats,
    including across a wraparound (last track of one shuffle != first track
    of the next). advance()/rewind() move exactly one step at a time, so
    rewind() after advance() always returns the track actually played
    before -- there's no play-history stack, just this invariant.
    """

    def __init__(self, tracks: list[str], rng: random.Random | None = None):
        if not tracks:
            raise ValueError("playlist must have at least one track")
        self._tracks = list(tracks)
        self._rng = rng or random.Random()
        self._order: list[str] = []
        self._pos = 0
        self._reshuffle(avoid=None)

    def _reshuffle(self, avoid: str | None) -> None:
        order = list(self._tracks)
        self._rng.shuffle(order)
        if avoid is not None and len(order) > 1 and order[0] == avoid:
            swap_idx = self._rng.randrange(1, len(order))
            order[0], order[swap_idx] = order[swap_idx], order[0]
        self._order = order
        self._pos = 0

    def current(self) -> str:
        return self._order[self._pos]

    def advance(self) -> str:
        if self._pos + 1 >= len(self._order):
            self._reshuffle(avoid=self._order[self._pos])
        else:
            self._pos += 1
        return self._order[self._pos]

    def rewind(self) -> str:
        # no history before the start of the current shuffle; replaying the
        # same track here is an accepted edge case, not a bug
        if self._pos > 0:
            self._pos -= 1
        return self._order[self._pos]


class PlayMusicStateMachine:
    def __init__(
        self,
        playlists: dict[str, list[str]] | None = None,
        rng: random.Random | None = None,
        volume: int = 50,
    ):
        self.playlists = playlists or DEFAULT_PLAYLISTS
        self._rng = rng or random.Random()
        self._queues: dict[str, NoRepeatQueue] = {}

        self.state = PlaybackState.IDLE
        self.now_playing = NowPlaying()
        self.volume = volume
        self._pre_duck_volume: int | None = None
        self._last_playlist: str | None = None

        self._handlers = self._build_handlers()

    def _build_handlers(self) -> dict[str, object]:
        handlers: dict[str, object] = {
            "play": self._resume,
            "pause": self._pause,
            "stop": self._stop,
            "next": self._next,
            "previous": self._previous,
            "volume_up": self._volume_up,
            "volume_down": self._volume_down,
            "whats_playing": self._whats_playing,
        }
        for playlist in self.playlists:
            handlers[playlist] = partial(self._play_playlist, playlist)
        for egg_slot in EASTER_EGGS:
            handlers[egg_slot] = partial(self._play_easter_egg, egg_slot)
        return handlers

    def handle_command(self, command: str) -> dict:
        """Dispatch a play_media command. Always returns a dict with at
        least {"ok": bool, "message": str}."""
        handler = self._handlers.get(command)
        if handler is None:
            if command.startswith("playlist_"):
                name = command.removeprefix("playlist_").replace("_", " ")
                return {"ok": False, "message": f"The {name} playlist isn't available"}
            return {"ok": False, "message": f"unrecognized play_media command: {command!r}"}
        return handler()

    def _queue_for(self, playlist: str) -> NoRepeatQueue:
        if playlist not in self._queues:
            self._queues[playlist] = NoRepeatQueue(self.playlists[playlist], self._rng)
        return self._queues[playlist]

    # -- playlist selection / transport ---------------------------------

    def _play_playlist(self, playlist: str) -> dict:
        queue = self._queue_for(playlist)
        track = queue.current()
        self.state = PlaybackState.PLAYING
        self.now_playing = NowPlaying(playlist=playlist, track=track)
        self._last_playlist = playlist
        return {"ok": True, "message": f"Playing {track} from {playlist}", "track": track, "playlist": playlist}

    def _play_easter_egg(self, slot: str) -> dict:
        title = EASTER_EGGS[slot]
        self.state = PlaybackState.PLAYING
        self.now_playing = NowPlaying(playlist=None, track=title, is_easter_egg=True)
        return {"ok": True, "message": title, "track": title, "is_easter_egg": True}

    def _resume(self) -> dict:
        if self.state == PlaybackState.PLAYING:
            return {"ok": True, "message": f"Already playing {self.now_playing.track}", "track": self.now_playing.track}
        if self._last_playlist is None:
            return {"ok": False, "message": "nothing queued to play"}
        queue = self._queue_for(self._last_playlist)
        track = queue.current()
        self.state = PlaybackState.PLAYING
        self.now_playing = NowPlaying(playlist=self._last_playlist, track=track)
        return {"ok": True, "message": f"Playing {track}", "track": track, "playlist": self._last_playlist}

    def _pause(self) -> dict:
        if self.state != PlaybackState.PLAYING:
            return {"ok": False, "message": "nothing is playing to pause"}
        self.state = PlaybackState.PAUSED
        return {"ok": True, "message": f"Paused {self.now_playing.track}", "track": self.now_playing.track}

    def _stop(self) -> dict:
        if self.state == PlaybackState.IDLE:
            return {"ok": False, "message": "nothing is playing"}
        stopped_track = self.now_playing.track
        self.state = PlaybackState.IDLE
        self.now_playing = NowPlaying()
        # _last_playlist is kept so a bare "play" resumes the same queue
        # position rather than restarting from scratch
        return {"ok": True, "message": f"Stopped {stopped_track}", "track": stopped_track}

    def _next(self) -> dict:
        if self._last_playlist is None:
            return {"ok": False, "message": "nothing queued to skip"}
        queue = self._queue_for(self._last_playlist)
        track = queue.advance()
        self.state = PlaybackState.PLAYING
        self.now_playing = NowPlaying(playlist=self._last_playlist, track=track)
        return {"ok": True, "message": f"Skipping to {track}", "track": track, "playlist": self._last_playlist}

    def _previous(self) -> dict:
        if self._last_playlist is None:
            return {"ok": False, "message": "nothing queued to go back to"}
        queue = self._queue_for(self._last_playlist)
        track = queue.rewind()
        self.state = PlaybackState.PLAYING
        self.now_playing = NowPlaying(playlist=self._last_playlist, track=track)
        return {"ok": True, "message": f"Back to {track}", "track": track, "playlist": self._last_playlist}

    def _whats_playing(self) -> dict:
        if self.now_playing.track is None:
            return {"ok": True, "message": "Nothing is playing right now.", "track": None}
        verb = "Playing" if self.state == PlaybackState.PLAYING else "Paused on"
        return {
            "ok": True,
            "message": f"{verb} {self.now_playing.track}",
            "track": self.now_playing.track,
            "playlist": self.now_playing.playlist,
        }

    # -- volume -----------------------------------------------------------

    def _volume_up(self) -> dict:
        self.volume = min(100, self.volume + VOLUME_STEP)
        return {"ok": True, "message": f"Volume up to {self.volume}%", "volume": self.volume}

    def _volume_down(self) -> dict:
        self.volume = max(0, self.volume - VOLUME_STEP)
        return {"ok": True, "message": f"Volume down to {self.volume}%", "volume": self.volume}

    def duck(self) -> None:
        """Drop volume to ~5% while awaiting the next command during
        playback (classmate-sourced fix for music noise degrading
        recognition -- see HANDOFF.md). Call the instant a wake word is
        detected during playback; call unduck() when the command window
        closes. No wake-word detector exists yet to call this automatically
        -- this is only the volume-side half of the fix.
        """
        if self._pre_duck_volume is not None:
            return  # already ducked
        self._pre_duck_volume = self.volume
        self.volume = max(1, round(self.volume * DUCK_VOLUME_FRACTION))

    def unduck(self) -> None:
        if self._pre_duck_volume is None:
            return
        self.volume = self._pre_duck_volume
        self._pre_duck_volume = None
