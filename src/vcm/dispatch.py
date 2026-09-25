"""Command router: turns a classifier command label into an action.

    media_control/*, play_music/*  -> PlayMusicStateMachine.handle_command(slot)
    wake/*                         -> {"kind": "wake"} (the pipeline owns the wake window)
    reject                         -> no-op
    ask_time/none                  -> speaks the current time
    set_timer/<N>min               -> background timer, speaks when it expires
    set_temperature/<N>            -> simulated thermostat
    light_on_off/on|off, light_dim_color/brightness_<N> -> simulated lights

The non-media devices are simulated (state + spoken confirmation, no real
hardware) -- the assignment grades voice control, not smart-home drivers.
Every handler returns {"ok", "kind", "message", "speak"?, ...}; `speak` is
what should be said aloud, absent when the action should stay quiet (e.g. a
"next" skip shouldn't talk over the music).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Protocol

from . import labels
from .play_music_state_machine import PlayMusicStateMachine

MEDIA_INTENTS = {"media_control", "play_music"}
SPOKEN_MEDIA_COMMANDS = {"whats_playing"}


class Speaker(Protocol):
    def say(self, text: str) -> None: ...


class PrintSpeaker:
    def say(self, text: str) -> None:
        print(f"[speak] {text}")


class EspeakSpeaker:
    """On-device TTS via espeak-ng (no cloud), blocking until it finishes."""

    def __init__(self, voice: str = "en-us", words_per_minute: int = 165, exe: str = "espeak-ng"):
        path = shutil.which(exe)
        if path is None:
            raise RuntimeError(f"{exe} not found on PATH (sudo apt install espeak-ng)")
        self._cmd = [path, "-v", voice, "-s", str(words_per_minute)]

    def say(self, text: str) -> None:
        subprocess.run([*self._cmd, text], check=False)


def default_speaker() -> Speaker:
    try:
        return EspeakSpeaker()
    except RuntimeError:
        return PrintSpeaker()


@dataclass
class Devices:
    lights_on: bool = False
    brightness: int = 100
    temperature: int = 70


class TimerManager:
    """Fire-and-forget countdown timers. `seconds_per_minute` is only
    overridden in tests."""

    def __init__(self, on_expire: Callable[[int], None], seconds_per_minute: float = 60.0):
        self._on_expire = on_expire
        self._spm = seconds_per_minute
        self._timers: list[threading.Timer] = []

    def start(self, minutes: int) -> None:
        t = threading.Timer(minutes * self._spm, self._on_expire, args=(minutes,))
        t.daemon = True
        t.start()
        self._timers.append(t)

    def active(self) -> int:
        return sum(1 for t in self._timers if t.is_alive())

    def cancel_all(self) -> None:
        for t in self._timers:
            t.cancel()
        self._timers.clear()


class Dispatcher:
    def __init__(
        self,
        state_machine: PlayMusicStateMachine | None = None,
        speaker: Speaker | None = None,
        now: Callable[[], datetime] = datetime.now,
        seconds_per_minute: float = 60.0,
    ):
        self.state_machine = state_machine or PlayMusicStateMachine()
        self.speaker = speaker or default_speaker()
        self.devices = Devices()
        self._now = now
        self.timers = TimerManager(self._timer_expired, seconds_per_minute)

    def handle(self, label: str) -> dict:
        intent, slot = labels.intent_of(label), labels.slot_of(label)

        if intent == labels.REJECT:
            return {"ok": True, "kind": "reject", "message": "not a command"}
        if intent == "wake":
            return {"ok": True, "kind": "wake", "message": "wake word"}
        if intent in MEDIA_INTENTS:
            return self._media(slot)

        handler = {
            "ask_time": self._ask_time,
            "set_timer": self._set_timer,
            "set_temperature": self._set_temperature,
            "light_on_off": self._light_on_off,
            "light_dim_color": self._light_dim,
        }.get(intent)
        if handler is None:
            return {"ok": False, "kind": "unknown", "message": f"no handler for {label!r}"}
        return self._finish(handler(slot))

    # -- media ----------------------------------------------------------

    def _media(self, slot: str) -> dict:
        result = dict(self.state_machine.handle_command(slot))
        result["kind"] = "media"
        if slot in SPOKEN_MEDIA_COMMANDS or not result["ok"]:
            result["speak"] = result["message"]
        return self._finish(result)

    # -- simulated devices ------------------------------------------------

    def _ask_time(self, _slot: str) -> dict:
        t = self._now()
        hour = t.hour % 12 or 12
        text = f"It's {hour}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"
        return {"ok": True, "kind": "time", "message": text, "speak": text}

    def _set_timer(self, slot: str) -> dict:
        if not (slot.endswith("min") and slot[:-3].isdigit()):
            return {"ok": False, "kind": "timer", "message": f"bad timer slot {slot!r}"}
        minutes = int(slot[:-3])
        self.timers.start(minutes)
        unit = "minute" if minutes == 1 else "minutes"
        text = f"Timer set for {minutes} {unit}"
        return {"ok": True, "kind": "timer", "message": text, "speak": text, "minutes": minutes}

    def _timer_expired(self, minutes: int) -> None:
        unit = "minute" if minutes == 1 else "minutes"
        self.speaker.say(f"Your {minutes} {unit} timer is done")

    def _set_temperature(self, slot: str) -> dict:
        if not slot.isdigit():
            return {"ok": False, "kind": "temperature", "message": f"bad temperature slot {slot!r}"}
        self.devices.temperature = int(slot)
        text = f"Setting the temperature to {slot} degrees"
        return {"ok": True, "kind": "temperature", "message": text, "speak": text,
                "temperature": self.devices.temperature}

    def _light_on_off(self, slot: str) -> dict:
        if slot not in ("on", "off"):
            return {"ok": False, "kind": "light", "message": f"bad light slot {slot!r}"}
        self.devices.lights_on = slot == "on"
        text = f"Turning the lights {slot}"
        return {"ok": True, "kind": "light", "message": text, "speak": text,
                "lights_on": self.devices.lights_on}

    def _light_dim(self, slot: str) -> dict:
        prefix = "brightness_"
        if not (slot.startswith(prefix) and slot[len(prefix):].isdigit()):
            return {"ok": False, "kind": "light", "message": f"bad brightness slot {slot!r}"}
        self.devices.lights_on = True
        self.devices.brightness = int(slot[len(prefix):])
        text = f"Setting brightness to {self.devices.brightness} percent"
        return {"ok": True, "kind": "light", "message": text, "speak": text,
                "brightness": self.devices.brightness}

    def _finish(self, result: dict) -> dict:
        speak = result.get("speak")
        if speak:
            self.speaker.say(speak)
        return result
