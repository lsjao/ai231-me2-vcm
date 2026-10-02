"""Command router: turns a classifier command label into an action.

    media_control/*, play_music/*  -> PlayMusicStateMachine.handle_command(slot)
    wake/*                         -> {"kind": "wake"} (the pipeline owns the wake window)
    reject                         -> no-op
    ask_time/none                  -> speaks the current time
    set_timer/<N>min               -> background timer, speaks when it expires
    set_temperature/<N>            -> simulated thermostat
    light_on_off/on|off,
    light_dim_color/brightness_<N>|brightness_other|color_<name> -> lights, real via GPIO
        on the Pi when the LED kit is wired up, else simulated (state + spoken
        confirmation only) -- see LightController below.
    weather/none, call/none, message/none, alarm/<time>,
    create_reminder/<task>, list_reminders/none -> predefined/canned
        responses (fixed TTS line, or alarm time / reminder list echoed
        back) -- no live weather API, no real telephony, per the
        assignment's "don't complicate things" guidance. Added 2026-10-02
        to cover the 10 required command categories; `reminders` state
        lives on Devices.

`set_temperature` stays simulated either way (no actuator planned for it).
Every handler returns {"ok", "kind", "message", "speak"?, ...}; `speak` is
what should be said aloud, absent when the action should stay quiet (e.g. a
"next" skip shouldn't talk over the music).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Protocol

from . import labels
from .play_music_state_machine import PlayMusicStateMachine

MEDIA_INTENTS = {"media_control", "play_music"}


class Speaker(Protocol):
    def say(self, text: str) -> None: ...


class PrintSpeaker:
    def say(self, text: str) -> None:
        print(f"[speak] {text}")


class EspeakSpeaker:
    """On-device TTS via espeak-ng (no cloud), blocking until it finishes.

    Piped through `aplay` targeting a named ALSA device explicitly, rather
    than letting espeak-ng pick its own output -- on the demo Pi, espeak-ng's
    own default audio backend exits 0 and produces genuinely nothing audible
    (confirmed: `raspi-config do_audio` forcing the system ALSA default to
    the headphone jack did NOT fix it, but explicit `aplay -D
    plughw:Headphones,0` does), so letting it pick silently is not safe to
    assume works. `alsa_device` names the device by its driver name
    ("Headphones", i.e. the bcm2835 jack), not a card *number* -- numbers can
    shift depending on what's plugged in at boot, the name shouldn't.
    """

    def __init__(
        self,
        voice: str = "en-us",
        words_per_minute: int = 165,
        amplitude: int = 150,  # espeak-ng default is 100; demo rooms are noisier than a dev desk
        exe: str = "espeak-ng",
        alsa_device: str = "plughw:Headphones,0",
    ):
        path = shutil.which(exe)
        if path is None:
            raise RuntimeError(f"{exe} not found on PATH (sudo apt install espeak-ng)")
        aplay = shutil.which("aplay")
        if aplay is None:
            raise RuntimeError("aplay not found on PATH (sudo apt install alsa-utils)")
        self._espeak_cmd = [path, "--stdout", "-v", voice, "-s", str(words_per_minute), "-a", str(amplitude)]
        self._aplay_cmd = [aplay, "-D", alsa_device, "-q"]

    def say(self, text: str) -> None:
        espeak = subprocess.Popen([*self._espeak_cmd, text], stdout=subprocess.PIPE)
        result = subprocess.run(self._aplay_cmd, stdin=espeak.stdout, capture_output=True)
        if espeak.stdout:
            espeak.stdout.close()
        espeak.wait()
        if result.returncode != 0:
            # Surface it rather than fail silently -- that silence is exactly
            # what cost real debugging time on 2026-10-03.
            print(f"[EspeakSpeaker] aplay failed (rc={result.returncode}): "
                  f"{result.stderr.decode(errors='replace').strip()}")


def default_speaker() -> Speaker:
    try:
        return EspeakSpeaker()
    except RuntimeError:
        return PrintSpeaker()


# Approximate RGB (0-1 per channel) for every color word the Snips import's
# keyword classifier can produce (see scripts/import_snips_dataset.py's
# COLOR_WORDS) plus our own red/blue/green/yellow. "warm"/"cool" have no
# literal RGB meaning -- approximated as warm-white / cool-white.
COLOR_MAP: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "orange": (1.0, 0.5, 0.0),
    "purple": (0.5, 0.0, 1.0),
    "pink": (1.0, 0.4, 0.7),
    "warm": (1.0, 0.6, 0.3),
    "cool": (0.7, 0.85, 1.0),
}


class LightController(Protocol):
    def set_state(self, on: bool, brightness: int, color: str | None) -> None: ...


class PrintLightController:
    """Hardware-absent fallback -- prints what the lights would do. Used
    automatically whenever gpiozero isn't installed or no Pi GPIO is
    present (e.g. the dev laptop, or the Pi before the LED kit is wired)."""

    def set_state(self, on: bool, brightness: int, color: str | None) -> None:
        state = "off" if not on else f"on, {brightness}% brightness, color={color or 'white'}"
        print(f"[lights] {state}")


class GPIOLightController:
    """Real actuation for the breadboard LED kit: one single-color LED for
    on/off, one RGB LED (PWM) for brightness/color -- covers both light
    intents with the one kit described in HANDOFF.md. BCM pin numbers below
    are provisional; update them to match the actual wiring once the kit is
    wired up (still not in hand as of this writing).
    """

    ON_OFF_PIN = 17
    RED_PIN = 22
    GREEN_PIN = 23
    BLUE_PIN = 24

    def __init__(self) -> None:
        from gpiozero import LED, PWMLED  # Pi-only; raises ImportError elsewhere

        self._on_off = LED(self.ON_OFF_PIN)
        self._red = PWMLED(self.RED_PIN)
        self._green = PWMLED(self.GREEN_PIN)
        self._blue = PWMLED(self.BLUE_PIN)

    def set_state(self, on: bool, brightness: int, color: str | None) -> None:
        if not on:
            self._on_off.off()
            self._red.off()
            self._green.off()
            self._blue.off()
            return
        self._on_off.on()
        r, g, b = COLOR_MAP.get(color, (1.0, 1.0, 1.0))
        scale = max(0, min(100, brightness)) / 100.0
        self._red.value = r * scale
        self._green.value = g * scale
        self._blue.value = b * scale


def default_light_controller() -> LightController:
    try:
        return GPIOLightController()
    except Exception:  # gpiozero missing, or no GPIO hardware (ImportError/RuntimeError/etc.)
        return PrintLightController()


@dataclass
class Devices:
    lights_on: bool = False
    brightness: int = 100
    color: str | None = None
    temperature: int = 70
    reminders: list[str] = field(default_factory=list)


class TimerManager:
    """Fire-and-forget countdown timers. `start` takes real-world seconds;
    `seconds_per_minute` is an acceleration factor (overridden in tests to
    make timers fire near-instantly) applied to any duration, not just
    literal minutes."""

    def __init__(self, on_expire: Callable[[int, str], None], seconds_per_minute: float = 60.0):
        self._on_expire = on_expire
        self._scale = seconds_per_minute / 60.0
        self._timers: list[threading.Timer] = []

    def start(self, seconds: int, label: str) -> None:
        t = threading.Timer(seconds * self._scale, self._on_expire, args=(seconds, label))
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
        lights: LightController | None = None,
        now: Callable[[], datetime] = datetime.now,
        seconds_per_minute: float = 60.0,
    ):
        self.state_machine = state_machine or PlayMusicStateMachine()
        self.speaker = speaker or default_speaker()
        self.lights = lights or default_light_controller()
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
            "weather": self._weather,
            "alarm": self._set_alarm,
            "create_reminder": self._create_reminder,
            "list_reminders": self._list_reminders,
            "call": self._call,
            "message": self._message,
        }.get(intent)
        if handler is None:
            return {"ok": False, "kind": "unknown", "message": f"no handler for {label!r}"}
        return self._finish(handler(slot))

    # -- media ----------------------------------------------------------

    def _media(self, slot: str) -> dict:
        # Used to stay quiet on successful transport commands (pause/stop/
        # next/volume) to avoid talking over real playing music -- but
        # volume_up/down have no failure case at all, so they could *never*
        # speak, and with no real music loaded for most of a demo, a silent
        # success looks identical to a silent failure. Found live on
        # 2026-10-03: evaluators have no other way to tell these worked.
        # Every other intent in this project always speaks its result;
        # media is no longer the exception.
        result = dict(self.state_machine.handle_command(slot))
        result["kind"] = "media"
        result["speak"] = result["message"]
        return self._finish(result)

    # -- simulated devices ------------------------------------------------

    def _ask_time(self, _slot: str) -> dict:
        t = self._now()
        hour = t.hour % 12 or 12
        text = f"It's {hour}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"
        return {"ok": True, "kind": "time", "message": text, "speak": text}

    def _set_timer(self, slot: str) -> dict:
        # "sec" checked first: a bug where only "min" was handled meant
        # set_timer/10sec and set_timer/30sec (2 of the 3 required Option B
        # timer values) silently failed at dispatch time -- found live on
        # 2026-10-03 during Pi testing, the classifier got it right and the
        # action still failed.
        if slot.endswith("sec") and slot[:-3].isdigit():
            seconds = int(slot[:-3])
            unit = "second" if seconds == 1 else "seconds"
            label = f"{seconds} {unit}"
            self.timers.start(seconds, label)
            text = f"Timer set for {label}"
            return {"ok": True, "kind": "timer", "message": text, "speak": text, "seconds": seconds}
        if slot.endswith("min") and slot[:-3].isdigit():
            minutes = int(slot[:-3])
            unit = "minute" if minutes == 1 else "minutes"
            label = f"{minutes} {unit}"
            self.timers.start(minutes * 60, label)
            text = f"Timer set for {label}"
            return {"ok": True, "kind": "timer", "message": text, "speak": text, "minutes": minutes}
        return {"ok": False, "kind": "timer", "message": f"bad timer slot {slot!r}"}

    def _timer_expired(self, _seconds: int, label: str) -> None:
        self.speaker.say(f"Your {label} timer is done")

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
        self.lights.set_state(self.devices.lights_on, self.devices.brightness, self.devices.color)
        text = f"Turning the lights {slot}"
        return {"ok": True, "kind": "light", "message": text, "speak": text,
                "lights_on": self.devices.lights_on}

    def _light_dim(self, slot: str) -> dict:
        brightness_prefix = "brightness_"
        color_prefix = "color_"
        if slot.startswith(brightness_prefix) and slot[len(brightness_prefix):].isdigit():
            self.devices.brightness = int(slot[len(brightness_prefix):])
            text = f"Setting brightness to {self.devices.brightness} percent"
        elif slot == "brightness_other":
            # No specific percentage was recognized (e.g. "dim it a bit") --
            # keep the current level rather than guessing a number.
            text = "Adjusting the brightness"
        elif slot.startswith(color_prefix):
            self.devices.color = slot[len(color_prefix):]
            text = f"Setting the light color to {self.devices.color}"
        else:
            return {"ok": False, "kind": "light", "message": f"bad light_dim_color slot {slot!r}"}
        self.devices.lights_on = True
        self.lights.set_state(self.devices.lights_on, self.devices.brightness, self.devices.color)
        return {"ok": True, "kind": "light", "message": text, "speak": text,
                "brightness": self.devices.brightness, "color": self.devices.color}

    # -- predefined-action commands (no live API / cloud / telephony, per
    # the assignment's "don't complicate things" guidance: each command
    # triggers a fixed, canned response rather than a real integration) ----

    def _weather(self, _slot: str) -> dict:
        text = "It's sunny and 28 degrees"
        return {"ok": True, "kind": "weather", "message": text, "speak": text}

    _ALARM_TIMES = {"6am": "6:00 AM", "8am": "8:00 AM", "9pm": "9:00 PM"}

    def _set_alarm(self, slot: str) -> dict:
        time_str = self._ALARM_TIMES.get(slot)
        if time_str is None:
            return {"ok": False, "kind": "alarm", "message": f"bad alarm slot {slot!r}"}
        text = f"Alarm set for {time_str}"
        return {"ok": True, "kind": "alarm", "message": text, "speak": text, "alarm_time": time_str}

    _REMINDER_TASKS = {"drink_water": "Drink water", "study": "Study", "exercise": "Exercise"}

    def _create_reminder(self, slot: str) -> dict:
        task = self._REMINDER_TASKS.get(slot)
        if task is None:
            return {"ok": False, "kind": "reminder", "message": f"bad reminder slot {slot!r}"}
        self.devices.reminders.append(task)
        text = f"Reminder added: {task}"
        return {"ok": True, "kind": "reminder", "message": text, "speak": text, "reminders": list(self.devices.reminders)}

    def _list_reminders(self, _slot: str) -> dict:
        if not self.devices.reminders:
            text = "You have no reminders"
        else:
            text = "Your reminders: " + ", ".join(self.devices.reminders)
        return {"ok": True, "kind": "reminder", "message": text, "speak": text, "reminders": list(self.devices.reminders)}

    def _call(self, _slot: str) -> dict:
        text = "Calling your emergency contact"
        return {"ok": True, "kind": "call", "message": text, "speak": text}

    def _message(self, _slot: str) -> dict:
        text = "Message sent"
        return {"ok": True, "kind": "message", "message": text, "speak": text}

    def _finish(self, result: dict) -> dict:
        speak = result.get("speak")
        if speak:
            self.speaker.say(speak)
        return result
