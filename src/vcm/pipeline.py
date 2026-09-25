"""Live pipeline: mic frames -> endpointer -> classifier -> wake window ->
dispatcher -> player.

Interaction model (decided with the user): commands are only acted on while
a *wake window* is open. Saying the wake phrase opens a window for
`wake_window_s`, ducks the music to ~5% (classmate fix for music degrading
recognition) and pauses ambient auto-volume; the next command inside the
window closes it (restoring volume *before* the command runs, so a
"volume up" isn't overwritten by the unduck), and a window that expires
unused restores volume on its own.

Time is counted in processed frames, not wall-clock, so replaying a WAV file
is deterministic and testable.

Ambient auto-volume only adapts while nothing is playing: the mic hears the
speaker, so adapting during playback would chase the music's own loudness
upward. (Echo-aware adaptation is future work.) A manual volume command also
suspends it for `manual_volume_hold_s`.

Usage (from src/):
    python -m vcm.pipeline --source file --wav ../session.wav --no-audio
    python -m vcm.pipeline --source mic            # live, needs mic + speaker
"""

from __future__ import annotations

import argparse
import queue
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .paths import root_path
from . import audio, labels
from .ambient_volume import FRAME_MS, FRAME_SAMPLES, AmbientAutoVolume
from .dispatch import Dispatcher
from .play_music_state_machine import PlaybackState
from .vad import Endpointer, iter_frames

VOLUME_COMMANDS = {"media_control/volume_up", "media_control/volume_down"}


@dataclass
class PipelineConfig:
    min_confidence: float = 0.5
    wake_window_s: float = 5.0
    require_wake: bool = True
    manual_volume_hold_s: float = 60.0


class Pipeline:
    def __init__(
        self,
        classifier,
        dispatcher: Dispatcher,
        player=None,
        ambient: AmbientAutoVolume | None = None,
        endpointer: Endpointer | None = None,
        config: PipelineConfig | None = None,
        log: Callable[[str], None] = print,
    ):
        self.classifier = classifier
        self.dispatcher = dispatcher
        self.sm = dispatcher.state_machine
        self.player = player
        self.ambient = ambient
        self.endpointer = endpointer or Endpointer()
        self.config = config or PipelineConfig()
        self._log = log
        self.t = 0.0
        self._window_until: float | None = None
        self._ambient_hold_until = 0.0

    @property
    def window_open(self) -> bool:
        return self._window_until is not None

    # -- per-frame entry points ------------------------------------------

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        events: list[dict] = []
        self.t += FRAME_MS / 1000.0

        self._update_ambient(frame)

        if self._window_until is not None and self.t >= self._window_until:
            self._close_window()
            events.append(self._event("window_timeout"))

        utterance = self.endpointer.process(frame)
        if utterance is not None:
            events.append(self._handle_utterance(utterance))

        if self.player is not None and self.player.track_finished():
            if self.sm.state == PlaybackState.PLAYING:
                result = self.dispatcher.handle("media_control/next")
                self._sync()
                events.append(self._event("auto_next", result=result))

        for e in events:
            self._log(self._format(e))
        return events

    def finish(self) -> list[dict]:
        """End of stream: process an utterance still in progress."""
        utterance = self.endpointer.flush()
        if utterance is None:
            return []
        event = self._handle_utterance(utterance)
        self._log(self._format(event))
        return [event]

    # -- internals --------------------------------------------------------

    def _event(self, type_: str, **fields) -> dict:
        return {"type": type_, "t": round(self.t, 2), **fields}

    def _sync(self) -> None:
        if self.player is not None:
            self.player.sync(self.sm)

    def _update_ambient(self, frame: np.ndarray) -> None:
        if self.ambient is None:
            return
        suppressed = (
            self.sm.state == PlaybackState.PLAYING
            or self.window_open
            or self.t < self._ambient_hold_until
        )
        if suppressed:
            self.ambient.pause()
        else:
            self.ambient.resume()
        new_volume = self.ambient.update(frame)
        if new_volume is not None:
            self.sm.volume = new_volume
            self._sync()

    def _open_window(self) -> None:
        if not self.window_open:
            self.sm.duck()
        self._window_until = self.t + self.config.wake_window_s
        self._sync()

    def _close_window(self) -> None:
        if self.window_open:
            self.sm.unduck()
            self._window_until = None
            self._sync()

    def _handle_utterance(self, utterance: np.ndarray) -> dict:
        label, confidence, latency_ms = self.classifier.predict(utterance)
        heard = label
        if confidence < self.config.min_confidence:
            label = labels.REJECT
        info = {
            "heard": heard,
            "label": label,
            "confidence": round(confidence, 3),
            "latency_ms": round(latency_ms, 1),
        }

        intent = labels.intent_of(label)
        if intent == "wake":
            self._open_window()
            return self._event("wake", **info)
        if intent == labels.REJECT:
            return self._event("reject", **info)

        if self.config.require_wake and not self.window_open:
            return self._event("ignored_no_wake", **info)

        self._close_window()  # restore volume first, so the command sees the real volume
        result = self.dispatcher.handle(label)
        if label in VOLUME_COMMANDS:
            self._ambient_hold_until = self.t + self.config.manual_volume_hold_s
        self._sync()
        return self._event("command", result=result, **info)

    @staticmethod
    def _format(e: dict) -> str:
        parts = [f"[{e['t']:6.2f}s] {e['type']}"]
        if "label" in e:
            parts.append(f"{e['label']} ({e['confidence']:.2f}, {e['latency_ms']}ms)")
        if "result" in e:
            parts.append(f"-> {e['result'].get('message')}")
        return " ".join(parts)


# -- sources ----------------------------------------------------------------

def load_stream_wav(path: str) -> np.ndarray:
    """Whole file as mono float32 at TARGET_SR (unlike audio.load_waveform,
    which pads/trims to one clip)."""
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != audio.TARGET_SR:
        wav = audio.resample(wav, sr, audio.TARGET_SR)
    return wav


def run_wav(pipeline: Pipeline, wav: np.ndarray, realtime: bool = False) -> list[dict]:
    events: list[dict] = []
    for frame in iter_frames(wav):
        events.extend(pipeline.process_frame(frame))
        if realtime:
            time.sleep(FRAME_MS / 1000.0)
    events.extend(pipeline.finish())
    return events


def run_mic(pipeline: Pipeline, device: int | str | None = None) -> None:
    import sounddevice as sd

    frames: queue.Queue = queue.Queue()

    def callback(indata, _frames, _time, status):
        if status:
            print(f"[mic] {status}")
        frames.put(indata[:, 0].copy())

    print("listening... Ctrl+C to stop")
    with sd.InputStream(samplerate=audio.TARGET_SR, channels=1, dtype="float32",
                        blocksize=FRAME_SAMPLES, device=device, callback=callback):
        try:
            while True:
                events = pipeline.process_frame(frames.get())
                spoke = any(e["type"] == "command" and e["result"].get("speak") for e in events)
                if spoke:  # TTS blocked us; drop the backlog (it's mostly our own voice)
                    while not frames.empty():
                        frames.get_nowait()
        except KeyboardInterrupt:
            print("\nstopped")


# -- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", choices=["file", "mic"], default="mic")
    p.add_argument("--wav", help="input WAV for --source file")
    p.add_argument("--realtime", action="store_true", help="pace file replay at real time")
    p.add_argument("--model-dir", default=root_path("models"))
    p.add_argument("--music-dir", default=root_path("music"))
    p.add_argument("--no-audio", action="store_true", help="don't play music (no speaker needed)")
    p.add_argument("--speaker", choices=["auto", "print"], default="auto")
    p.add_argument("--min-confidence", type=float, default=PipelineConfig.min_confidence)
    p.add_argument("--wake-window", type=float, default=PipelineConfig.wake_window_s)
    p.add_argument("--no-wake", action="store_true", help="act on commands without a wake word")
    p.add_argument("--no-ambient", action="store_true")
    p.add_argument("--device", default=None, help="mic device index/name")
    return p.parse_args()


def main() -> None:
    from .classifier import Classifier
    from .dispatch import PrintSpeaker, default_speaker
    from .play_music_state_machine import PlayMusicStateMachine
    from .player import Library, Player

    args = parse_args()
    library = Library(args.music_dir)
    sm = PlayMusicStateMachine(playlists=library.playlists())
    speaker = PrintSpeaker() if args.speaker == "print" else default_speaker()
    dispatcher = Dispatcher(sm, speaker)
    player = None if args.no_audio else Player(library)
    ambient = None if args.no_ambient else AmbientAutoVolume(sm.volume)
    config = PipelineConfig(
        min_confidence=args.min_confidence,
        wake_window_s=args.wake_window,
        require_wake=not args.no_wake,
    )
    pipeline = Pipeline(Classifier(args.model_dir), dispatcher, player, ambient, config=config)

    try:
        if args.source == "file":
            if not args.wav:
                raise SystemExit("--source file needs --wav")
            run_wav(pipeline, load_stream_wav(args.wav), realtime=args.realtime)
        else:
            device = int(args.device) if args.device and args.device.isdigit() else args.device
            run_mic(pipeline, device)
    finally:
        if player is not None:
            player.close()
        dispatcher.timers.cancel_all()


if __name__ == "__main__":
    main()
