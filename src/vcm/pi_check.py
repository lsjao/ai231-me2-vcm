"""Hardware/environment diagnostic. Run on the Pi after scripts/pi_setup.sh,
and again on Sunday when the USB mic is plugged in:

    python -m vcm.pi_check                      # imports, speaker, TTS, model latency
    python -m vcm.pi_check --mic --seconds 10   # + mic level / room-noise floor

The `--mic` run is the dual-fan noise test from HANDOFF.md: run it once with
the fans idle/off and once with them running, a foot from the mic, and
compare the p50 lines. Those numbers are what ambient_volume.py's
NOISE_FLOOR_QUIET_DBFS / NOISE_FLOOR_LOUD_DBFS and vad.py's margins should
be calibrated against.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import time

import numpy as np

from . import audio
from .ambient_volume import FRAME_SAMPLES, NOISE_FLOOR_LOUD_DBFS, NOISE_FLOOR_QUIET_DBFS, frame_dbfs

REALTIME_BUDGET_MS = 500.0  # a command should be classified well under this after the utterance ends


def level_stats(wav: np.ndarray) -> dict:
    """Per-frame dBFS percentiles: p10 ~ the floor, p90 ~ peaks/speech."""
    levels = np.array(
        [frame_dbfs(wav[i : i + FRAME_SAMPLES]) for i in range(0, len(wav) - FRAME_SAMPLES + 1, FRAME_SAMPLES)]
    )
    return {
        "p10": float(np.percentile(levels, 10)),
        "p50": float(np.percentile(levels, 50)),
        "p90": float(np.percentile(levels, 90)),
        "peak": float(np.max(np.abs(wav))),
    }


def latency_stats(latencies_ms: list[float]) -> dict:
    arr = np.array(latencies_ms)
    return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)), "max": float(arr.max())}


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    return ok


def check_imports() -> bool:
    all_ok = True
    for mod, required in [("numpy", True), ("soundfile", True), ("sounddevice", True), ("librosa", False)]:
        try:
            m = importlib.import_module(mod)
            check(f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:  # ImportError, or PortAudio missing for sounddevice
            if required:
                all_ok = check(f"import {mod}", False, str(e))
            else:
                print(f"[--] import {mod}: not installed (optional; only needed to resample files)")
    from .classifier import RUNTIME

    check("TFLite runtime", True, RUNTIME)
    return all_ok


def check_speaker(device: int | str | None) -> bool:
    import sounddevice as sd

    ok = True
    try:
        sr = 22050
        t = np.arange(sr) / sr
        tone = (0.15 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        print("playing a 1s 440 Hz tone (low volume)...")
        sd.play(tone, sr, device=device)
        sd.wait()
        ok = check("speaker tone", True, "did you hear it?")
    except Exception as e:
        ok = check("speaker tone", False, str(e))

    if shutil.which("espeak-ng"):
        subprocess.run(["espeak-ng", "-v", "en-us", "Hello, this is the Raspberry Pi voice assistant."], check=False)
        check("espeak-ng TTS", True, "did you hear it?")
    else:
        check("espeak-ng TTS", False, "not installed (sudo apt install espeak-ng)")
        ok = False
    return ok


def check_model(model_dir: str, runs: int = 50) -> bool:
    from .classifier import Classifier

    try:
        t0 = time.perf_counter()
        clf = Classifier(model_dir)
        load_s = time.perf_counter() - t0
    except Exception as e:
        return check("model load", False, str(e))
    check("model load (incl. warmup)", True, f"{load_s:.2f}s, {len(clf.idx_to_label)} labels")

    rng = np.random.default_rng(0)
    wav = (rng.standard_normal(audio.CLIP_SAMPLES) * 0.1).astype(np.float32)
    stats = latency_stats([clf.predict(wav)[2] for _ in range(runs)])
    return check(
        "classify latency (features + inference)",
        stats["p95"] < REALTIME_BUDGET_MS,
        f"mean {stats['mean']:.1f} ms, p95 {stats['p95']:.1f} ms, max {stats['max']:.1f} ms "
        f"(budget {REALTIME_BUDGET_MS:.0f} ms)",
    )


def check_mic(seconds: float, device: int | str | None) -> bool:
    import sounddevice as sd

    try:
        print(f"recording {seconds:.0f}s of room tone -- stay quiet...")
        rec = sd.rec(int(seconds * audio.TARGET_SR), samplerate=audio.TARGET_SR, channels=1,
                     dtype="float32", device=device)
        sd.wait()
    except Exception as e:
        return check("mic capture", False, str(e))

    s = level_stats(rec[:, 0])
    check("mic capture", s["peak"] > 0.0, f"peak {s['peak']:.3f}")
    print(f"      room level dBFS: p10 {s['p10']:.1f}  p50 {s['p50']:.1f}  p90 {s['p90']:.1f}")
    print(f"      ambient_volume placeholders: quiet {NOISE_FLOOR_QUIET_DBFS:.0f} / loud {NOISE_FLOOR_LOUD_DBFS:.0f} dBFS")
    if s["peak"] < 1e-4:
        print("      mic looks dead/muted (check alsamixer capture level and `arecord -l`)")
    if s["peak"] >= 0.99:
        print("      clipping -- lower the capture gain (alsamixer)")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model-dir", default="../models")
    p.add_argument("--skip-audio", action="store_true", help="skip speaker/TTS tests")
    p.add_argument("--mic", action="store_true", help="also test the mic + measure the noise floor")
    p.add_argument("--seconds", type=float, default=10.0, help="mic recording length")
    p.add_argument("--device-in", default=None)
    p.add_argument("--device-out", default=None)
    return p.parse_args()


def _dev(x: str | None):
    return int(x) if x is not None and x.isdigit() else x


def main() -> int:
    args = parse_args()
    ok = check_imports()
    if ok:
        import sounddevice as sd

        print("\naudio devices:\n" + str(sd.query_devices()) + "\n")
        ok &= check_model(args.model_dir)
        if not args.skip_audio:
            ok &= check_speaker(_dev(args.device_out))
        if args.mic:
            ok &= check_mic(args.seconds, _dev(args.device_in))
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED -- see [FAIL] lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
