"""Standalone hardware check for the breadboard LED kit -- run this once it's
wired (see GPIO_WIRING.md at the project root) to confirm the wiring is
correct before trusting `dispatch.py`'s real light handlers. Cycles through
every state with a printed label so you can match what's on screen to what
the lights are actually doing, without needing the classifier, mic, or
anything else running.

Usage (on the Pi, in the venv): python -m vcm.gpio_smoke_test
"""

from __future__ import annotations

import time

from .dispatch import GPIOLightController

STEPS: list[tuple[str, bool, int, str | None]] = [
    ("off", False, 100, None),
    ("on, full white", True, 100, None),
    ("red", True, 100, "red"),
    ("green", True, 100, "green"),
    ("blue", True, 100, "blue"),
    ("yellow", True, 100, "yellow"),
    ("dim white (20%)", True, 20, None),
    ("off", False, 100, None),
]


def main() -> None:
    lights = GPIOLightController()
    print("GPIO smoke test -- watch the breadboard, not this terminal, for the real check.\n")
    for label, on, brightness, color in STEPS:
        print(f"-> {label}")
        lights.set_state(on, brightness, color)
        time.sleep(2.0)
    print("\nDone. If every step above matched what you saw on the breadboard, wiring is good.")


if __name__ == "__main__":
    main()
