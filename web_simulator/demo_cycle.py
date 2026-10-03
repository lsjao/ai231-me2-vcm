"""Throwaway test driver: cycles state.json through light states every 2s.
Run: python web_simulator/demo_cycle.py   (serve with: python -m http.server -d web_simulator 8000)
"""
import itertools, json, os, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = [(False, 100, None)] + [
    (True, b, c) for c in ("red", "green", "blue", "yellow", "white", "pink", "cool") for b in (100, 60, 20)
]
for on, b, c in itertools.cycle(STATES):
    tmp = os.path.join(HERE, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump({"lights_on": on, "brightness": b, "color": c,
                   "updated_at": datetime.now().isoformat(timespec="seconds")}, f)
    os.replace(tmp, os.path.join(HERE, "state.json"))
    time.sleep(2)
