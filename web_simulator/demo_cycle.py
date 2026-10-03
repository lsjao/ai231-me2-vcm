"""Throwaway test driver: cycles state.json through light, call and message
states every 2.5s. Run: python web_simulator/demo_cycle.py
Serve with: python -m http.server -d web_simulator 8000
"""
import itertools, json, os, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "state.json")
STATES = [(False, 100, None)] + [
    (True, b, c) for c in ("red", "green", "blue", "yellow", "white", "pink", "cool") for b in (100, 60, 20)
]
state = {"lights_on": False, "brightness": 100, "color": None, "phone": None}
seq = 0
for i, (on, b, c) in enumerate(itertools.cycle(STATES)):
    state.update(lights_on=on, brightness=b, color=c)
    if i % 6 == 3:  # every 6th step fire a phone event, alternating
        seq += 1
        state["phone"] = {"status": "calling" if seq % 2 else "message", "seq": seq,
                          "updated_at": datetime.now().isoformat(timespec="seconds")}
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(PATH + ".tmp", "w") as f:
        json.dump(state, f)
    os.replace(PATH + ".tmp", PATH)
    time.sleep(2.5)
