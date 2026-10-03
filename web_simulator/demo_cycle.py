"""Throwaway test driver: cycles state.json through light states and every
phone screen. Run: python web_simulator/demo_cycle.py
Serve with: python -m http.server -d web_simulator 8000
"""
import itertools, json, os, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "state.json")
LIGHTS = [(False, 100, None)] + [
    (True, b, c) for c in ("red", "green", "blue", "yellow", "white", "pink", "cool") for b in (100, 60, 20)
]
PHONE = [
    ("time", "It's 3:15 PM", {"clock": "15:15"}),
    ("weather", "It's sunny and 28 degrees", {}),
    ("alarm", "Alarm set for 6:00 AM", {"alarm_time": "6:00 AM"}),
    ("timer", "Timer set for 10 seconds", {"seconds": 10}),
    ("timer_done", "Your 10 seconds timer is done", {"label": "10 seconds"}),
    ("temperature", "Setting the temperature to 22 degrees", {"temperature": 22}),
    ("reminder", "Reminder added: Study", {"reminders": ["Study"]}),
    ("reminder", "Your reminders: Study", {"reminders": ["Study"]}),
    ("call", "Calling your emergency contact", {}),
    ("message", "Message sent", {}),
]
state = {"lights_on": False, "brightness": 100, "color": None, "phone": None}
device = {"alarm": None, "temperature": None, "reminders": []}
phone = itertools.cycle(PHONE)
seq = 0
for i, (on, b, c) in enumerate(itertools.cycle(LIGHTS)):
    state.update(lights_on=on, brightness=b, color=c)
    if i % 3 == 2:  # every 3rd step fire the next phone screen
        kind, text, data = next(phone)
        seq += 1
        device["alarm"] = data.get("alarm_time", device["alarm"])
        device["temperature"] = data.get("temperature", device["temperature"])
        device["reminders"] = data.get("reminders", device["reminders"])
        state["phone"] = {"status": "calling" if kind == "call" else kind, "text": text, "data": data,
                          "seq": seq, "device": dict(device),
                          "updated_at": datetime.now().isoformat(timespec="seconds")}
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(PATH + ".tmp", "w") as f:
        json.dump(state, f)
    os.replace(PATH + ".tmp", PATH)
    time.sleep(4 if i % 3 == 2 else 2)
