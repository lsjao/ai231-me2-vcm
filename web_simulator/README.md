# Kuya Jukebox simulator

Browser view of the lights (bulb) and phone (call / message) for when the
hardware isn't wired. Fully offline.

Run on the Pi: `python -m vcm.pipeline --lights web` (add `--web-port N` to change 8000).
Open `http://<pi-ip>:8000/` from any browser on the same network. Real GPIO LEDs
still fire if wired.

Standalone test: `python web_simulator/demo_cycle.py`, with
`python -m http.server -d web_simulator 8000` running.
