#!/usr/bin/env bash
# Set up the voice-command project on a Raspberry Pi (Raspberry Pi OS, 64-bit).
# Run from the repo root on the Pi. Safe to re-run.
set -euo pipefail

sudo apt update
sudo apt install -y python3-venv python3-pip libportaudio2 libsndfile1 espeak-ng alsa-utils

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt

# TFLite runtime: prefer Google's current ai-edge-litert (model is converted with
# TF 2.20, which older tflite-runtime builds can fail to load); fall back to tflite-runtime.
pip install ai-edge-litert || pip install tflite-runtime

echo
echo "Setup done. Next (from src/, venv active):"
echo "  cd src && python -m vcm.pi_check                    # speaker, TTS, model latency"
echo "  python -m vcm.pi_check --mic --seconds 10           # once the USB mic is plugged in"
echo "  python -m vcm.pipeline --source mic                 # live assistant"
