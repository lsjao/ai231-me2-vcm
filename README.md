# VCM -- On-Device Voice Command Model (AI 231 ME2)

A small, from-scratch-trained voice assistant that runs entirely on a
Raspberry Pi 4: always-on wake-word + command recognition, no cloud
round-trip. Built for the AI 231 ME2 class assignment (Option B schema,
19 required commands).

## Architecture
- Features: 16 kHz mic audio -> 40-bin log-mel spectrogram x 301 frames (3 s clips)
- Model: Conv2D x3 (16/32/32 ch, BN + maxpool + dropout) -> BiGRU (24-dim,
  unrolled for static-shape TFLite conversion) -> Dense(32) -> 47-way softmax
  joint `"intent/slot"` head (not causal/streaming -- classifies a fixed 3 s window)
- 44,271 params. Exported as a dynamic-range-quantized TFLite model
  (~597 KB) and run on-device via `ai_edge_litert`.
- Actuation is canned/local: GPIO-driven lights, `espeak-ng` TTS piped
  through `aplay`, and scripted responses for calls/messages/weather/reminders
  (no network APIs, per the assignment's standalone-Pi constraint).

## Repo layout
- `src/vcm/` -- feature extraction (`audio.py`), model (`model.py`),
  training (`train.py`), the live pipeline (`pipeline.py`), command dispatch
  (`dispatch.py`), Pi deployment helpers (`pi_check.py`, `gpio_smoke_test.py`)
- `scripts/` -- dataset import/merge scripts (class HuggingFace dataset,
  supplemental datasets, synthetic negatives) and `eval_hf_master_test.py`
  for scoring a trained model against the class's official test/holdout splits
- `src/tests/` -- pytest suite (dispatch logic, state machine, timers)
- `phrase_list.csv` -- locked phrase list used for synthetic/TTS data generation

Training data and trained weights are **not** included in this repository
(see Dataset and Model weights below) -- `data_real/`, `data_real_pi_wake/`,
`external_data/`, `external_data_hf/`, and `models/*.keras`/`*.tflite` are
gitignored.

## Reproducing training
```
python -m vcm.train --data-root <path-with-manifest.csv> --extra-data <other-root> ...
```
Any data root just needs its own `manifest.csv` with columns
`filepath,intent,slot,phrase,speaker,condition,source`; `scripts/import_*.py`
show how each dataset source was pulled and mapped into that schema. Output
(`models/vcm_crnn.{keras,tflite}`, `labels.json`, `training_config.json`,
`eval_report.txt`) is written to `--output-dir` (default `models/`).

Evaluate against the class's official splits:
```
python scripts/eval_hf_master_test.py --split test      # 4,443 clips -> 82.85% command accuracy
python scripts/eval_hf_master_test.py --split holdout    # 202 clips  -> 78.22% command accuracy
```

Run the test suite:
```
pytest src/tests/
```

## Dataset
Training data (~8,169 clips / ~6.8 h, 774 speaker/voice tags) is pooled from:
- Class master set (Option B schema): [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) on HuggingFace
- Filipino-accent supplement: [`martinnavs/ai231-fil-supplemental-data`](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data)
- Synthetic hard-negative clips (noise/babble/reversed/truncated/near-silence) from the same HF org, mapped to the `reject` class
- Public speech datasets folded into the class master set (SNIPS, SLURP,
  TimersAndSuch, FluentSpeechCommands, SpeechCommands v2, Common Voice)
- Locally recorded additions: wake-word clips recorded on the deployment
  Pi's own USB mic (`data_real_pi_wake/`), and earlier own-voice/TTS
  recordings (`data_real/`) -- not yet published; see "Still open" below

Each is used under its own license/access terms as published by the
respective source; none of it is redistributed in this repo.

## Model weights
Not yet released publicly. `models/vcm_crnn.tflite` (and `.keras`) are
produced locally by `vcm.train` and are gitignored in this repo pending a
release decision (location + license for the weights themselves).

## License
This repository's code is MIT-licensed (see `LICENSE`). That covers the code
only -- training data is drawn from third-party and class-provided datasets
under their own licenses/access terms (see Dataset above), and trained model
weights are not included in this repository (see Model weights above).

## Training compute
Trained locally (not on the class A100/DGX cluster) due to timeline
constraints: 150 epochs, batch size 16, Adam (lr 1e-3, ReduceLROnPlateau to
2.5e-4), class-weighted sparse categorical cross-entropy, ~409 steps/epoch
(61,350 steps total). The pipeline accepts data roots via CLI args, so it can
be pointed at a cluster-hosted dataset path the same way, it just wasn't run
there for this submission.

## Status / known limitations
See `daily_log_report.md` for the full build log and `HANDOFF.md` for a
running state-of-the-project summary, including open issues (reject-class
false-accept rate, a few slot-level confusions) and what's still outstanding
for submission.
