# HANDOFF — ME2 Voice Command Model

Project: ME2 Voice Command Model, individual topic = `play_media` (play_music + media_control combined per assignment's explicit "No. 1 can fold into No. 8")

Single source of truth for this project. Lives in the repo root, not duplicated into a separate chat document.

## Correction vs. earlier handoff draft (2026-09-25)

The previous handoff described `play_music_state_machine.py`, `ambient_and_beatsync_skeleton.py`,
`smoke_test_train.py`, `benchmark_harness.py`, and `personal_play_music/` as already built and
included in `vcm_dataset_scaffold.zip`. On unzipping, **none of those existed** — the zip contained
only `phrase_list.csv`, `manifest.csv`, `dataset/`, and `README.md`. A filesystem search turned up no
copies elsewhere on this machine either. Treating those components as **not built** until proven
otherwise. This file now reflects actual repo state, not the aspirational description.

## Hard constraints

- Deadline: live demo **Oct 3**. Today: **Sep 25**. 8 days, no buffer built in yet.
- No cloud, no LLM, on-device only (exception: optional Spotify metadata lookup, lowest priority, never built)
- "Train from scratch" ambiguity unresolved with professor, current working assumption: no pretrained weights, small classifier built from zero. Confirm with professor if time allows, not currently blocking.
- Benchmark must include evaluators other than the owner (professor's own quote: "should be able to respond to any person"), each command said n times by each evaluator, output logs required (transcript/prediction/confidence/latency per attempt).

## Hardware status

- Bought: Pi 4 (4GB confirmed), Okdo PSU, 64GB SD, dual-fan aluminum case, HDMI cable, OS pre-installed. ₱8,000.
- **Not yet bought, still the literal first blocking task**: USB mic (~₱250-400, avoid sub-₱100 listings) and a speaker (~₱200-500, USB or 3.5mm). Professor confirmed generic mic/speaker is fine.
- ReSpeaker HAT rejected, not worth the premium given current scope.
- Untested risk: dual-fan noise near the mic. Test the moment mic + Pi are together, before building noise-floor calibration around it.

## Intent scope (locked)

7 core intents + reject class: `light_on_off`, `light_dim_color`, `set_timer`, `set_temperature`, `ask_time`, `media_control`, `play_music` (individual specialty). Cut: ask_weather, reminders, call_contact, compound queries, chart-ranking music.

`play_media` feature stack, priority order:

1. **State machine (#5)** — **built this session**, `src/vcm/play_music_state_machine.py`, 17 passing tests in `src/tests/`. No-repeat playlist selection (shuffled cycle, no immediate repeat even across wraparound), play/pause/stop/next/previous, easter eggs (`good_morning` / `stage_fright`), "what's playing" query.
2. **Ambient auto-volume (#4)** — **built this session**, `src/vcm/ambient_volume.py` (renamed from the planned `ambient_and_beatsync_skeleton.py` — beat-sync is fully cut, no skeleton half left to justify that name). Asymmetric-EMA noise-floor tracker (rises slowly so a brief loud sound doesn't fool it, falls quickly to recognize a quieter room) + calibration curve + hysteresis + rate limiting. 12 passing tests against synthetic white noise. **Calibration constants (`NOISE_FLOOR_QUIET_DBFS`, `NOISE_FLOOR_LOUD_DBFS`) are placeholders** — recalibrate the moment mic + Pi are together, especially against the dual-fan noise risk below.
3. **Volume-ducking during playback** — **volume-side logic built** (`duck()`/`unduck()` methods on the state machine, drop to ~5%, restore on unduck). **Not wired to a wake word** — no wake-word detector exists yet (needs the mic, which isn't bought), so nothing calls `duck()` automatically yet. This is the one piece of the stack that's half-done rather than fully done or fully not-done.
4. **Volume_up/volume_down handlers** — **built**, on the state machine (`_volume_up`/`_volume_down`, ±10 steps, clamped 0-100).
5. **Beat-sync (#3)** — cut entirely, confirmed. Don't build.

**New gap found while building the state machine, not previously flagged:** the classifier trains on the 8-way *intent* (`media_control`, `play_music`, etc.), but `media_control` alone bundles 7 different actions as *slots* (play/pause/stop/next/previous/volume_up/volume_down) that the intent-only classifier cannot distinguish between — it'll say "media_control" but not which of the 7. `play_music` has a similar spread (5 playlists + whats_playing + 2 easter eggs). The state machine's `handle_command()` takes these fine-grained slot values directly (that's the correct API for it to expose), but **something has to resolve intent+audio down to a slot value first, and that something doesn't exist and isn't scheduled anywhere in the 8-day plan yet.** Options: (a) train a second, small slot classifier scoped to just media_control/play_music utterances, (b) keyword-spot within the recognized utterance, (c) redefine scope so each slot becomes its own top-level intent (blows up the "7 intents" count, probably not viable this late). Needs a decision before the Pi wiring step, flagged as open below.

Creative additions, priority order if time allows: sleep timer (cross-intent w/ set_timer, highest value, build this one if only one), confidence-gated clarification, repeat/undo last command, time-of-day default playlist, volume crossfade. None built. Realistically 1-2 achievable in 8 days.

## Dataset status

- Team collective: SLURP + FSC + Snips + classmates' synthetic pipelines, pooled via shared Drive/repo (a classmate's whisper.cpp-validated pooling repo exists for the class, ask in group chat for access if needed).
- Personal dataset (`dataset/`, `phrase_list.csv`, `manifest.csv`): 69 phrases across 7 intents + reject, 201 synthetic WAVs (espeak-ng, 3 voices: us/gb/rp), 22050Hz mono PCM16, 0.6–2.9s duration. Almost entirely synthetic TTS, **zero real utterances** as of this repo snapshot (`personal_play_music/` 70-sample slice mentioned in the earlier draft does not exist in this repo).
- Per-intent counts: media_control 48, play_music 30, set_timer 30, light_dim_color 24, light_on_off 24, set_temperature 18, ask_time 12, reject 15 (offvocab only — `reject/slot=silence` and `reject/slot=noise` still need real recordings, TTS can't produce them).
- Confirmed by class data, not theoretical: synthetic-only training fails on real voices. One classmate: 73% word error rate training on synthetic, testing on real voice. **Real voice data is the single highest-priority gap.** Folder/manifest structure is built for real recordings to drop straight in (see `README.md`).

## Code inventory (actual, as of this session)

| File | Status |
|---|---|
| `src/vcm/audio.py` | Built this session — waveform loading, resample, fixed-length pad/trim, log-mel feature extraction |
| `src/vcm/data.py` | Built this session — manifest loading, stratified train/val split, tf.data pipeline with on-the-fly augmentation |
| `src/vcm/model.py` | Built this session — small CRNN (Conv2D blocks + GRU + softmax), sized for Pi 4 real-time inference |
| `src/vcm/train.py` | Built this session — trains, evaluates, exports Keras + TFLite (dynamic-range quantized) + labels/config JSON |
| `src/vcm/play_music_state_machine.py` | Built this session — playlist state machine, transport controls, volume, easter eggs, duck/unduck hooks. 17 tests in `src/tests/test_play_music_state_machine.py`, all passing. |
| `src/vcm/ambient_volume.py` | Built this session — noise-floor tracker + volume curve + hysteresis. 12 tests in `src/tests/test_ambient_volume.py`, all passing. Calibration constants are placeholders pending real mic. |
| `benchmark_harness.py` | **Not built.** Needs an evaluator-logging companion (transcript/prediction/confidence/latency per attempt). |
| `phrase_list.csv`, `dataset/`, `manifest.csv` | Present, scaffolding intact, real recordings drop in here per `README.md`. |

## Honest current status

Rebuilding the completion estimate from the corrected file inventory: a trainable classifier pipeline now exists and runs end-to-end on the synthetic dataset, including a real TFLite export verified to load and run through the standard `tf.lite.Interpreter` (XNNPACK delegate, no Flex ops needed -- important, means plain `tflite-runtime` on the Pi will work, no custom build required).

**Training run results (synthetic data only, 201 clips, 80-epoch budget, early-stopped ~epoch 20):**
- Val accuracy: **24%** against 8 classes (chance is ~12.5%), so the model is learning *something*, but it's collapsing toward the two largest classes (`media_control` 48 samples, `set_timer` 30 samples) and getting ~0% recall on smaller classes (`ask_time` 12, `reject` 15, `light_on_off`/`light_dim_color` 24 each). See `models/eval_report.txt` for the full confusion matrix.
- This is consistent with, not contradicting, the handoff's warning about synthetic-only training: 201 clips across 8 classes from only 3 TTS voices is not enough signal, and the model is doing the predictable thing (betting on the majority classes). Expect a real jump once real recordings land and the class balance improves.
- Artifacts: `models/vcm_crnn.keras`, `models/vcm_crnn.tflite` (595KB, not tracked in git, regenerate with `python -m vcm.train` from `src/`), `models/labels.json`, `models/training_config.json`, `models/eval_report.txt` (tracked).

The play_media state machine (playlist selection, transport, volume, easter eggs, duck hooks) and the ambient auto-volume module are also built and tested this session -- 29 passing tests total across both. **Nothing has been validated on real voice, nothing has touched the Pi, the evaluator-logging harness is unbuilt, the ambient-volume calibration constants are unverified guesses, and the intent→slot resolution gap above has no owner yet.** Mic/speaker purchase is still the literal first blocking task for anything hardware-facing.

**Next code priorities, in order:** (1) retrain once real recordings exist -- same `python -m vcm.train` command, no code changes needed, (2) decide + build the intent→slot resolution approach (see gap above), (3) evaluator-logging harness, (4) once mic exists, recalibrate `ambient_volume.py`'s noise-floor constants against real (and dual-fan) ambient audio.

## 8-day critical path

- [ ] Today: buy mic + speaker
- [ ] Today/tomorrow: record own voice against full phrase list, retrain, see the real-vs-synthetic gap directly
- [x] Build the actual CNN/CRNN classifier (replace nonexistent `smoke_test_train.py`), quantize, export (TFLite) — pipeline built and verified this session, 24% val accuracy on synthetic-only data (expected to be weak, real data is the fix)
- [x] Build the state machine, add ducking + volume_up/down — built and tested this session (ducking logic exists but isn't wired to a wake word yet, since no wake-word detector exists)
- [x] Build ambient auto-volume (#4) — built and tested this session against synthetic noise; calibration constants are placeholders, recalibrate once mic exists
- [ ] Wire mic capture -> VAD -> classifier -> state machine on the Pi. Test dual-fan noise the moment hardware is together
- [ ] Recruit 2-3 external evaluators, build the evaluator-logging script, run real validation
- [ ] Build 1 (max 2) creative additions, sleep timer first
- [ ] Rehearsal buffer, last 1-2 days minimum, don't skip

## Open, non-blocking

- Train-from-scratch clarification with professor.
- Whether play_music/media_control stay as two internal labels (recommended — used as two separate classes in the classifier built this session) vs merge into one, decide before final report writing, not before building.
- **Intent→slot resolution** (see feature-stack section above): the classifier says "media_control" or "play_music", the state machine needs "next" or "playlist_jazz". Nothing bridges that yet. Not blocking today (mic still isn't bought, nothing plugs into the state machine yet either), but it blocks the "wire mic → VAD → classifier → state machine" critical-path step, so it needs a decision before that step starts, not during it.
