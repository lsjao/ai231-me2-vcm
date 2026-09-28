# HANDOFF — ME2 Voice Command Model

Project: ME2 Voice Command Model, individual topic = `play_media` (play_music + media_control combined per assignment's explicit "No. 1 can fold into No. 8")

Single source of truth for this project. Lives in the repo root, not duplicated into a separate chat document. For the chronological story of what happened and how problems were diagnosed/fixed (useful for the assignment's process writeup), see `daily_log_report.md` -- this file is current state and the plan, that one is the log.

## Correction vs. earlier handoff draft (2026-09-25)

The previous handoff described `play_music_state_machine.py`, `ambient_and_beatsync_skeleton.py`,
`smoke_test_train.py`, `benchmark_harness.py`, and `personal_play_music/` as already built and
included in `vcm_dataset_scaffold.zip`. On unzipping, **none of those existed** — the zip contained
only `phrase_list.csv`, `manifest.csv`, `dataset/`, and `README.md`. A filesystem search turned up no
copies elsewhere on this machine either. Treating those components as **not built** until proven
otherwise. This file now reflects actual repo state, not the aspirational description.

**Second correction (same day, later sanity-check pass):** `handoff.txt` (untracked, gitignored) had
an "Addendum Part 2" appended to it after the initial read at the start of this session — a recording
protocol, easter-egg/playlist audio sourcing tasks, a confidence-recalibration risk, a retries-to-
success benchmark metric, a concrete evaluator plan, and a day-5 fallback checkpoint. It went
unnoticed until an explicit sanity-check pass re-read the file. **Going forward, edit HANDOFF.md
directly instead of handoff.txt** — handoff.txt being gitignored means changes to it don't show up
in `git status` and are easy to lose track of silently, exactly what happened here.

## Hard constraints

- Deadline: live demo **Oct 3**. Today: **Sep 25**. 8 days, no buffer built in yet.
- No cloud, no LLM, on-device only (exception: optional Spotify metadata lookup, lowest priority, never built)
- "Train from scratch" ambiguity unresolved with professor, current working assumption: no pretrained weights, small classifier built from zero. Confirm with professor if time allows, not currently blocking.
- Benchmark must include evaluators other than the owner (professor's own quote: "should be able to respond to any person"), each command said n times by each evaluator, output logs required (transcript/prediction/confidence/latency per attempt). **Built this session** — see feature-stack section and code inventory below. On "transcript": this project has no ASR step anywhere (audio -> intent classifier directly), so the harness logs the ground-truth prompt phrase as the transcript field. Judgment call, not certain that's what's meant; cheap to revisit if wrong.

## Hardware status

- Bought: Pi 4 (4GB confirmed), Okdo PSU, 64GB SD, dual-fan aluminum case, HDMI cable, OS pre-installed. ₱8,000.
- Speaker bought (Sep 25). **USB mic arrives Sun Sep 27** -- until then the laptop's built-in mic is used for recording (see Decisions). Professor confirmed a generic mic/speaker is fine.
- ReSpeaker HAT rejected, not worth the premium given current scope.
- Untested risk: dual-fan noise near the mic. No PWM fan control detected in sysfs and nothing in config.txt, so this case's fans (if wired at all yet) are likely simple always-on units wired straight to power, not something controllable/queryable from software -- whether they're actually spinning has to be confirmed by looking/listening at the case, not by a command. Baseline quiet-room reading taken (see Pi bring-up below); fan-on comparison still needed.

## Pi bring-up (Sep 28, done)

SSH access set up and working: key-based login (`ssh rpi`, alias in `~/.ssh/config` on the dev laptop), passwordless sudo. Host: `rpi-jao` / `jaolacuata@192.168.86.4`. Repo copied via `git archive | ssh ... tar -x` (tracked files only) plus `models/*.tflite`/`labels.json`/`training_config.json` via `scp` (gitignored, not in git archive). `scripts/pi_setup.sh` ran clean on Debian 13 (trixie) / Python 3.13 / aarch64 -- `ai-edge-litert` had a prebuilt wheel for this exact combo, no fallback needed. `python -m vcm.pi_check --mic` now passes every check on real hardware:

- Model load 0.14s (incl. warmup), classify latency mean 17.5ms / p95 17.6ms (budget 500ms) -- comfortably real-time on a Pi 4.
- Speaker + espeak-ng TTS confirmed audible.
- Mic capture confirmed working, peak 0.054, room level dBFS p10 -44.8 / p50 -42.9 / p90 -39.4 (quiet room, fan state unconfirmed -- see risk above). Reasonably close to the -50 "quiet" placeholder in `ambient_volume.py`; not yet worth recalibrating off one reading, revisit once the fan-on comparison exists too.

**Two real hardware problems found and fixed, not hypothetical:**
1. **PipeWire, not raw ALSA, owns audio routing on this OS image.** `~/.asoundrc` is silently ignored. The USB mic's card was the default *sink* (wrong -- it has no real speaker), which is why the 3.5mm-jack speaker (confirmed working via `speaker-test`) produced no sound through code that used the "default" device. Fixed with `wpctl set-default <sink-id>` pointed at the 3.5mm jack (`Built-in Audio Stereo`). **This is a runtime setting, not a config file — if it doesn't survive a reboot, re-run `wpctl status` to find the sink id and `wpctl set-default <id>`.**
2. **The USB mic only supports 48000 Hz capture, not the model's 16000 Hz** (confirmed via `sd.check_input_settings` probing every common rate -- only 48000 succeeded). Fixed properly in code, not worked around: `src/vcm/capture.py` + `audio.resample_integer_ratio()` (dependency-free windowed-sinc decimator, since the Pi deliberately has no scipy/librosa) -- picks the model's rate directly when a mic supports it (e.g. the laptop's mic, unaffected), otherwise captures at the mic's native rate and downsamples. Wired into every capture site (`pi_check`, `record_dataset`, `benchmark_harness` live mode, `pipeline`'s streaming mic loop). Caught and fixed a real bug in the decimator itself during testing (wrong center-tap value distorted the filter) -- verified against `librosa.resample` and an above-Nyquist attenuation test before trusting it. **Moral: this class of "device doesn't support the rate we assumed" bug is real and would have silently broken the live demo on the actual hardware if `pi_check` hadn't been run before demo day.**

Not yet done: fan-on noise comparison, live pipeline test on the Pi (`python -m vcm.pipeline --source mic`), copying/using real recorded data on the Pi (training still happens on the laptop; only inference artifacts belong on the Pi).

## Decisions (Sep 25-26, confirmed with the user)

1. **Real voice is recorded now on the laptop mic**, then topped up with the USB mic when it arrives (Sun Sep 27) so the model sees the demo mic too.
2. **One classifier over command labels** (`intent/slot`, e.g. `media_control/next`, `play_music/playlist_jazz`, `set_timer/5min`; every reject slot collapses to `reject`). Chosen over a two-stage intent-then-slot design: one model on the Pi, and the state machine gets its action straight from the label. Cost: 33+ classes, so per-command real reps matter.
3. **Wake word is a trained class** in the same classifier (`wake/hey_pi`, phrase "hey pi" -- change it in `phrase_list.csv` if you want another). Commands are only acted on while a *wake window* is open (5 s); opening it ducks the music to ~5%, and the command (or the timeout) restores it.

## Intent scope (locked)

7 core intents + reject class: `light_on_off`, `light_dim_color`, `set_timer`, `set_temperature`, `ask_time`, `media_control`, `play_music` (individual specialty). Cut: ask_weather, reminders, call_contact, compound queries, chart-ranking music.

`play_media` feature stack, priority order:

1. **State machine (#5)** — **built this session**, `src/vcm/play_music_state_machine.py`, 17 passing tests in `src/tests/`. No-repeat playlist selection (shuffled cycle, no immediate repeat even across wraparound), play/pause/stop/next/previous, easter eggs (`good_morning` / `stage_fright`), "what's playing" query.
2. **Ambient auto-volume (#4)** — **built this session**, `src/vcm/ambient_volume.py` (renamed from the planned `ambient_and_beatsync_skeleton.py` — beat-sync is fully cut, no skeleton half left to justify that name). Asymmetric-EMA noise-floor tracker (rises slowly so a brief loud sound doesn't fool it, falls quickly to recognize a quieter room) + calibration curve + hysteresis + rate limiting. 12 passing tests against synthetic white noise. **Calibration constants (`NOISE_FLOOR_QUIET_DBFS`, `NOISE_FLOOR_LOUD_DBFS`) are placeholders** — recalibrate the moment mic + Pi are together, especially against the dual-fan noise risk below.
3. **Volume-ducking during playback** — **built and wired** (`duck()`/`unduck()` on the state machine, driven by the wake window in `src/vcm/pipeline.py`; tested with a scripted classifier + synthetic audio). Untested with a real wake-word model or real music through the speaker.
4. **Volume_up/volume_down handlers** — **built**, on the state machine (`_volume_up`/`_volume_down`, ±10 steps, clamped 0-100).
5. **Beat-sync (#3)** — cut entirely, confirmed. Don't build.

**Intent-to-slot gap: resolved** (was flagged earlier: an intent-only classifier can't tell `media_control/next` from `media_control/pause`). The classifier now predicts command labels directly -- see Decisions above.

Creative additions, priority order if time allows: sleep timer (cross-intent w/ set_timer, highest value, build this one if only one), confidence-gated clarification, repeat/undo last command, time-of-day default playlist, volume crossfade. None built. Realistically 1-2 achievable in 8 days.

## Dataset status

- Team collective: SLURP + FSC + Snips + classmates' synthetic pipelines, pooled via shared Drive/repo (a classmate's whisper.cpp-validated pooling repo exists for the class, ask in group chat for access if needed).
- Personal dataset (`dataset/`, `phrase_list.csv`, `manifest.csv`): 70 phrases (69 across 7 intents + reject, plus `wake/hey_pi`), 201 synthetic WAVs (espeak-ng, 3 voices: us/gb/rp), 22050Hz mono PCM16, 0.6–2.9s duration. Almost entirely synthetic TTS, **zero real utterances** as of this repo snapshot (`personal_play_music/` 70-sample slice mentioned in the earlier draft does not exist in this repo).
- Per-intent counts: media_control 48, play_music 30, set_timer 30, light_dim_color 24, light_on_off 24, set_temperature 18, ask_time 12, reject 15 (offvocab only — `reject/slot=silence` and `reject/slot=noise` still need real recordings, TTS can't produce them).
- Confirmed by class data, not theoretical: synthetic-only training fails on real voices. One classmate: 73% word error rate training on synthetic, testing on real voice. **Real voice data is the single highest-priority gap.** Folder/manifest structure is built for real recordings to drop straight in (see `README.md`).

### Recording protocol (concrete spec, was never written down before this)

- **Mic distance**: most samples at 6-12 inches (typical demo distance), plus a handful at 2-3 feet (worst-case) so the model isn't brittle to exact positioning.
- **Phrasing variants**: minimum 2-3 different ways of saying each command (e.g. "pause" / "pause the music" / "stop the music for now").
- **Noise conditions**: a portion in silence, a portion with background TV/music, a portion with ambient room noise (fan, traffic). Doesn't need to be even, but zero noisy samples is the failure mode to avoid — and this is also what fills in `reject/slot=silence` and `reject/slot=noise`, which TTS can't produce (see dataset status above).
- **Target volume**: ~20-30 real repetitions per intent/slot combination as a baseline, more for play_media specifically since it's the graded specialty.

## Code inventory (actual, as of this session)

| File | Status |
|---|---|
| `src/vcm/labels.py` | Command labels (`command_label`, `intent_of`, `slot_of`), manifest reading. Deliberately TensorFlow-free so the Pi can import it. |
| `src/vcm/audio.py` | Waveform loading, fixed-length pad/trim, `trim_to_speech`, and a **numpy log-mel verified against librosa** (`tests/test_audio.py`) so the Pi needs no librosa/numba. librosa only used lazily for resampling files. |
| `src/vcm/data.py` | Per-label train/val split, tf.data pipeline with augmentation (noise, gain, shift). Training-side only (imports TensorFlow). |
| `src/vcm/model.py` | Small CRNN (Conv2D blocks + unrolled BiGRU + softmax), ~43K params, sized for Pi 4. |
| `src/vcm/train.py` | Trains on command labels, `--extra-data <dir>` merges real/pooled data roots, exports Keras + TFLite (plain builtin ops) + `labels.json`/`training_config.json`, per-command + per-intent report. |
| `src/vcm/classifier.py` | TFLite wrapper (`ai_edge_litert` -> `tflite_runtime` -> `tensorflow` fallback chain), warms up at load. Shared by harness and pipeline. |
| `src/vcm/record_dataset.py` | Guided real-voice recording: passes over `phrase_list.csv`, auto-trims each take, resume-safe, redo/skip/quit. Writes to `data_real/` (gitignored). |
| `src/vcm/import_recording.py` | Imports long phone/friend recordings: prints a numbered script, splits the memo with the live VAD, labels takes by order, refuses on count mismatch. |
| `src/vcm/dispatch.py` | Routes a command label to the state machine or a handler: ask_time, set_timer (real background timer), set_temperature/lights (simulated devices), TTS via espeak-ng (`PrintSpeaker` fallback). |
| `src/vcm/player.py` | Plays `music/<playlist>/*.mp3, wav, ogg, flac` (+ `music/easter/`) via sounddevice with software-gain volume; playlists with no files fall back to a distinct tone per stub title so transport/volume/ducking are testable with no music. |
| `src/vcm/vad.py` | Streaming energy endpointer (adaptive floor frozen during speech, hysteresis, pre-roll/tail matching the training-clip trim). Constants are placeholders until tuned on the real mic. |
| `src/vcm/pipeline.py` | Live loop: mic/WAV frames -> VAD -> classifier -> wake window (duck/unduck) -> dispatch -> player. Ambient auto-volume adapts only while nothing is playing (the mic hears the speaker). `--source file` replays a WAV (deterministic), `--source mic` is live. |
| `src/vcm/pi_check.py` | Diagnostic for the Pi: imports, speaker tone, TTS, model latency, and `--mic` room-noise floor (the dual-fan test). |
| `scripts/pi_setup.sh` | One-shot Pi install (apt deps, venv, `requirements-pi.txt`, `ai-edge-litert` with `tflite-runtime` fallback). |
| `src/vcm/play_music_state_machine.py` | Built this session — playlist state machine, transport controls, volume, easter eggs, duck/unduck hooks. 17 tests in `src/tests/test_play_music_state_machine.py`, all passing. |
| `src/vcm/ambient_volume.py` | Built this session — noise-floor tracker + volume curve + hysteresis. 12 tests in `src/tests/test_ambient_volume.py`, all passing. Calibration constants are placeholders pending real mic. |
| `src/vcm/benchmark_harness.py` | Built this session — evaluator-logging CLI, CSV per attempt (timestamp/evaluator/phrase/predicted_intent/confidence/correct/latency_ms/audio_filepath), plus a summary with accuracy, per-intent breakdown, and retries-to-success (mean attempts until first correct prediction per command, commands that never succeeded within `reps` — the addendum's requested metric). Two modes: `live` (real mic via `sounddevice`, needs the Pi's mic) and `replay` (existing dataset WAVs, for smoke-testing the harness itself, not a real benchmark). `--evaluator-plan` runs the addendum's exact plan (every play_media command, one prompt per slot, 3x each). scores at command level (right intent AND slot; also logs intent-only correctness), replay picks clips per command; 19 tests in `src/tests/test_benchmark_harness.py`, all passing; also run end-to-end against the real trained model in replay mode, including with `--evaluator-plan` (45 attempts across 15 play_media commands, consistent with the classifier's known weakness, not a harness bug). |
| `phrase_list.csv`, `dataset/`, `manifest.csv` | Synthetic scaffolding, tracked in git. Real recordings do NOT go here -- `record_dataset.py` writes them to the gitignored `data_real/` (own manifest), merged at train time with `--extra-data`. |

## Honest current status

**Software is feature-complete end to end; the remaining gaps are all data and hardware.** 128 tests pass, pyflakes clean. Verified for real (not just unit tests): laptop mic capture through the recording tool, real speaker playback through the player (play/pause/next/duck/stop), the stitched-WAV pipeline run with the real TFLite model (4 clips -> 4 utterances -> classified -> dispatched), and `pi_check` on the laptop (model latency ~4 ms mean; laptop mic room floor p50 -48 dBFS, close to the -50 "quiet" placeholder in `ambient_volume.py`).

- **Classifier is still untrained for real**: the current `models/` artifacts are a 33-command-label model trained on 201 synthetic clips (~5 per command) -- ~5% val accuracy, confidence ~0.03. It exists to prove the pipeline/export, not to demo. Real recordings + retrain is the next step and needs no code changes.
- **Not yet done**: nothing has touched the Pi; no real wake-word/command audio exists; no real music files (tones stand in); VAD margins, ambient constants, and the wake/reject confidence threshold (`--min-confidence`, default 0.5) are all untuned guesses until the USB mic + real data exist; no evaluator has been run.
- Artifacts: `models/vcm_crnn.tflite` (~596KB) and `.keras` are gitignored -- regenerate with `python -m vcm.train` from `src/`, or copy the `.tflite` + `labels.json` + `training_config.json` to the Pi.
- **Real recordings are gitignored** (`data_real/`: bulky, and it's your voice). Back it up yourself (zip to Drive) -- git will not.

### How to record (do this now, laptop mic)

**Windows gotcha:** this machine has several Pythons, and a bare `python` (or a VS Code terminal's active interpreter) can be one without the packages (`ModuleNotFoundError: soundfile`). From the repo root use the launcher `run.cmd`, which pins the right interpreter and runs from `src/`: `.\run.cmd vcm.record_dataset --speaker josh ...` -- i.e. wherever a command below says `python -m X ...`, type `.\run.cmd X ...`.

Install once: `pip install sounddevice`. Play_media first (graded), then everything else. `--auto` needs no Enter: it shows the phrase, pauses 1 s, plays a beep, then records -- **speak right after the beep**; `Ctrl+C` stops safely and re-running resumes.

```
python -m vcm.record_dataset --speaker <you> --condition quiet --distance near --reps-per-slot 16 --intents media_control,play_music --auto
python -m vcm.record_dataset --speaker <you> --condition quiet --distance near --reps-per-slot 16 --intents wake,reject --auto
python -m vcm.record_dataset --speaker <you> --condition quiet --distance near --reps-per-slot 16 --auto     # the rest
python -m vcm.record_dataset --speaker <you> --condition tv    --distance near --reps-per-slot 6  --auto     # background TV/music on
python -m vcm.record_dataset --speaker <you> --condition quiet --distance far  --reps-per-slot 3  --auto     # 2-3 ft
```

For the `reject` silence/noise prompts: stay silent, or make TV/fan/typing noise as prompted. Then retrain: `.\run.cmd vcm.train` (paths default to the project root; `data_real/` is merged in automatically). Do a short top-up run with the USB mic on Sunday using the same commands with `--condition usbmic`.

### Recording with a phone (yours, or a friend's)

Good for mic diversity and for getting *other people's* voices without them touching Python. Formats: m4a/aac (needs `pip install av`, already installed here), mp3, wav, flac, ogg -- no conversion needed. Per block of ~20 phrases:

1. Print the script: `.\run.cmd vcm.import_recording --speaker <name> --condition phone --distance near --intents media_control,play_music --block-size 20` (add `--script-file script.txt` to write it to a file you can send a friend).
2. Record ONE voice memo reading it: ~1 s of silence first, each phrase once, ~1.5 s pause between phrases, no pauses inside a phrase, in order.
3. Import: `.\run.cmd vcm.import_recording --file "C:\path\to\memo.m4a"`. Then repeat step 1 for the next block.

It refuses (saving nothing) if it hears a different number of utterances than the script has, and prints what it heard next to what was expected -- one miscount would mislabel every take after it. Silence/noise prompts aren't in phone scripts; record those with `record_dataset` on the laptop. Use different `--speaker` names per person. Friends recorded for training must not also be your benchmark evaluators.

### Pi bring-up (needs only Pi + speaker; mic Sunday)

1. Copy the repo (incl. `models/vcm_crnn.tflite`, `labels.json`, `training_config.json`) to the Pi. `bash scripts/pi_setup.sh`.
2. `cd src && python -m vcm.pi_check` -- confirms speaker tone, espeak TTS, model loads and classifies fast (budget 500 ms). **If the model fails to load on the Pi's TFLite runtime, paste the error back**; the fix is a runtime/converter version match.
3. Sunday, mic plugged in: `python -m vcm.pi_check --mic --seconds 10` with fans idle, then with fans running -- compare the `p50` lines; that is the dual-fan noise test and the calibration data for `ambient_volume.py`/`vad.py`.
4. Live: `python -m vcm.pipeline --source mic` (add `--no-wake` to skip the wake window while debugging).

## 8-day critical path (day 1 = Fri Sep 25; demo Sat Oct 3)

- [x] Classifier pipeline, state machine, ambient auto-volume, evaluator harness (built + tested)
- [x] Command-level relabel (closes intent->slot gap), wake-word class, recording tool, dispatcher, player, VAD, live pipeline, Pi setup/diagnostic (built + tested Sep 25-26)
- [x] Mic + speaker + Pi all in hand (Sep 28)
- [ ] **Record real voice on laptop mic** (commands above), back up `data_real/`, retrain, look at `models/eval_report.txt` per-command results
- [ ] Ask classmate group chat for pooled dataset repo access (SLURP/FSC/Snips: FSC has real voices for lights, volume, heat); merge via `--extra-data`
- [x] Pi bring-up: SSH access, `pi_setup.sh`, `pi_check --mic` all passing on real hardware (Sep 28) -- see Pi bring-up section above for the two real bugs found and fixed (PipeWire routing, USB mic 48kHz-only)
- [ ] Fan-on noise comparison (case fans' actual state unconfirmed), USB-mic top-up recording, retrain, live pipeline test (`python -m vcm.pipeline --source mic`) on the Pi; tune `--min-confidence`, VAD margins, ambient constants against real audio
- [ ] Source easter-egg songs ("Good Morning", "No"/stage-fright) and 5+ playlist tracks as local files into `music/` (copyrighted -- must be sourced by you; see `player.py` docstring for layout)
- [ ] **Day-5 fallback checkpoint (Tue Sep 29)**: if the real classifier isn't trained and running on the Pi end-to-end, cut every creative addition and put all remaining time into core play_media + reject-class reliability
- [ ] Evaluator sessions, day 5-6 (Sep 29-30): 2-3 people other than you, `python -m vcm.benchmark_harness --mode live --evaluator "<name>" --evaluator-plan`
- [ ] Creative addition, max 1-2, sleep timer first -- only if the day-5 checkpoint passes
- [ ] Rehearsal buffer, Oct 1-2, don't skip

## Open, non-blocking

- Train-from-scratch clarification with professor.
- Whether play_music/media_control stay as two intents in the report vs one `play_media` (they are separate label families in the classifier either way; decide before writing the report, not before building).
- **Confidence threshold recalibration**: a model trained partly on clean synthetic TTS will likely be overconfident on synthetic-style input and miscalibrated on real recordings. Once real data is in, tune `--min-confidence` (pipeline) against a real-audio validation slice rather than assuming 0.5 holds. Directly affects the false-accept rate. The benchmark harness already logs confidence per attempt to support this.
- **Val accuracy will be optimistic**: the val split is random per label, so the same phrase from the same speaker can land in train and val. The honest number is the evaluator benchmark with people who aren't you.
- Ambient auto-volume can't adapt during playback (mic hears the speaker); echo-aware adaptation is out of scope unless time is left.
