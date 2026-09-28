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
- **Speaker + USB mic bought and fully working (Sep 28)** — Pi bring-up complete, see below. (An earlier scope note said these were "not yet bought" — that was stale, from before Sep 28; this line is the current truth.)
- ReSpeaker HAT rejected, not worth the premium given current scope.
- **New (Sep 28): electronic component kit bought/orderable** for real `light_on_off`/`light_dim_color` actuation — 830-point breadboard, single-color LEDs (on/off) + 1 RGB LED (PWM dim/color) + full resistor range incl. LED-safe 220R/330R, from a vetted Shopee listing (~₱403). Covers both light intents with one kit, no second kit needed. Deliberately GPIO-driven, not a smart bulb — a smart bulb would actuate through a vendor cloud API (Tuya/Xiaomi/etc.), which conflicts with the no-cloud constraint; raw GPIO stays fully local. Not yet in hand — code should build the GPIO control path now with a hardware-absent fallback (same pattern as `player.py`'s tone fallback when no music files exist), ready to wire up real pins once the kit arrives.
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
3. **Wake word is a trained class** in the same classifier (`wake/kuya_jukebox`, phrase "kuya jukebox" -- change it in `phrase_list.csv` if you want another). Commands are only acted on while a *wake window* is open (5 s); opening it ducks the music to ~5%, and the command (or the timeout) restores it.

## Intent scope

**Scope change (Sep 28), supersedes "play_media only" framing:** all 7 core intents + reject are now in scope for the **individual** model, not just play_media. Earlier drafts described play_media as the individual deliverable with the other 6 intents being classmates' specialties within a shared team project; after reviewing the class group chat, classmates are building single models covering the full intent list, not specializing — and the professor's "respond to any person" quote refers to the full list. Treat every instruction elsewhere in this file as applying across all 7 intents unless explicitly scoped to play_media.

The 7 intents + reject, explicitly:
- `light_on_off` — real GPIO-driven LED (hardware bought, not yet in hand — see Hardware status)
- `light_dim_color` — real GPIO PWM + RGB LED (same kit)
- `set_timer` — spoken/logged confirmation only, no physical hardware (already how `dispatch.py`'s `TimerManager` works)
- `set_temperature` — spoken/logged confirmation only, **simulated, no real sensor** — don't build real temperature sensing
- `ask_time` — spoken/logged system-clock output (already built)
- `media_control` / `play_music` — the `play_media` grouping, still gets priority polish (most-recorded, most-built: state machine, ducking, easter eggs)
- `reject` — silence/noise/off-vocabulary, applies across all 7 intents now, not just play_media's near-misses

Cut (unchanged): ask_weather, reminders, call_contact, compound queries, chart-ranking music.

**Revised day-5/6 fallback priority, if the checkpoint shows the classifier isn't handling all 7 intents + reject reliably**: cut down toward `play_media` + the 1-2 simplest others (`light_on_off`, `ask_time`) and document the rest as future work in the report, rather than cutting creative play_media features first. This is a real, load-bearing decision point, not a formality — see the 33-class capacity finding below, which makes this fallback more likely to actually get invoked.

**Tension with the 33-class capacity finding (see Honest current status)**: this scope expansion means *more* classes need real data, right when we've just confirmed the model can't learn 33 classes well even with moderate per-class data. Recording priority order should follow the fallback list above: play_media (done) → `light_on_off` (simple, now hardware-relevant) → `ask_time` (simple, single slot) → `light_dim_color`/`set_timer`/`set_temperature` (more classes each, lower priority, first to cut).

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
- Personal dataset (`dataset/`, `phrase_list.csv`, `manifest.csv`): 70 phrases (69 across 7 intents + reject, plus `wake/kuya_jukebox`), 201 synthetic WAVs (espeak-ng, 3 voices: us/gb/rp), 22050Hz mono PCM16, 0.6–2.9s duration. Almost entirely synthetic TTS, **zero real utterances** as of this repo snapshot (`personal_play_music/` 70-sample slice mentioned in the earlier draft does not exist in this repo).
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

## Honest current status (updated Sep 28, evening)

**Software is feature-complete end to end. Hardware/Pi bring-up is done. The real blocker is now a data-capacity wall, not a missing piece.**

- **267 real recordings exist** (246 USB-mic play_media takes, all 246/246 targets hit exactly + 21 earlier laptop-mic takes), verified clean: no clipping, no suspiciously-quiet takes, healthy peak distribution. See "How to record" below for the tool, already proven working end to end on real hardware.
- **Major finding (Sep 28): 33 command classes is too many for the data volume achievable in this timeframe.** Retraining with the real data merged in still gave ~2-3% val accuracy (worse than hoped), with training loss stuck near `ln(33)` (pure-guessing level) for 40+ epochs even on train data. Root-caused via controlled ablations (see `daily_log_report.md` for the full investigation): ruled out class_weight, learning rate, gradient clipping, and a label/feature alignment bug one by one. **Confirmed root cause**: the exact same model/pipeline trained cleanly on the original 8-way intent labels (steady climb to ~30% val accuracy, well above chance) using the identical synthetic data -- it's not a code bug, it's that 33 classes at ~5 examples/class average (many classes as low as 2-3) is a genuine from-scratch-learning capacity wall, not something more epochs or tuning fixes.
- **Scope just expanded to all 7 intents + reject** (see Intent scope above) right as this capacity wall was found -- these two facts are in tension, see the note in Intent scope. Recording priority order matters more than ever now.
- VAD margins, ambient constants, and confidence threshold (`--min-confidence`, default 0.5) are still untuned guesses pending more real data breadth. No evaluator has been run yet.
- Artifacts: `models/vcm_crnn.tflite`/`.keras` are gitignored -- regenerate with `python -m vcm.train` from `src/` (auto-merges `data_real/`), or copy `.tflite`+`labels.json`+`training_config.json` to the Pi.
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

### Pi bring-up -- DONE (Sep 28), see the "Pi bring-up" section above for what actually happened

Quick reference now that it's set up: `ssh rpi` (key-based, passwordless sudo already configured). To push updated code/model: `git archive --format=tar HEAD -- src | ssh rpi "cd ~/vcm && tar -x"` then `scp models/vcm_crnn.tflite models/labels.json models/training_config.json rpi:~/vcm/models/`. Run on the Pi: `cd ~/vcm/src && source ../.venv/bin/activate && python -m vcm.pi_check` (add `--mic --seconds N` for the mic/fan-noise test, still not done -- see risk above). Live: `python -m vcm.pipeline --source mic` (`--no-wake` to skip the wake window while debugging).

## 8-day critical path (day 1 = Fri Sep 25; demo Sat Oct 3)

- [x] Classifier pipeline, state machine, ambient auto-volume, evaluator harness (built + tested)
- [x] Command-level relabel (closes intent->slot gap), wake-word class, recording tool, dispatcher, player, VAD, live pipeline, Pi setup/diagnostic (built + tested Sep 25-26)
- [x] Mic + speaker + Pi all in hand (Sep 28)
- [x] Record real play_media voice data on the USB mic (267 clips, 246/246 targets hit), verified clean
- [x] Pi bring-up: SSH access, `pi_setup.sh`, `pi_check --mic` all passing on real hardware (Sep 28) -- see Pi bring-up section above for the two real bugs found and fixed (PipeWire routing, USB mic 48kHz-only)
- [x] Retrain with real data -- **result: 33-class capacity wall found, root-caused via controlled ablations, not fixable by tuning** (see Honest current status)
- [ ] **Scope now 7 intents, not just play_media (Sep 28 decision) -- record the rest in fallback-priority order**: `light_on_off` next (simple, hardware-relevant), then `ask_time` (simple, no hardware), then `light_dim_color`/`set_timer`/`set_temperature` (lower priority, first to cut)
- [ ] Build GPIO control path for `light_on_off`/`light_dim_color` in `dispatch.py` (hardware-absent fallback, same pattern as `player.py`), ready for when the LED kit arrives
- [ ] Ask classmate group chat for pooled dataset repo access (SLURP/FSC/Snips: FSC has real voices for lights, volume, heat); merge via `--extra-data` -- **now more urgent** given the capacity wall, this is real data for classes we can't record enough of ourselves in time
- [ ] Fan-on noise comparison (case fans' actual state unconfirmed), live pipeline test on the Pi with real data (`python -m vcm.pipeline --source mic`); tune `--min-confidence`, VAD margins, ambient constants against real audio
- [ ] Source easter-egg songs ("Good Morning", "No"/stage-fright) and 5+ playlist tracks as local files into `music/` (copyrighted -- must be sourced by you; see `player.py` docstring for layout)
- [ ] **Day-5 fallback checkpoint (Tue Sep 29 -- tomorrow)**: revised per the scope change above -- if the classifier isn't handling all 7 intents + reject reliably by then, cut down toward play_media + `light_on_off` + `ask_time` and document the rest as future work, not "cut creative features first"
- [ ] Evaluator sessions, day 5-6 (Sep 29-30): 2-3 people other than you, `python -m vcm.benchmark_harness --mode live --evaluator "<name>" --evaluator-plan`
- [ ] Creative addition, max 1-2, sleep timer first -- only if the day-5 checkpoint passes
- [ ] Rehearsal buffer, Oct 1-2, don't skip

## Open, non-blocking

- Train-from-scratch clarification with professor.
- Whether play_music/media_control stay as two intents in the report vs one `play_media` (they are separate label families in the classifier either way; decide before writing the report, not before building).
- **Confidence threshold recalibration**: a model trained partly on clean synthetic TTS will likely be overconfident on synthetic-style input and miscalibrated on real recordings. Once real data is in, tune `--min-confidence` (pipeline) against a real-audio validation slice rather than assuming 0.5 holds. Directly affects the false-accept rate. The benchmark harness already logs confidence per attempt to support this.
- **Val accuracy will be optimistic**: the val split is random per label, so the same phrase from the same speaker can land in train and val. The honest number is the evaluator benchmark with people who aren't you.
- Ambient auto-volume can't adapt during playback (mic hears the speaker); echo-aware adaptation is out of scope unless time is left.
