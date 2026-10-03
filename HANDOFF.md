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
- **Breadboard/LED kit: arrived Sep 30** — 830-point breadboard, single-color LEDs (on/off) + 1 RGB LED (PWM dim/color) + full resistor range incl. LED-safe 220R/330R, for real `light_on_off`/`light_dim_color` actuation, ~₱403. Covers both light intents with one kit, no second kit needed. Deliberately GPIO-driven, not a smart bulb — a smart bulb would actuate through a vendor cloud API (Tuya/Xiaomi/etc.), which conflicts with the no-cloud constraint; raw GPIO stays fully local. Wiring guide at `GPIO_WIRING.md` (pin numbers match `dispatch.py`'s `GPIOLightController`); once wired, `python -m vcm.gpio_smoke_test` on the Pi cycles every state for a visual check — needs the Pi powered but not network-reachable, so it's not blocked by today's WiFi issue.
- **Confirmed via classmate group chat (Sep 29, 3pm)**: a simulated UI/state for lights and
  temperature is an explicitly acceptable fallback if real hardware isn't ready in time, and
  breadboard + jumper cables (no soldering) is a valid way to wire the LED kit. Matches what's
  already built: `dispatch.py`'s `PrintLightController` fallback covers the "simulate it" case,
  and the kit itself is breadboard-based, not solder-based -- no scope or code change needed,
  just removes a risk (soldering skill/time) that wasn't actually required.
- **Fan noise: checked Oct 1, not a problem.** Fans confirmed always-on (can't be toggled), so there's no true "fans off" baseline, but the "fans on, otherwise quiet" reading is -50 to -51 dBFS -- matches the existing "quiet" placeholder (-50 dBFS) almost exactly. No recalibration needed.

## Pi bring-up (Sep 28, done)

SSH access set up and working: key-based login (`ssh rpi`, alias in `~/.ssh/config` on the dev laptop), passwordless sudo. Host: `rpi-jao` / `jaolacuata@192.168.88.12` (**IP changed Oct 1** after a physical move + WiFi network change + a full reflash -- see `daily_log_report.md` 2026-10-01 for the two-day saga and root cause: a missing cloud-init module silently never created a working NetworkManager WiFi profile despite reporting success). Repo copied via plain `tar` over SSH (not `git archive`, which only grabs committed snapshots -- a lot of working-tree changes weren't committed yet) plus `models/*.tflite`/`labels.json`/`training_config.json` via `scp`. `scripts/pi_setup.sh` now also needs `swig`+`liblgpio-dev` (added) to build the `lgpio` package (gpiozero's PWM backend) from source. `python -m vcm.pi_check --mic` passes every check on real hardware:

- Model load 0.14s (incl. warmup), classify latency mean 18ms (budget 500ms) -- comfortably real-time on a Pi 4.
- Speaker + espeak-ng TTS confirmed audible.
- Mic capture confirmed working, USB mic detected correctly.
- **First real live `vcm.pipeline --source mic` test, Oct 1** -- see `daily_log_report.md` for
  full findings. Working: `--min-confidence 0.15`, most intents recognized reasonably. Wake-word
  live reliability was the single biggest risk after this (~18%, later confirmed as bad as 0/15
  on a later model) -- **resolved 2026-10-03** by recording fresh wake data on the demo Pi
  itself; see the "Pi bring-up" ALSA-fix entry below and `daily_log_report.md` for the full
  investigation.

**Three real hardware problems found and fixed, not hypothetical:**
1. **PipeWire, not raw ALSA, owns audio routing on this OS image.** `~/.asoundrc` is silently ignored. The USB mic's card was the default *sink* (wrong -- it has no real speaker), which is why the 3.5mm-jack speaker (confirmed working via `speaker-test`) produced no sound through code that used the "default" device. Fixed with `wpctl set-default <sink-id>` pointed at the 3.5mm jack (`Built-in Audio Stereo`). **This is a runtime setting, not a config file — if it doesn't survive a reboot, re-run `wpctl status` to find the sink id and `wpctl set-default <id>`.**
2. **The USB mic only supports 48000 Hz capture, not the model's 16000 Hz** (confirmed via `sd.check_input_settings` probing every common rate -- only 48000 succeeded). Fixed properly in code, not worked around: `src/vcm/capture.py` + `audio.resample_integer_ratio()` (dependency-free windowed-sinc decimator, since the Pi deliberately has no scipy/librosa) -- picks the model's rate directly when a mic supports it (e.g. the laptop's mic, unaffected), otherwise captures at the mic's native rate and downsamples. Wired into every capture site (`pi_check`, `record_dataset`, `benchmark_harness` live mode, `pipeline`'s streaming mic loop). Caught and fixed a real bug in the decimator itself during testing (wrong center-tap value distorted the filter) -- verified against `librosa.resample` and an above-Nyquist attenuation test before trusting it. **Moral: this class of "device doesn't support the rate we assumed" bug is real and would have silently broken the live demo on the actual hardware if `pi_check` hadn't been run before demo day.**
3. **(2026-10-03) TTS went silent again after a reboot/reflash -- the #1 fix above didn't persist, and a second, separate issue was hiding behind it.** Confirmed live: `speaker-test`/`aplay` direct to the named ALSA device played fine (speaker powered, AUX-mode, volume physically maxed), but raw `espeak-ng` produced nothing, even after re-forcing the system default with `sudo raspi-config nonint do_audio 1` (equivalent goal to the `wpctl` fix above, different mechanism -- this one *should* survive reboots better, via `amixer cset`, not a PipeWire runtime setting). Two root causes, both real:
   - **PCM volume was at 70% (~-27dB)**, too quiet for this hardware's weak analog output stage. Fixed: `amixer -c 2 sset PCM 100% unmute`, persisted with `sudo alsactl store 2` (**if this doesn't survive a reboot either, re-run both the `amixer` and `alsactl store` commands, same caveat as the `wpctl` fix above**).
   - **`espeak-ng`'s own default audio backend ignores the system ALSA/PipeWire default entirely** -- it exits 0 and produces genuinely nothing audible regardless of what `wpctl`/`raspi-config` point the system default at. The real fix is in code, not system config: `EspeakSpeaker` in `src/vcm/dispatch.py` no longer lets espeak-ng pick its own output -- it pipes explicitly (`espeak-ng --stdout | aplay -D plughw:Headphones,0`) through the named device. Named by driver name (`Headphones`), not a card *number*, since numbers can shift depending on what's plugged in at boot. `aplay` failures are now surfaced (printed), not swallowed -- that exact silence (no error, no sound) cost real debugging time this session.
   - **If this bites again on a fresh reflash**: run `aplay -l` to confirm the headphone jack's card name is still `Headphones`; if it's different, update `alsa_device` in `EspeakSpeaker.__init__` (`src/vcm/dispatch.py`) to match. Test with `espeak-ng --stdout -a 200 'test' | aplay -D plughw:<name>,0` directly over SSH before trusting the full pipeline.

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

Quick reference (current as of Oct 1): `ssh rpi` (key-based, passwordless sudo, IP `192.168.88.12` -- see above if it stops resolving again). To push updated code/model: `tar -cf - --exclude='__pycache__' --exclude='.pytest_cache' src requirements-pi.txt scripts/pi_setup.sh | ssh rpi "cd ~/vcm && tar -xf -"` (plain tar, not `git archive` -- see above) then `scp models/vcm_crnn.tflite models/labels.json models/training_config.json rpi:~/vcm/models/`. Run on the Pi: `cd ~/vcm/src && source ../.venv/bin/activate && python -m vcm.pi_check --mic --seconds 10`. Live: `python -u -m vcm.pipeline --source mic --min-confidence 0.15 2>&1 | tee ~/pipeline_test.log` (`-u` avoids Python output buffering when piped; `--no-wake` to skip the wake window while debugging).

## Plan from here to the demo (revised Sep 29 evening, supersedes the "8-day critical path" below it)

Everything up to tonight (classifier pipeline, Pi bring-up, 267 real play_media clips, the
33-class capacity-wall finding, the 7-intent scope change, and tonight's 1,746-clip external
data merge) is done and logged in `daily_log_report.md` -- this section is what's left,
day by day, to the Oct 3 demo.

### Tonight (Sep 29) -- the decision gate -- DONE, see daily_log_report.md
- [x] Retrain with `external_data/` + `data_real/` merged -- capacity wall confirmed, did not
  clear at 48 classes (5% overall accuracy).
- [x] Root-caused and fixed rather than just working around it: 12 of the 48 labels were
  leftover slots from before the Sep 25-26 schema change (old Fahrenheit temps, old brightness
  %, old timer durations), never cleaned out of `manifest.csv`. Removed them for real (60 rows
  + 60 WAV files deleted, confirmed with the user first). Curated-scope retrain: 27%. Final
  retrain with wake + reject data also merged in: **38% overall, 37 classes, 2,479 clips.**

### Scope decision (locked, Sep 29 daytime)
Keeping all 7 intents for the demo -- no scope cut. 5 of 7 intents plus wake are working
reasonably to well (`set_temperature` 97% intent-acc, `set_timer` 89%, `media_control` 73%,
`play_music` 65%, `ask_time` 64%, `wake` 100% on a small sample). Two are weak
(`light_on_off` 29% intent-acc despite adequate data; `reject` 2% recall, but root-caused to
an unusually hard negative set, not necessarily representative of real background noise/chat)
-- neither looks unfixable, and both are "how does this behave on real audio" questions that
need a live Pi session to actually answer, not more offline dataset work. Cutting either now
would trade assignment scope for a problem that live confidence-threshold/VAD tuning may
resolve on its own. Revisit only if live testing shows they're still broken after tuning.

### Remaining before the demo
- [x] **Record `wake/kuya_jukebox` for real** -- 25 clips, laptop mic, verified clean.
- [x] **Reject-class data** -- `scripts/import_snips_reject.py`, 300 clips imported (see
  scope decision above for how well it's actually working).
- [x] `phrase_list.csv` -- the canonical version never arrived, fixed it directly instead:
  dropped the same 12 stale slots removed from `manifest.csv`, added phrases for every slot
  the Sep 28 scope expansion introduced (`brightness_20/60/other`, all 10 `color_*`,
  `set_timer/10sec+30sec`, `set_temperature/18+22+26`). Ran the phrase-overlap audit this
  unblocked: first pass caught that every new color phrase shared "turn ... the lights" with
  `light_on_off`'s phrases (0.60 word-overlap) -- plausibly contributing to `light_on_off`'s
  weak, scattered confusion in tonight's retrain. Reworded to "change the light color to X" /
  "make the lights X"; audit re-run clean except two pre-existing, accepted design ambiguities
  (`media_control/play` "play" vs `play_music/playlist_general` "play music").
- [x] Build the GPIO control path for `light_on_off`/`light_dim_color` in `dispatch.py` --
  `LightController`/`GPIOLightController`/`PrintLightController`, hardware-absent fallback,
  same pattern as `player.py`'s tone fallback. Also fixed a pre-existing bug found while doing
  this: `color_*`/`brightness_other` slots were silently falling through to "bad slot" since
  the Sep 28 scope expansion added them but `_light_dim` was never updated to parse them.

### Oct 1 (Thu) -- Pi + live testing, mostly done
- [x] Push the retrained model to the Pi, run `pi_check` + a real `pipeline --source mic`
  session. Unreachable Sep 29-30 (two days lost, see `daily_log_report.md` for the WiFi root
  cause and fix), finally online Oct 1. First-ever live test done -- see findings logged above
  and in the daily log. **`--min-confidence 0.15` settled on** after empirical testing.
- [x] Fan-on noise comparison -- fans always-on, no toggle, but clean reading taken: -50 to
  -51 dBFS, matches the "quiet" placeholder almost exactly. No action needed.
- [x] Added a wake-word audio cue (beep -> TTS, since a second concurrent audio stream crashed
  ALSA) since there's no screen on demo day. Text changed 2026-10-03 from "mm-hmm" (too quiet/
  mumbled to reliably notice) to a clearly-enunciated "Yes?", alongside a TTS amplitude bump
  (100->150) -- see the ALSA fix in the "Pi bring-up" section above.
- [x] **Wake-word live reliability, resolved 2026-10-03**: was the single biggest open risk
  (~18% live vs. 69-75% offline on Oct 1-2, then 0/15 live on Oct 2 night's 19-intent model
  despite similar offline numbers -- a real train/live mic mismatch, not a threshold issue).
  Fixed by recording 30 fresh wake clips directly on the demo Pi's own mic, in the actual demo
  room (`data_real_pi_wake/`), and retraining. Live recall after the fix: effectively 100%
  across 40+ consecutive attempts in the same session. See `daily_log_report.md`
  2026-10-03 (overnight) for the full investigation.
- [ ] VAD margins / ambient constants beyond `--min-confidence` -- not deeply tuned yet, lower
  priority than wake reliability.
- [ ] Wire up the GPIO code for real -- kit arrived Sep 30, still not physically wired.
- [ ] Recruit 2-3 evaluators and run the benchmark session -- **not started at all**, and this
  is a professor-quoted grading requirement, not optional polish. Start recruiting early on
  Oct 2 since other people's availability isn't something you control.
- [ ] Source the easter-egg songs + 5+ playlist tracks into `music/` -- still placeholder tones.
- [ ] **Commit today's substantial uncommitted work to git** -- real risk sitting in the
  working tree right now (GPIO code, benchmark harness fix, pipeline fixes, manifest/
  phrase_list cleanup, the retrained model). Do this before anything else touches the repo.

### Oct 2 (Fri) -- the real full work day, tight but doable
- [ ] Commit everything first (5 min, pure risk mitigation).
- [ ] One more wake-word data round: bigger, more deliberately varied (distance, pacing,
  pitch) than the 65-clip batch, then retrain. Highest-leverage remaining lever on the biggest
  open risk.
- [ ] Recruit evaluators *early in the day* -- their schedule, not yours, is the constraint.
- [ ] Wire the breadboard (independent of everything else, ~15-30 min, code/guide ready).
- [ ] Source easter-egg/playlist music files (independent, whenever there's a spare moment).
- [ ] Run the actual evaluator benchmark session once evaluators + a stable model are ready.
- [ ] Fix whatever the evaluator session turns up.
- [ ] At least one full rehearsal run-through -- not optional given how much changed Oct 1.
- [ ] Creative addition (sleep timer first) only if everything above is actually solid --
  do not trade core reliability for this.

### Oct 3 (Sat) -- demo day
- [ ] Final morning sanity check, then demo.
- [ ] If wake word is still unreliable: coach the "just try 2-3 times" fallback rather than
  betting the demo on first-attempt reliability -- realistic even for commercial assistants.

### Oct 3 (Sat) overnight/morning -- live-tuning pass, demo script guidance
Decided: no more evaluator recruitment, no breadboard wiring -- the web light
simulator (`web_simulator/`, merged to master) is the confirmed-working
fallback, class-approved. Live-tested on the Pi and fixed what came up:
- `--min-confidence` raised **0.15 -> 0.35** (the 0.15 figure above was
  tuned Oct 1 against an older 7-intent model, never revalidated after the
  Oct 2-3 expansion to 19 commands/47 classes; live testing showed 0.15 let
  low-confidence guesses like 0.20-0.38 fire as real actions).
- Ghost playlists (`playlist_general`/`jazz`/`workout` -- classifier labels
  with no real music files) no longer fall through to a placeholder sine
  tone if the classifier mis-picks one; `pipeline.py` only hands
  `playlist_chill`/`playlist_focus` to the state machine, the other three
  are cleanly declined ("the X playlist isn't available").
- Easter-egg placeholder tone (no `music/easter/*.mp3` sourced) now
  self-stops after 2.5s instead of droning until someone says "stop".
- **`media_control/volume_down` ("quieter") is a known, unfixed live-mic
  reliability gap** -- confirmed NOT a model-bias issue (validation
  confusion matrix shows volume_down is fine, 76% correct, barely confused
  with volume_up) -- it's the same class of live-mic/acoustic domain
  mismatch that caused the wake-word problem, just not fixed the same way
  (would need real Pi-mic volume_down recordings + a full ~2hr retrain,
  ruled out tonight on a tight time budget). **Demo script guidance**: say
  **"volume down"** (literal, untested live but structurally different from
  the known-bad phrase below); avoid **"turn it down"** (live-tested,
  always misfired as `volume_up`); **"quieter"** is a fallback (inconsistent
  -- worked a few times, then reverted).
- **`call/none` is confused with `wake/kuya_jukebox`, confirmed via two
  independent tests** (a controlled per-utterance probability check using
  `vcm.debug_topk`, and a live pipeline retest with an explicit pause after
  the wake word): saying "Kuya Jukebox, [pause], call" repeatedly produced
  five straight `wake` detections with zero commands in between -- the
  model's own top-1 guess for "call" kept landing on `wake/kuya_jukebox`
  itself, not a near-miss confusion with some other command. In a clean,
  isolated capture (no adjacent wake phrase) `call/none` scored 0.972 --
  very recognizable on its own -- so this is specifically a wake/call
  acoustic overlap in the live-mic domain, same root-cause class as
  `volume_down` (needs real data + retrain to actually fix, ruled out
  tonight). **Demo risk**: don't rely on "call" working first try; if it
  keeps re-triggering wake instead of firing, that's this known issue, not
  a user error. `message` is unaffected -- confirmed working multiple times
  tonight (`"message"` / `"send a message"`, 0.80-0.88 confidence).

### Honest progress assessment (Oct 1 night)
~55-60% complete, not higher. A working model isn't a finished assignment: evaluator testing
(required), breadboard wiring, real music files, rehearsal, and git hygiene are all
not-started-at-all, not just in-progress, on top of the wake-word reliability risk.

## 8-day critical path (original, day 1 = Fri Sep 25; kept for the report's process narrative)

- [x] Classifier pipeline, state machine, ambient auto-volume, evaluator harness (built + tested)
- [x] Command-level relabel (closes intent->slot gap), wake-word class, recording tool, dispatcher, player, VAD, live pipeline, Pi setup/diagnostic (built + tested Sep 25-26)
- [x] Mic + speaker + Pi all in hand (Sep 28)
- [x] Record real play_media voice data on the USB mic (267 clips, 246/246 targets hit), verified clean
- [x] Pi bring-up: SSH access, `pi_setup.sh`, `pi_check --mic` all passing on real hardware (Sep 28) -- see Pi bring-up section above for the two real bugs found and fixed (PipeWire routing, USB mic 48kHz-only)
- [x] Retrain with real data -- **result: 33-class capacity wall found, root-caused via controlled ablations, not fixable by tuning** (see Honest current status)
- [x] Scope expanded to 7 intents (Sep 28 decision)
- [x] External data merge: 1,746 real clips across the 6 non-play_media intents (Sep 29) -- see daily_log_report.md
- [ ] Everything else -- see the day-by-day plan above, which is now the live source of truth for what's left.

## Open, non-blocking

- Train-from-scratch clarification with professor.
- Whether play_music/media_control stay as two intents in the report vs one `play_media` (they are separate label families in the classifier either way; decide before writing the report, not before building).
- **Confidence threshold recalibration**: a model trained partly on clean synthetic TTS will likely be overconfident on synthetic-style input and miscalibrated on real recordings. Once real data is in, tune `--min-confidence` (pipeline) against a real-audio validation slice rather than assuming 0.5 holds. Directly affects the false-accept rate. The benchmark harness already logs confidence per attempt to support this.
- **Val accuracy will be optimistic**: the val split is random per label, so the same phrase from the same speaker can land in train and val. The honest number is the evaluator benchmark with people who aren't you.
- Ambient auto-volume can't adapt during playback (mic hears the speaker); echo-aware adaptation is out of scope unless time is left.
