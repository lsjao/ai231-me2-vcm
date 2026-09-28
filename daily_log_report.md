# Daily Log Report — ME2 Voice Command Model

Chronological log of what actually happened: problems hit, how they were diagnosed, and how they
were fixed. Written for the assignment's process/methodology section — this is the story, not the
current status. For current state and the plan going forward, see `HANDOFF.md` instead.

**Update this whenever something major happens or a problem is hit — append, don't rewrite history.**

---

## 2026-09-25 (Day 1 of 8, demo Oct 3)

### Handoff correction #1
The handoff describing this project claimed four Python files (`play_music_state_machine.py`,
`ambient_and_beatsync_skeleton.py`, `smoke_test_train.py`, `benchmark_harness.py`) and a
`personal_play_music/` dataset slice already existed, bundled in `vcm_dataset_scaffold.zip`.
On unzipping, **none of it existed** — the zip only had `phrase_list.csv`, `manifest.csv`,
`dataset/`, `README.md`. Filesystem search turned up no copies elsewhere. Treated everything as
not built and corrected `HANDOFF.md` to reflect actual repo state, not the aspirational draft.

### Classifier pipeline built from scratch
Built `src/vcm/audio.py` (feature extraction), `data.py` (manifest/split/augmentation),
`model.py` (small CRNN, ~43K params), `train.py`. Two real bugs found and fixed while getting
the first training run to complete:
- **Label dtype mismatch** (`int32` vs declared `int64`) inside the `tf.data` pipeline — this
  didn't raise a clean error, it caused a multi-minute hang under `model.fit()` instead.
- **`Bidirectional(GRU)` produced dynamic `TensorListReserve` ops** that the standard TFLite
  converter can't lower without the heavyweight Flex delegate. Fixed with `unroll=True` (sequence
  length is fixed anyway), producing a plain-builtin-ops `.tflite` file — important since the Pi
  was never going to get the Flex delegate working in this timeframe.

First real training run: 24% val accuracy on 201 synthetic-only clips (8-way intent), chance is
~12.5% — learning something, but collapsing toward majority classes. Expected, not a bug: matches
the handoff's own warning that synthetic-only training fails on real voices.

### State machine, ambient auto-volume, benchmark harness built
`play_music_state_machine.py` (no-repeat playlist queue, transport, easter eggs, duck/unduck
hooks), `ambient_volume.py` (asymmetric-EMA noise floor tracker), `benchmark_harness.py`
(evaluator-logging CLI, live + replay modes). All built with tests alongside, verified against
the real trained model in replay mode as smoke tests (not real evaluator data).

### Second correction: found a missed addendum
A sanity-check pass re-read `handoff.txt` (untracked, gitignored) and found an "Addendum Part 2"
had been appended to it after the initial read — a concrete recording protocol, easter-egg/
playlist audio sourcing tasks, a confidence-recalibration risk, a retries-to-success benchmark
metric, a concrete evaluator plan, and a day-5 fallback checkpoint. It went unnoticed because a
gitignored file's changes don't show up in `git status`. Merged into `HANDOFF.md`; going forward,
`handoff.txt` is retired in favor of editing `HANDOFF.md` directly, specifically to prevent this
exact silent-desync failure mode from happening again.

### Sanity-check pass: real bugs found
- Unused import (pyflakes).
- **Real deployment bug**: `benchmark_harness.py` hard-imported full `tensorflow`, but the Pi was
  only ever going to have the lightweight `tflite_runtime`/`ai_edge_litert` installed
  (`requirements-pi.txt`) — would have failed the first time it ran on the Pi. Fixed with a
  try/fallback import chain, later centralized into `src/vcm/classifier.py`.
- Removed an empty, unused `scripts/` directory left over from initial scaffolding.

---

## 2026-09-25/26 — command-level relabel, recording tooling, live pipeline (all built pre-hardware)

### Decision: classifier predicts command labels, not just intent
Discovered while building the state machine: an intent-only classifier (`media_control`,
`play_music`, ...) can't tell `media_control/next` from `media_control/pause` — both are just
"media_control." Decided (with the user) to retrain on `intent/slot` command labels directly
(33+ classes, `reject` slots collapsed to one label) rather than build a second-stage slot
classifier. One model, and the state machine gets the exact action straight from the label.

### Decision: wake word as a trained class
Added `wake/kuya_jukebox` as another label in the same classifier, rather than a separate wake-word
detector. Commands only act while a wake window is open (5s); opening it ducks the music to ~5%
(a classmate-sourced fix for music-degrades-recognition), matching the state machine's existing
`duck()`/`unduck()` hooks.

### Built: recording tool, dispatcher, player, VAD, live pipeline, Pi tooling
`record_dataset.py` (guided recording sessions), `dispatch.py` (routes commands to the state
machine or simple handlers — ask_time, real background timers, simulated lights/temperature,
TTS via espeak-ng), `player.py` (local file playback with a tone fallback for un-sourced music),
`vad.py` (streaming energy endpointer), `pipeline.py` (mic → VAD → classifier → wake window →
dispatch → player), `pi_check.py` + `scripts/pi_setup.sh` (Pi diagnostic/setup, untestable
without hardware at this point). 128 tests passing at this point. Retrained on the relabeled
33-command synthetic set as a pipeline smoke test only (still no real voice data).

### Recording tool bug: silent room saved as "speech"
A dry run in a silent room saved two takes as if they were spoken phrases. Root cause:
`trim_to_speech` is relative to a clip's own noise floor, so it "finds" speech in room noise
that just happens to wiggle a bit. Fixed by adding `audio.speech_level_ok()` — requires the
loudest frame to be absolutely loud, well above the clip's own floor, *and* sustained for
150ms+ (so a click or chair creak doesn't count). Also added retry caps (3 misses skips the
prompt, 6 in a row stops entirely) so an unattended or muted mic can't loop forever. Verified
against the real laptop mic: zero false saves across multiple silent-room runs afterward.

### Windows environment issues
- **Wrong Python on PATH**: a bare `python`/`py` in some terminals resolved to a Python with none
  of the project's packages installed (`ModuleNotFoundError: soundfile`), while a different,
  correctly-provisioned Python existed elsewhere on the machine. Fixed with `run.cmd`, a launcher
  batch file at the repo root that pins the correct interpreter regardless of which terminal or
  directory it's run from.
- **Relative default paths broke when run from a different directory**: `record_dataset.py`
  defaulted to `phrase_list.csv` (relative), which failed with `FileNotFoundError` when launched
  via `run.cmd` from `src/`. Anchored every CLI's default paths to the project root
  (`src/vcm/paths.py`) regardless of current working directory.

### Phone/friend recording importer built
Added `import_recording.py`: prints a numbered script, the user (or a friend) reads it into one
voice memo (m4a/mp3/wav — PyAV added for m4a/aac decoding), and the importer splits it into
labeled takes using the same VAD as the live pipeline. Refuses and saves nothing if the heard
utterance count doesn't match the script (one miscount would mislabel every later take).
End-to-end tested with a real `.m4a` file through the actual CLI.

---

## 2026-09-28 — hardware day: mic, speaker, Pi bring-up

### Raspberry Pi Imager walkthrough
Walked through installing Raspberry Pi Imager, writing Raspberry Pi OS (64-bit) with the
advanced-options gear icon (hostname, SSH, Wi-Fi, locale) so the Pi would be headless-ready with
no monitor needed.

### SSH connectivity: a long diagnostic chain
1. **Hostname wouldn't resolve** (`raspberrypi-jao.local`) — Windows mDNS is unreliable from
   both Git Bash and PowerShell. Worked around by scanning the local subnet directly.
2. **Chased the wrong device.** A network scan turned up `192.168.86.19`, which looked plausible
   and was reachable, but never accepted an SSH connection no matter what was tried. It had a
   "locally administered" (randomized-looking) MAC address — consistent with a phone or laptop's
   Wi-Fi privacy MAC, not the Pi. The real Pi (`192.168.86.4`, MAC prefix `d8:3a:dd`, a real
   Raspberry Pi Foundation OUI) didn't even appear on the network until several boot cycles later.
   Lesson: cross-check MAC vendor prefixes before trusting a scan result, don't just take
   "responds to ping" as identification.
3. **No boot at all** (solid red LED, zero green activity) — traced to an underpowered/incorrect
   power supply. Fixed by swapping to the Pi's actual rated supply.
4. **SD card corrupted by an unsafe removal.** The card was pulled directly from the USB adapter
   without ejecting, which (on this reader) left cached writes unflushed and the FAT32 boot
   partition flagged "Full Repair Needed" by Windows. This is very likely why a manually-added
   `ssh` marker file didn't survive to actually reach the Pi. Fixed going forward by forcing a
   proper flush via the Shell "Eject" verb (`Dismount-Volume` wasn't available on this system).
5. **cloud-init's SSH-enable step never ran.** The card used cloud-init (`user-data`), which
   looked syntactically correct (`ssh_pwauth: true`, explicit `systemctl enable --now ssh`) but
   had a `packages: [avahi-daemon]` step ahead of it in execution order — if that apt install
   stalls (flaky first-boot network), SSH enabling never gets reached. Matched the observed
   symptom exactly (disk activity for a while, then quiet, no SSH ever).
6. **Clean re-image, this time with the SSH public key pasted directly into the Imager**
   (sidesteps cloud-init's package-install ordering issue entirely, and sidesteps password
   typos). Host key mismatch after re-imaging was expected (fresh OS = fresh host key) and
   cleared with `ssh-keygen -R` + `ssh-keyscan`.
7. **`sudo` demanded a password** despite key-based SSH login working — the account password
   ended up not being blank as first assumed. Got the real password from the user, verified it
   with a scripted `sudo -S` check, then set up passwordless `sudo` via `/etc/sudoers.d/` so
   future commands don't need it re-entered.

End state: full key-based SSH access (`ssh rpi` alias), passwordless sudo, confirmed working.

### Pi bring-up
Copied the repo via `git archive | ssh ... tar -x` (tracked files only) plus the trained
`.tflite`/`labels.json`/`training_config.json` via `scp` (gitignored, not in the archive).
`scripts/pi_setup.sh` ran clean on Debian 13 (trixie) / Python 3.13 / aarch64 — `ai-edge-litert`
had a prebuilt wheel for that exact combination, no fallback needed.

### Two real hardware bugs found by `pi_check`, not hypothetical
1. **Wrong default audio device.** This OS image runs PipeWire, which owns audio routing and
   ignores `~/.asoundrc` entirely. The USB mic's card was set as the default *output* sink
   (it has no real speaker), which is why TTS produced no sound even though a raw tone test
   worked on a different, explicitly-specified device. Diagnosed by testing each ALSA card
   directly with `speaker-test`, confirmed the 3.5mm jack was the real speaker, then fixed with
   `wpctl set-default <sink-id>` — a PipeWire-native tool, not an ALSA config file.
2. **The USB mic only supports 48000 Hz capture, not the model's 16000 Hz.** Confirmed by probing
   every common rate with `sd.check_input_settings`; only 48000 succeeded. This would have
   silently broken live capture on demo day if `pi_check` hadn't caught it first. Fixed properly:
   added `audio.resample_integer_ratio()` (dependency-free windowed-sinc decimator — the Pi
   deliberately has neither scipy nor librosa) and `src/vcm/capture.py` (tries the model's rate
   first, falls back to the device's native rate + resampling), wired into every capture site
   (`pi_check`, `record_dataset`, `benchmark_harness` live mode, `pipeline`'s streaming mic loop).
   **Caught a real bug in the decimator itself during testing**: the filter's center tap was
   `1.0` instead of `1/factor`, which distorted the filter shape enough that stopband
   attenuation didn't improve even after tripling the tap count. Root-caused by testing against
   `librosa.resample` and an above-Nyquist attenuation check, then fixed and reverified.

`pi_check --mic` now passes every check on real hardware: model load 0.14s, classify latency
mean 17.5ms / p95 17.6ms (budget 500ms), speaker/TTS confirmed audible, mic capture confirmed
working with real ambient noise floor readings (p50 -42.9 dBFS quiet room).

### First live pipeline test on real hardware
Ran `vcm.pipeline --source mic --no-wake --min-confidence 0` and spoke several commands. Result:
the **full mechanical chain works** — mic capture, 48kHz→16kHz resampling, voice-activity
segmentation (5 distinct utterances correctly split from real speech), classification (18–45ms
per utterance), dispatch to the state machine, and a real spoken-back response, all on actual
hardware with a real voice for the first time. Every utterance was misclassified as the same
label (`play_music/playlist_jazz`, confidence ~0.03 — close to 1-in-33, essentially a random
guess) — expected and not a new problem: this model has still never heard the user's actual
voice, only synthetic TTS. The point of this test was proving the plumbing works, which it does.

**Still open at end of day:** dual-fan noise comparison unconfirmed (no software fan control
detected at all — case fans, if wired, appear to be simple always-on units), real voice
recording + retraining not yet done on this hardware, no evaluators run yet.

---

## 2026-09-28, continued — real recording, a real bug, and a capacity wall

### The recording session: three real bugs found, none of them the mic
Moved the USB mic from the Pi to the laptop (simpler interactive recording UX than over SSH)
and hit a chain of real, reproducible bugs before it worked:
1. **`beep()` used sounddevice's "default" output device**, which plugging in the USB mic
   (it exposes a fake playback endpoint) silently changed out from under Windows — a loud,
   long test tone confirmed the *device* routing worked, but that didn't prove the actual
   short/quiet beep was audible, and it wasn't. Fixed by making the device explicit
   (`--device-out`) and by making the beep itself louder/longer, verified by the user
   directly confirming *that exact* beep, not a proxy tone.
2. **Python's stdout was buffered**, so when a long-running `--auto` session got moved to
   this tool's background execution (120s foreground timeout), the prompt text stopped
   updating live even though the process kept running and beeping — same class of bug as
   the pipeline test earlier. Fixed by always using `-u` (unbuffered) for interactive runs,
   and by having the user run long sessions in their own independent terminal window instead
   of through this tool, sidestepping the 120s live-view cutoff entirely.
3. **`mv` silently failed with a Windows file-lock "Permission denied"** while trying to set
   up a synthetic-only control run — the background task didn't surface the failure clearly,
   and a "control" run accidentally trained on the full merged dataset instead. Caught by
   checking the run's own `total=` log line rather than trusting the setup steps succeeded.

Once fixed: recorded 267 real play_media takes (246 on the USB mic hitting all 246/246
per-slot targets exactly, plus 21 earlier laptop-mic takes). Verified clean: no clipping,
no suspiciously-quiet takes, peak levels well-distributed (median 0.15).

### Retrain: accuracy got worse, not better — a real investigation, not just "needs more data"
Retraining with the real data merged in gave ~2-3% val accuracy — worse than the prior
synthetic-only run, with `media_control`/`play_music` (the categories with the *most* real
data) at 0% while tiny synthetic-only categories showed noisy partial success. That pattern
(large-data classes failing, tiny classes not) doesn't fit "just needs more data," so it got
investigated properly rather than accepted as expected:
- Checked training's own trajectory: loss stuck near `ln(33)` (pure-guessing level) for the
  *entire* run, even on the training set — a 43K-parameter model failing to fit even 160-374
  examples is not normal; something was actively preventing learning, not just limiting it.
- Ruled out, one at a time, via clean controlled ablations (each verified by checking the
  run's own `total=`/`extra data` log line, after the `mv`-failure lesson above):
  - **Real-vs-synthetic data mismatch** (duration/padding differences) — ruled out: a
    genuinely synthetic-only run (empty `--extra-data` manifest, not a moved directory)
    showed the *identical* stuck-at-chance pattern.
  - **class_weight instability** (33 uneven classes -> up to 11.5x weight ratio) — tested
    removing it entirely: marginal improvement only, not the cause.
  - **Learning rate / vanishing gradients through the unrolled 301-step GRU** — tested
    lower LR + gradient clipping: no meaningful change.
  - **Label/feature misalignment in the tf.data pipeline** — directly verified by comparing
    each batch's labels against their source rows in an unshuffled dataset: correctly
    aligned, not a bug.
  - **EarlyStopping firing too early** — was real (monitoring `val_accuracy` on a 94-clip val
    set is too coarse, ~1%/sample), fixed (switched to `val_loss`, more patience), but alone
    didn't fix the core problem either.
- **Decisive test**: retrained the identical architecture and data, but with the *original*
  8-way intent labels instead of 33-way commands. Clean, healthy learning — accuracy climbing
  steadily to ~30-36% (well above chance), loss dropping meaningfully. Same model, same data,
  only the label granularity changed.
- **Conclusion**: not a code bug anywhere. 33 classes at ~5 examples/class on average (many
  as low as 2-3) is a genuine from-scratch learning capacity wall for this model size and
  data volume — 8 classes at ~20 examples/class is learnable, 33 at ~5 isn't. This took real
  investigation to establish rather than assume, since the alternative (silently accepting a
  bad number as "expected, small dataset") would have hidden a real fixable bug if one had
  existed, and conversely would have wasted remaining days chasing hyperparameters if it were
  purely a data problem, which it turned out to be.

### Scope change: all 7 intents now in individual scope, not just play_media
After reviewing the class group chat, learned classmates are building single models covering
the full intent list rather than specializing in one — the professor's "respond to any
person" quote refers to the full list, not a specialty subset. This **directly conflicts**
with the capacity-wall finding above: more intents now need real data right as the ceiling on
how many classes this approach can learn well got confirmed. Revised the day-5/6 fallback
accordingly: if the full 7-intent + reject classifier isn't reliable by the checkpoint
(tomorrow, Sep 29), cut down toward play_media + the simplest 1-2 other intents
(`light_on_off`, `ask_time`) and document the rest as future work, rather than cutting
creative play_media features first. Also: a new electronic component kit (breadboard, LEDs,
RGB LED, resistors, ~₱403) was sourced for real GPIO-driven `light_on_off`/`light_dim_color`
actuation, deliberately not a smart bulb (avoids a vendor cloud API dependency that would
conflict with the no-cloud constraint). Not yet in hand.
