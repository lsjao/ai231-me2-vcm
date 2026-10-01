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

---

## 2026-09-29 — external data merge: 1,746 real clips for the 6 non-play_media intents

### The big discovery, restated for the record
Yesterday's capacity-wall finding (33 classes at ~5 examples/class is a genuine from-scratch
learning limit, confirmed via controlled ablations, not a code bug) directly collided with
today's scope expansion to all 7 intents. Tonight's work is the response to that collision:
instead of cutting scope, find enough real data to actually clear the wall.

### Two external sources merged, after a missing-file blocker and real bugs found mid-import
A fresh Claude conversation had planned a merge of two raw data sources (Mark's classmate
dataset — 150 speakers, real recordings; Snips SLU — a public smart-lighting speech corpus)
and handed over a detailed spec to execute. Two real blockers surfaced immediately:
- **The spec referenced a "canonical phrase_list.csv" that never actually arrived in the
  paste** — cut straight from "replace with the version below" to describing a diff, no file
  content. Flagged rather than fabricated; turned out not to matter for this pass (see below).
- **Mark's dataset's own manifest.csv had real transcripts per file** (e.g. "Wake me up at
  8 AM"), contradicting the spec's assumption that none existed — used the real transcripts
  instead of approximating with our canonical phrasing, confirmed with the user first.

Before writing anything, caught two consequential design issues the spec hadn't accounted
for and raised them rather than proceeding blind:
- **Repo-size risk**: the spec said copy into git-tracked `dataset/`; the actual available
  volume (Mark alone: 18,375 matching rows, ~590/slot) would have added an estimated 1GB+ to
  a git repository. Redirected to a new gitignored `external_data/` root, mirroring the
  existing `data_real/` pattern and reusing the same `--extra-data` training mechanism —
  confirmed with the user rather than assumed.
- **Data volume vs. benefit**: ~590 real examples per slot is far more than needed and would
  meaningfully slow every training epoch (18,375 examples at batch_size 16 is ~1,150 steps/
  epoch instead of ~24) for no clear benefit over a smaller sample. Subsampled to 60/slot with
  the user's confirmation — still 12-24x the prior per-class data.

Real bugs hit and fixed during the actual import, not just planning:
- **`shutil.copy2` crashed with `WinError 3`** partway through Source A. Root cause: Mark's
  own manifest references ~5,643 rows (the entire `ALARM`/`WEATHER`/`CALL`/`MESSAGE`/
  `CREATE_REMINDER`/`LIST_REMINDERS` categories intended as reject-class hard negatives) whose
  audio was never actually included in the shared `OptionB` folder — his README's "844
  excluded/flagged" note undersold the real gap. Confirmed by checking `os.path.exists` on the
  specific failing path before assuming a code bug. Made the importer skip-and-count missing
  source files instead of crashing, rather than silently catching everything broadly.
- **Snips' `datasets.Audio` auto-decode required `torchcodec`**, a new and fairly heavy
  dependency. Avoided adding it: cast the column to `decode=False`, confirmed the raw bytes
  were plain WAV (`RIFF...WAVE` header), and read them directly with `soundfile` (already a
  project dependency) instead.
- **A shell diagnostic (`cut -d,`) produced garbled-looking counts** right after the Snips
  import, which looked like real data corruption at first glance. Root cause: `cut` doesn't
  respect CSV quoting, and some phrase text contains commas. Re-verified with Python's `csv`
  module before concluding anything was wrong — it wasn't.

Snips required keyword classification (its rows have no pre-built intent/slot labels, only
free text). Spot-checked actual matched phrases before trusting the classifier — quality was
good overall; found one real, minor false-positive class ("...to fifty. on the patio" matched
the "on" keyword despite being a brightness command, not on/off) and accepted it as a known,
small-scale limitation of keyword matching rather than a blocker.

### Result
1,746 real clips added across the 6 non-play_media intents (`light_dim_color` 726,
`media_control` 360, `light_on_off` 240, `set_temperature`/`set_timer` 180 each, `ask_time`
60), 115MB, gitignored. Per-slot counts now 60-120, versus ~5 before — a 12-24x increase for
exactly the intents hit hardest by the capacity wall. **Confirmed training needs none of the
still-missing canonical `phrase_list.csv`** — `train.py` derives labels from manifest data,
not that file, so a retrain against this expanded dataset can happen immediately; the
canonical file is still needed for the phrase-overlap audit and to keep `record_dataset.py`'s
prompts in sync with the new slots, but isn't blocking the one experiment that actually
matters most right now: does this much real data clear the capacity wall or not.

**Not yet done**: the retrain against this new data (next), the phrase-overlap audit, GPIO
code, wake-word (`kuya jukebox`) real recordings (still zero), and reject-class data now that
Mark's out-of-scope negatives turned out to be unavailable -- worth noting Snips' 3,472
unmatched rows (didn't match any of our keyword rules) are themselves naturally-occurring
smart-home-adjacent speech that isn't one of our commands, a plausible substitute source for
reject-class negatives, not yet acted on.

## 2026-09-29 (late night) — retrain result: capacity wall did not clear

Ran `python -m vcm.train` with `data_real/` + `external_data/` merged in (2,214 clips total,
48 command classes, 1,770 train / 444 val). This was the test that mattered: does 12-24x more
real data per class (up from ~5/class to ~37/class average) clear the capacity wall found last
night.

### Result: no, not in aggregate
Overall command accuracy: 5% (444 val clips), effectively unchanged from before the merge.
But the failure is not uniform across intents, which rules out a few simpler explanations:

- `light_dim_color` (726 real clips): 77% intent-level accuracy, 11% slot-level -- the model
  finds the right intent reliably but can't separate its 13 slots.
- `set_temperature` (180 real clips): 66% intent-level, 17% slot-level -- same pattern.
- `ask_time`, `light_on_off`, `play_music`, `reject`, `set_timer`: 0% at both intent and slot
  level -- total collapse.
- `media_control` (267 real play_media clips from the earlier recording session + 360 more
  from tonight's merge): only 17% intent-level, 1% slot-level -- **worse than play_media was
  getting on its own before tonight's merge**, despite having more data now than before.

### Why this changes the working theory
The original capacity-wall finding (last night) was framed as a data-volume problem: ~5
clips/class average couldn't separate 33 classes even though the same data separated 8
intents cleanly. Going to ~37 clips/class average should have been a decisive test of that
theory. It wasn't uniformly fixed -- and `media_control` getting worse with more data added is
the clearest signal that this isn't purely about volume per class anymore. The more likely
explanation now: a single flat classifier over 48 similar-sounding classes has larger, more
data-rich intents (`light_dim_color` at 726 clips, `media_control` at 627) crowding out
smaller ones in a shared decision boundary, independent of whether any individual class has
"enough" data in isolation.

This also invalidates the Sep 28 fallback plan (play_media + `light_on_off` + `ask_time`) --
two of those three are exactly the intents that collapsed to 0% here.

### Next step (in progress)
Test whether a smaller, curated label set (drop thin synthetic-only slots, collapse
near-duplicate slots) does better than raw data volume did, as a fast, cheap experiment before
considering a bigger architecture change (e.g. two-stage intent-then-slot classification,
which was explicitly ruled out as a design choice on Sep 25-26 but may need revisiting if this
doesn't work either).

## 2026-09-29 (daytime) — curated-scope experiment confirms it, root cause found and fixed

Added `--drop-labels` to `train.py` (repeatable, excludes a command label's rows before
building the label list) and used it to test the curated-scope theory without touching
tracked data yet. Dropped 12 labels: `light_dim_color/brightness_{25,50,75}`,
`set_temperature/{60,65,70,75,80}`, `set_timer/{5,10,15,30}min` -- 48 classes down to 36,
2,214 rows down to 2,154.

**Result: 27% overall command accuracy, up from 5%.** `media_control` alone went from 17% to
83% intent-accuracy. This confirmed the theory and, combined with root-causing *why* those 12
labels existed, turned a "maybe try fewer classes" experiment into a real fix: those 12 slots
are leftovers from *before* the Sep 25-26 schema change (old Fahrenheit temperatures, old
brightness percentages, old timer durations) that the synthetic TTS manifest was never
regenerated to drop. Nothing downstream (`dispatch.py`, `phrase_list.csv`, the real
recordings) has used those values since Sep 26 -- they were pure label cruft, silently
crowding out the 36 classes that actually matter, sitting undetected in `manifest.csv` for
three days.

Removed them for real (with explicit confirmation first, since deleting tracked WAV files is
irreversible-ish): 60 rows dropped from `manifest.csv`, 60 WAV files + 12 now-empty slot
directories deleted from `dataset/`. `--drop-labels` stays in `train.py` as a general-purpose
option but is no longer needed for this specific cleanup.

### Also done today (parallel to the above)
- **Real `wake/kuya_jukebox` recordings**: 25 clips, laptop mic, `record_dataset.py --auto`.
  Verified clean (no silence, no clipping, peak median 0.145) -- wake word had zero real
  examples before this.
- **GPIO control path** built in `dispatch.py`: `LightController` protocol, `GPIOLightController`
  (gpiozero, one on/off LED + one RGB PWM LED, BCM pins provisional pending actual wiring) and
  `PrintLightController` fallback (auto-selected when gpiozero/hardware isn't present), same
  pattern as `player.py`'s tone fallback. Caught and fixed a real pre-existing bug while doing
  this: `_light_dim` only ever parsed `brightness_<N>`, so every `color_*` slot added by
  tonight's Snips/Mark merge (Sep 28 scope expansion) was silently falling through to "bad
  slot" -- never caught because no test exercised a color slot until now. Added tests for
  `color_red` and `brightness_other`; all 167 tests pass.
- **Reject-class negatives**: new `scripts/import_snips_reject.py` imports Snips rows the
  keyword classifier couldn't match (3,472 available) as `reject/near_domain`, capped at 300.
- **Classmate group chat (3pm)** confirmed a simulated UI is an acceptable fallback for
  light/temperature hardware, and breadboard + jumper cables (no soldering) is fine for the
  LED kit -- matches what's already built, removes a false worry rather than changing scope.
  Saved to memory (`project_hardware_fallback_acceptable`).

### Final retrain of the day: manifest cleaned + wake + reject data all merged
`total=2479 train=1982 val=497 classes=37`. **38% overall command accuracy**, up from 27%.

- `wake/kuya_jukebox`: 100% precision/recall (n=5, small sample but zero confusion with the
  watched labels)
- Strong: `set_temperature` 97% intent-acc, `set_timer` 89%, `media_control` 73%, `play_music`
  65%, `ask_time` 64% (up from 0% two retrains ago)
- **`reject` collapsed to 2% recall** despite the 300 new clips. Investigated via the
  confusion matrix rather than left as a number: not scattered -- roughly half of reject's 63
  val clips are predicted as `light_dim_color`, another third as `light_on_off`. Root cause:
  the 300 Snips clips are lighting-domain speech the keyword classifier couldn't cleanly match
  (deliberately hard negatives, per the design intent noted when they were imported), so
  they're acoustically/lexically close to real lighting commands. Working hypothesis (not yet
  verified): this is a worst-case number specific to that negative set, and real background
  speech/noise on demo day (not lighting-adjacent) will be rejected far more easily --
  untested until there's a live mic session.
- **`light_on_off` still weak** (29% intent-acc) despite having as much real data (240 clips)
  as `set_temperature`/`set_timer`, which both did well. Its errors are scattered across many
  unrelated classes rather than concentrated on one confusable neighbor. Working hypothesis
  (not yet verified): "on"/"off" are short, low-information utterances carrying less
  distinguishing acoustic signal for this architecture than longer phrase-specific commands.

### Still open
Both open items above are really "how does this behave on real, live audio" questions rather
than confirmed training bugs -- decided the fastest way to get a real answer is a live Pi mic
session (originally an Oct 1 task, pulled forward since the day is ahead of schedule) rather
than more offline experimentation. Attempted this: the Pi was unreachable (SSH timeout to
`192.168.86.4`), paused rather than debugged further per the user's call to skip it for now.

## 2026-09-29 (daytime, continued) — phrase_list.csv fixed directly, phrase-overlap audit run

The canonical `phrase_list.csv` promised from the other planning thread never arrived (flagged
Sep 28-29, still true). Decided not to keep waiting on it -- fixed it directly instead, since
the file only affects `record_dataset.py`'s prompts and future synthetic regeneration, not
training (labels come from the manifest). Two real, mechanical staleness issues, symmetric
with today's `manifest.csv` cleanup:
- Removed the same 12 stale-schema slots (old Fahrenheit temps, old brightness %, old timer
  durations) that were deleted from `manifest.csv` earlier today.
- Added phrases for every slot the Sep 28 scope expansion introduced but never got prompts
  for: `light_dim_color/brightness_20`, `brightness_60`, `brightness_other`, and all 10
  `color_*` slots (`red/blue/green/yellow/pink/white/orange/purple/warm/cool`, matching
  `dispatch.py`'s `COLOR_MAP`); `set_timer/10sec`, `30sec`; `set_temperature/18`, `22`, `26`
  (Celsius).

Then ran the phrase-overlap audit that was blocked on this file arriving -- wrote a quick
word-Jaccard-similarity check across phrases from *different intents* (same-intent overlap is
expected and fine; that's literally what a slot classifier is supposed to disambiguate).
**First pass caught a real problem in the phrases just written**: every new `color_*` phrase
used the template "turn the lights `<color>`", which shares "turn ... the lights" with
`light_on_off`'s "turn on/off the lights" (0.60 word-overlap, the highest score found). This
lines up exactly with tonight's retrain finding that `light_on_off` has scattered, hard-to-explain
confusion -- shared carrier phrasing between two different intents is a concrete, fixable
contributor to that, on top of the "short utterance" hypothesis from before. Reworded every
color phrase to "change the light color to `<color>`" / "make the lights `<color>`" (drops
below 0.4 overlap with `light_on_off` everywhere). Re-ran the audit clean: only two remaining
cross-intent pairs, both pre-existing and both inherent design ambiguity rather than sloppy
phrasing (`media_control/play` "play" vs `play_music/playlist_general` "play music"; "skip
this song" vs "what song is this") -- not fixed, noted as accepted risk.

All 167 tests still pass (test fixtures for `dispatch.py`/`benchmark_harness.py` use their own
inline slot values, not `phrase_list.csv`, so they were unaffected by either cleanup).

### Still open
`phrase_list.csv` was fixed directly rather than waiting further -- no longer blocked on the
other thread. Next real retrain, if any new real recordings are made against the reworded
color phrases, should show whether the `light_on_off`/`light_dim_color` confusion improves.

## 2026-09-30 (Sep 30 morning/daytime) -- WiFi move, breadboard prep, evaluator-plan gap fixed

Physically moved locations; new WiFi meant the Pi (still on the old network's credentials)
became unreachable. Reprovisioned via the SD card's cloud-init seed files
(`network-config`/`meta-data`/`user-data` on the FAT32 boot partition, readable/writable
straight from Windows, no reflash needed) -- edited the WiFi SSID/password and bumped
`instance-id` so cloud-init treats it as a fresh boot and actually reapplies the network
config. First attempt used the new network's 5GHz SSID; after ~9 minutes with no device
appearing on the subnet, switched to the 2.4GHz variant (this hardware/regulatory-domain
combo has a known history of being less reliable on 5GHz) -- still nothing after another ~4
minutes, plus no HDMI signal on a monitor even on the correct port. Only real diagnostic
available without a working monitor or Ethernet cable is the SD card's Linux partition
(cloud-init logs), which needs a Linux-aware reader Windows doesn't have out of the box (WSL
or similar) -- decided not to chase this further today; **paused, will retry this evening**,
with a full SD card reflash as a fallback option (costs ~30-45 min, mostly unattended;
project code/model aren't at risk either way since the Pi is a deployment target, not where
the actual work lives).

**Breadboard LED kit arrived.** Since wiring it and testing it don't need the Pi to be
network-reachable (just powered), did this in parallel rather than wait on the WiFi issue:
- `GPIO_WIRING.md`: exact pin-by-pin wiring instructions matching `dispatch.py`'s
  `GPIOLightController` (BCM 17/22/23/24), written so the user can wire it independently
  without a live session, including a troubleshooting note for common-anode vs common-cathode
  RGB LEDs (a real gotcha: gpiozero does not auto-detect this, needs `active_high=False` in
  code if the LED turns out to be common-anode -- caught and corrected an earlier draft of
  this doc that incorrectly claimed it self-detects).
- `src/vcm/gpio_smoke_test.py`: standalone script cycling every light state with printed
  labels, for a fast visual wiring check independent of the classifier/pipeline.

**Found and fixed a real staleness bug in `benchmark_harness.py`**: `--evaluator-plan`
hardcoded `intents = "media_control,play_music"` from before the Sep 28 scope expansion to 7
intents -- meaning if evaluators had been run as originally written, they'd have only tested
a third of the actual demo scope, silently. Replaced the hardcoded list with a new
`all_command_intents()` helper that reads the current `phrase_list.csv` and excludes only
`reject` (not a "command"), so `--evaluator-plan` can't go stale the same way again as scope
changes further. Verified it now returns all 8 real intents (7 command intents + `wake`).
All 167 tests still pass.

### Still open
Pi WiFi/reachability (paused, evening retry planned), phrase_list-informed re-recording of
`light_on_off`/color phrases (not started), evaluator recruitment, easter-egg song sourcing.

## 2026-10-01 -- the WiFi saga resolved, Pi finally online, real live testing begins

Physically moved locations Sep 29 night/Sep 30, onto a new WiFi network. Two full days lost to
this before it resolved today. Root-caused, not worked around blindly:

### Root cause, found via the SD card's actual logs
Repeated manual edits to the cloud-init `network-config` seed file (new SSID/password, tried
both 5GHz and 2.4GHz bands, bumped `instance-id` each time to force reapplication) never
worked -- confirmed via the router's own admin panel (not just laptop-side ping/ARP, which
turned out to be an insufficient check): zero devices ever associated on either band, across
every attempt, including a full SD card reflash via Raspberry Pi Imager with a guaranteed-
clean config.

Installed DiskInternals Linux Reader (free, read-only ext4 browser) to pull `/var/log/
cloud-init.log` and `/var/log/cloud-init-output.log` directly off the card without needing the
Pi to cooperate at all. Found the real cause: `modules.py[WARNING]: Could not find module
named cc_netplan_nm_patch` on every boot -- the cloud-init module responsible for translating
the netplan-style `network-config` into an actual NetworkManager connection profile was
missing from this image. Confirmed directly: `/etc/NetworkManager/system-connections/` was
completely empty. Cloud-init reported success on every run (no errors, SSH got enabled fine)
while silently never producing a working WiFi connection -- a broken translation step, not a
credentials or hardware problem.

### Fix
Bypassed the broken translation instead of trying to fix it: added a `write_files` entry to
`user-data` that writes a hand-authored `/etc/NetworkManager/system-connections/
preconfigured.nmconnection` keyfile directly (plain-text NetworkManager format, no netplan
involved), plus `runcmd` steps for `rfkill unblock wifi` and `raspi-config nonint
do_wifi_country PH` (Raspberry Pi OS soft-blocks the WiFi radio until a regulatory domain is
set -- the netplan path had been setting this as a side effect, so bypassing it dropped this
too). Both edits done entirely through the FAT32 boot partition, no reflash needed for these.

Even this didn't connect over SSH/network -- finally got a keyboard connected (monitor moved
to the Pi's location, not the other way around, since the Pi is the portable part) and
discovered NetworkManager's GUI applet was sitting there the whole time with an "authentication
required" popup for the WiFi network, needing the password typed in through the GUI once. Did
that -- connected immediately, IP `192.168.88.12`. The nmconnection/rfkill/country fixes likely
weren't even the final blocker; the GUI wanting interactive confirmation might have been. Not
fully certain which fix mattered -- didn't isolate it further given the time already spent, and
it's moot now that it's working.

**Total cost: two days.** Real lesson for the report: authoritative checks (router's own client
table, the SD card's actual logs) found the truth in minutes once used; ping/ARP from the
laptop and assumptions about cloud-init's behavior wasted far more time before that.

### Pi bring-up redone (fresh reflash meant starting over)
- Passwordless sudo: manual step, not part of the cloud-init image -- redone via a terminal
  opened locally on the Pi (password typed there, never told to the assistant).
- Code pushed via plain `tar` over SSH, not `git archive` -- a lot of the day's work (GPIO
  code, benchmark harness fix, phrase_list/manifest cleanup, retrained model) was still
  uncommitted, and `git archive HEAD` only grabs committed snapshots. (Also hit and fixed a
  real bug pushing this way the first time: `2>&1` on the git-archive-to-ssh pipe merged
  stderr text into the binary tar stream and corrupted it -- removed it, pushed clean.)
- `pi_setup.sh` needed two new system packages not previously required: `swig` and
  `liblgpio-dev`, both needed to build the `lgpio` Python package from source (no prebuilt
  wheel for this platform/Python combo). Added to the script for next time.
- `pi_check` (with and without `--mic`): all checks passed. Model loads in 0.14s, inference
  18ms mean (budget was 500ms) -- first real proof of real-time capability on actual hardware.
  Mic capture confirmed working once the USB mic was plugged back in.

### First real live pipeline testing, ever, on this project
Ran `vcm.pipeline --source mic` for the first time against real hardware. Several real
findings, not assumptions:
- **`--min-confidence` default (0.5) was too strict** -- most genuinely-correct top guesses on
  live audio scored well under 0.5, so almost everything got force-mapped to reject. Fixed
  `pipeline.py`'s `_format` to also show the raw pre-threshold guess (`heard`) when it differs
  from the displayed label, so this was actually visible instead of guessed at. Settled on
  `--min-confidence 0.15` as a working value after empirical testing at 0.5, 0.05, and 0.2.
- **The wake window closing after exactly one command is by design**, not a bug -- several
  rounds of "it's not responding" turned out to be forgetting to re-say the wake word before
  every single command, confirmed by re-reading the actual logs rather than trusting
  recollection of what was said.
- **Real confusion found between our own newly-written color phrases and `light_on_off`**
  wasn't retested live yet (no new recordings made against the reworded phrases) -- still open.
- **Music played through the speaker corrupts wake-word recognition** -- wake worked reliably
  before a `play_music` command was dispatched, then failed on every attempt afterward while
  music kept playing. Consistent with the wake word never having been trained with music
  playing behind it (all 65 clips are clean/quiet). Documented as a known limitation, not
  something to re-engineer this close to the deadline: sequence demo commands around it, and
  the pipeline already ducks music to ~5% the moment wake *is* detected -- the vulnerable
  moment is specifically saying the wake word while music is still at full volume.
- **Wake-word reliability is worse live than offline validation suggested**: offline eval
  after the retrain below showed 69-75% precision/recall (n=13, still a small sample); a live
  session of ~60 attempts (all intended as the wake phrase) registered as `wake` only ~18% of
  the time. The model's top-1 guess for failed attempts repeatedly landed on the same few
  classes (`play_music/easter_good_morning`, `media_control/volume_up`, `media_control/
  previous`) rather than spreading randomly -- a real, reproducible confusion, not noise. This
  is the single biggest open risk to the demo as of tonight.

### Wake-word data: more recordings, one bad training run, one good one
Root-caused (partially) the wake problem: the original 25 wake clips were recorded on the
**laptop's built-in mic**, but live testing uses the **USB mic** -- a real train/test mic
mismatch on top of thin data. Moved the USB mic to the laptop and recorded 35 more wake clips
with it (65 total now), verified clean (no silence, no clipping).

Retrained: **first attempt collapsed hard** -- overall accuracy fell to 12%, wake collapsed to
0% (all 13 val clips misclassified as `media_control/next` specifically, not spread out).
Treated this as a real finding to investigate, not something to just re-run past: re-ran with
a different seed rather than assuming either "bad data" or "bad luck" -- the second run hit
39% overall, wake at 75% precision / 69% recall. Concluded the first run was a genuinely bad/
unlucky training run (its learning-rate schedule had already collapsed to a low LR by epoch
50, while the good run was still improving at the initial LR at that point) -- not a real
problem with the new recordings. Pushed the good model to the Pi.

### Wake-word audio cue added, then had to fix it twice
Added a beep-on-wake feature (`make_wake_cue` in `pipeline.py`) so there's feedback on demo
day with no screen attached. First version used a second ad-hoc `sounddevice` output stream --
crashed the whole pipeline with ALSA errors as soon as music was also playing (three
concurrent audio streams -- mic input, Player's music output, beep's own output -- was too
much for this hardware/driver stack to share). Wrapping it in try/except wasn't enough; the
crash was in the mic/player streams' own contention, not something a try/except around the
beep call could catch. Fixed properly by routing the cue through the *existing* espeak-ng TTS
speaker instead (already proven stable everywhere else in this project, goes through a
completely different OS audio path than the sounddevice streams), run in a background thread
so it doesn't block the real-time capture loop. All 167 tests still pass both times.

### Fan-noise / ambient check, finally done
Fans can't be toggled (always-on, wired straight to power, confirmed earlier) so there's no
true "fans off" baseline -- but got a clean "fans on, otherwise quiet" reading: -50 to -51
dBFS, matching the existing `ambient_volume.py` "quiet" placeholder (-50 dBFS) almost exactly.
No recalibration needed; fan noise alone isn't a problem for the mic.

### Honest progress assessment (given to the user directly, logging it here too)
~55-60% complete, not higher -- a working model is not a finished assignment. Solid: core
pipeline, Pi bring-up, most intents at demo-usable quality, first real live end-to-end test.
Real open risk: wake-word live reliability (~18% in tonight's session). Not started at all,
not just in progress: evaluator testing (a professor-quoted grading requirement), breadboard
wiring (kit in hand, not wired), real easter-egg/playlist audio files, rehearsal, and
committing today's substantial uncommitted work to git.

### Still open
Committing today's work (many files uncommitted right now -- real risk), one more wake-word
data round (bigger, more varied distances/pacing), breadboard wiring, evaluator recruitment
and the actual benchmark session, easter-egg/playlist music files, full rehearsal. Two days
left to the Oct 3 demo.
