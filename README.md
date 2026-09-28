# VCM Dataset Scaffold

## What's in here
- `phrase_list.csv` — 70 phrases (69 locked + the `wake/kuya_jukebox` wake phrase) across 7 intents + reject class, with slot labels
- `dataset/` — folder structure `intent=X/slot=Y/`, pre-populated with 201 synthetic TTS WAV files (3 espeak-ng voices per phrase, ~14MB)
- `manifest.csv` — one row per audio file: filepath, intent, slot, phrase, speaker, condition, source

## Counts per intent (synthetic data only, so far)
| Intent | Files |
|---|---|
| media_control | 48 |
| play_music | 30 |
| set_timer | 30 |
| light_dim_color | 24 |
| light_on_off | 24 |
| set_temperature | 18 |
| ask_time | 12 |
| reject | 15 |

Not balanced yet on purpose, media_control has more sub-classes so it needed more phrases. Balance this out once real recordings come in.

## Still missing (real recordings needed)
- `reject/slot=silence` and `reject/slot=noise` — TTS can't produce these, need actual silence clips and ambient noise recordings
- All real human voices — this is 100% synthetic right now, meant to unblock early training only
- Easter egg songs — `good_morning` and `stage_fright` phrases are the *trigger commands*, you still need the actual Good Morning / No songs as local MP3 files for playback, that's separate from this dataset

## How to add real recordings
Real recordings are kept out of `dataset/` and out of git (bulky, personal voice). Use the guided tool from `src/`:
`python -m vcm.record_dataset --speaker <name> --condition quiet --distance near --auto` -- it writes to `data_real/`
with its own `manifest.csv`. Train with `--extra-data ../data_real`. See HANDOFF.md for the full recording plan.

## Next steps
- Record real silence + noise clips for the reject class (5 min effort, just record an empty room and a TV-on room)
- Start team recording sessions against `phrase_list.csv`
- Once ~20-30% real data is in, start swapping TTS-heavy batches out during training
