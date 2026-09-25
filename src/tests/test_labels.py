from vcm import labels
from vcm.labels import command_label, intent_of, slot_of


def test_command_label_joins_intent_and_slot():
    assert command_label("media_control", "next") == "media_control/next"
    assert command_label("ask_time", "none") == "ask_time/none"


def test_all_reject_slots_collapse_to_one_label():
    for slot in ("offvocab", "nearmiss", "silence", "noise"):
        assert command_label("reject", slot) == "reject"


def test_intent_and_slot_of_round_trip():
    assert intent_of("play_music/playlist_jazz") == "play_music"
    assert slot_of("play_music/playlist_jazz") == "playlist_jazz"
    assert intent_of("reject") == "reject"
    assert slot_of("reject") == ""


def test_read_manifest_carries_slot_and_label(tmp_path):
    (tmp_path / "manifest.csv").write_text(
        "filepath,intent,slot,phrase,speaker,condition,source\n"
        "a.wav,media_control,next,next,v,c,s\n"
        "b.wav,reject,silence,__silence__,v,c,s\n",
        encoding="utf-8",
    )
    rows = labels.read_manifest(str(tmp_path / "manifest.csv"), str(tmp_path))
    assert [r.label for r in rows] == ["media_control/next", "reject"]
    assert labels.build_label_list(rows) == ["media_control/next", "reject"]
