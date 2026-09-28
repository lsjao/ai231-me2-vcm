from vcm.train import intent_rollup, wake_confusion_report

LABELS = {
    0: "wake/kuya_jukebox",
    1: "media_control/stop",
    2: "media_control/volume_down",
    3: "media_control/next",
    4: "ask_time/none",
    5: "reject",
    6: "play_music/playlist_jazz",
}


def test_wake_confusion_report_computes_precision_and_recall():
    # 3 true wake: 2 correctly predicted wake, 1 predicted as "stop"
    # 1 false positive: a "stop" utterance predicted as wake
    y_true = [0, 0, 0, 1]
    y_pred = [0, 0, 1, 0]
    report = wake_confusion_report(y_true, y_pred, LABELS)
    assert "n=3" in report
    assert "precision=0.67" in report  # tp=2, fp=1 -> 2/3
    assert "recall=0.67" in report     # tp=2, fn=1 -> 2/3


def test_wake_confusion_report_lists_specific_watch_label_confusions():
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [1, 0, 0, 0, 0, 0]  # wake heard as stop; every watch label heard as wake
    report = wake_confusion_report(y_true, y_pred, LABELS)
    assert "wake/kuya_jukebox -> media_control/stop: 1" in report
    assert "media_control/stop -> wake/kuya_jukebox: 1" in report
    assert "media_control/volume_down -> wake/kuya_jukebox: 1" in report
    assert "media_control/next -> wake/kuya_jukebox: 1" in report
    assert "ask_time/none -> wake/kuya_jukebox: 1" in report
    assert "reject -> wake/kuya_jukebox: 1" in report


def test_wake_confusion_report_handles_no_wake_examples_without_crashing():
    y_true = [1, 2]
    y_pred = [1, 1]
    report = wake_confusion_report(y_true, y_pred, LABELS)
    assert "n=0" in report


def test_wake_confusion_report_handles_missing_wake_label():
    labels = {0: "media_control/stop", 1: "reject"}
    report = wake_confusion_report([0, 1], [0, 1], labels)
    assert "no wake label" in report


def test_intent_rollup_separates_command_accuracy_from_intent_accuracy():
    # both media_control clips: one exactly right, one right intent wrong slot
    y_true = [1, 2]
    y_pred = [1, 1]
    report = intent_rollup(y_true, y_pred, LABELS)
    assert "media_control" in report
    assert "overall command accuracy 0.50 over 2 val clips" in report
