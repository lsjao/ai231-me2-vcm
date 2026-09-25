from collections import Counter

from vcm import data
from vcm.labels import ManifestRow


def _rows(spec):
    return [ManifestRow(f"{label}_{i}.wav", label.split("/")[0], (label.split("/") + [""])[1])
            for label, n in spec.items() for i in range(n)]


def test_split_gives_every_multi_clip_label_a_val_clip_even_with_many_labels():
    # more labels than a 20% val slice has slots -- sklearn's stratified split would refuse this
    rows = _rows({f"intent{i}/slot": 3 for i in range(40)})
    train, val = data.split_rows(rows, val_fraction=0.2, seed=0)
    assert len(train) + len(val) == len(rows)
    assert Counter(r.label for r in val).keys() == {r.label for r in rows}


def test_split_keeps_single_clip_labels_in_train_only():
    rows = _rows({"a/x": 5, "lonely/x": 1})
    train, val = data.split_rows(rows, val_fraction=0.2, seed=0)
    assert any(r.label == "lonely/x" for r in train)
    assert all(r.label != "lonely/x" for r in val)


def test_split_is_deterministic_for_a_seed():
    rows = _rows({"a/x": 10, "b/x": 10})
    first = data.split_rows(rows, seed=7)
    second = data.split_rows(rows, seed=7)
    assert [r.filepath for r in first[1]] == [r.filepath for r in second[1]]


def test_class_weights_upweight_rare_labels():
    rows = _rows({"a/x": 9, "b/x": 1})
    label_to_idx = {"a/x": 0, "b/x": 1}
    w = data.class_weights(rows, label_to_idx)
    assert w[1] > w[0]
