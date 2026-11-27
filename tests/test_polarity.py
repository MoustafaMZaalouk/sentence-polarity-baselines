import numpy as np
import pytest

from polarity.data import SHA256, ensure_data, load_texts, sha256_of
from polarity.metrics import accuracy, bootstrap_ci, expected_calibration_error
from polarity.models import MODEL_NAMES, make_model
from polarity.run import run, split


def test_ece_perfect_and_overconfident():
    y = np.array([1, 0, 1, 0])
    assert expected_calibration_error(y, np.array([1.0, 0.0, 1.0, 0.0])) == pytest.approx(0.0)
    # always 100% confident but half wrong -> ECE 0.5
    assert expected_calibration_error(np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.5)


def test_accuracy_and_bootstrap_bounds():
    y = np.array([0, 1] * 50)
    assert accuracy(y, y) == 1.0
    lo, hi = bootstrap_ci(y, y, n=200)
    assert lo == hi == 1.0


def test_checksum_constant_format():
    assert len(SHA256) == 64


def test_synthetic_models_learn_toy_signal():
    pos = ["great fun wonderful film", "a delight and moving", "brilliant superb acting", "lovely warm story"] * 10
    neg = ["dull boring mess", "terrible and tedious", "awful waste of time", "bad clumsy script"] * 10
    texts = np.array(pos + neg, dtype=object)
    y = np.array([1] * len(pos) + [0] * len(neg))
    for name in MODEL_NAMES:
        m = make_model(name, 0)
        m.fit(texts, y)
        acc = accuracy(y, m.predict(texts))
        assert acc >= 0.5
        if name != "majority":
            assert acc > 0.95


def test_split_is_stratified_and_disjoint():
    y = np.array([0, 1] * 100)
    tr, te = split(list(range(200)), y, 0)
    assert not set(tr) & set(te)
    assert abs(y[te].mean() - 0.5) < 0.02


def test_real_data_download_matches_pin(tmp_path):
    d = ensure_data("data")
    assert sha256_of(d / "rt-polaritydata.tar.gz") == SHA256
    texts, labels = load_texts(d)
    assert len(texts) == 10662 and sum(labels) == 5331


def test_smoke_run_writes_results(tmp_path):
    r = run(seeds=[0], subsample=1500, out_dir=tmp_path)
    assert (tmp_path / "results.json").exists() and (tmp_path / "accuracy.png").exists()
    assert r["models"]["lr_word12"]["accuracy_mean"] > r["models"]["majority"]["accuracy_mean"]
