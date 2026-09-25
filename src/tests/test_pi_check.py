import numpy as np

from vcm import audio
from vcm.pi_check import latency_stats, level_stats


def test_level_stats_orders_percentiles_and_reports_peak():
    rng = np.random.default_rng(0)
    quiet = rng.standard_normal(audio.TARGET_SR) * 10 ** (-60 / 20)
    loud = np.sin(2 * np.pi * 300 * np.arange(audio.TARGET_SR // 2) / audio.TARGET_SR) * 0.3
    s = level_stats(np.concatenate([quiet, loud]).astype(np.float32))
    assert s["p10"] < s["p50"] <= s["p90"]
    assert s["p10"] < -50 and s["p90"] > -20
    assert abs(s["peak"] - 0.3) < 0.01


def test_latency_stats():
    s = latency_stats([1.0, 2.0, 3.0, 100.0])
    assert s["max"] == 100.0 and s["mean"] == 26.5 and s["p95"] > 3.0
