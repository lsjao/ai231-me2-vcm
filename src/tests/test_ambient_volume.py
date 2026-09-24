import numpy as np
import pytest

from vcm.ambient_volume import (
    FRAME_SAMPLES,
    NOISE_FLOOR_LOUD_DBFS,
    NOISE_FLOOR_QUIET_DBFS,
    UPDATE_INTERVAL_FRAMES,
    VOLUME_AT_LOUD,
    VOLUME_AT_QUIET,
    AmbientAutoVolume,
    AmbientNoiseTracker,
    frame_dbfs,
    volume_for_noise_floor,
)


def make_noise_frame(rms: float, rng: np.random.Generator, n: int = FRAME_SAMPLES) -> np.ndarray:
    return (rng.standard_normal(n) * rms).astype(np.float32)


def dbfs_to_rms(dbfs: float) -> float:
    return 10 ** (dbfs / 20.0)


def test_frame_dbfs_of_silence_hits_the_floor():
    silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    assert frame_dbfs(silence) <= -80.0


def test_frame_dbfs_increases_with_amplitude():
    rng = np.random.default_rng(0)
    quiet = make_noise_frame(dbfs_to_rms(-50), rng)
    loud = make_noise_frame(dbfs_to_rms(-20), rng)
    assert frame_dbfs(loud) > frame_dbfs(quiet)


def test_volume_for_noise_floor_clamps_at_endpoints():
    assert volume_for_noise_floor(NOISE_FLOOR_QUIET_DBFS - 10) == VOLUME_AT_QUIET
    assert volume_for_noise_floor(NOISE_FLOOR_LOUD_DBFS + 10) == VOLUME_AT_LOUD


def test_volume_for_noise_floor_interpolates_at_midpoint():
    mid = (NOISE_FLOOR_QUIET_DBFS + NOISE_FLOOR_LOUD_DBFS) / 2
    expected_mid_volume = (VOLUME_AT_QUIET + VOLUME_AT_LOUD) / 2
    assert volume_for_noise_floor(mid) == pytest.approx(expected_mid_volume, abs=1)


def test_tracker_converges_toward_a_sustained_noise_level():
    rng = np.random.default_rng(1)
    tracker = AmbientNoiseTracker(initial_dbfs=NOISE_FLOOR_QUIET_DBFS)
    rms = dbfs_to_rms(NOISE_FLOOR_LOUD_DBFS)
    for _ in range(300):
        tracker.update(make_noise_frame(rms, rng))
    assert tracker.floor_dbfs == pytest.approx(NOISE_FLOOR_LOUD_DBFS, abs=2)


def test_tracker_ignores_a_single_loud_transient():
    rng = np.random.default_rng(2)
    tracker = AmbientNoiseTracker(initial_dbfs=NOISE_FLOOR_QUIET_DBFS)
    before = tracker.floor_dbfs
    loud_burst = make_noise_frame(dbfs_to_rms(0.0), rng)  # very loud, single frame
    tracker.update(loud_burst)
    # release is slow -- one loud frame should barely move the floor
    assert tracker.floor_dbfs < before + 3


def test_tracker_drops_quickly_when_room_goes_quiet():
    rng = np.random.default_rng(3)
    tracker = AmbientNoiseTracker(initial_dbfs=NOISE_FLOOR_LOUD_DBFS)
    quiet_rms = dbfs_to_rms(NOISE_FLOOR_QUIET_DBFS)
    for _ in range(50):
        tracker.update(make_noise_frame(quiet_rms, rng))
    assert tracker.floor_dbfs == pytest.approx(NOISE_FLOOR_QUIET_DBFS, abs=2)


def _run_until_next_suggestion(auto_vol: AmbientAutoVolume, rms: float, rng: np.random.Generator, max_cycles: int = 20):
    for _ in range(max_cycles * UPDATE_INTERVAL_FRAMES):
        result = auto_vol.update(make_noise_frame(rms, rng))
        if result is not None:
            return result
    return None


def test_rate_limiting_suppresses_updates_before_the_interval_elapses():
    rng = np.random.default_rng(4)
    auto_vol = AmbientAutoVolume(current_volume=VOLUME_AT_QUIET)
    loud_rms = dbfs_to_rms(NOISE_FLOOR_LOUD_DBFS)
    for _ in range(UPDATE_INTERVAL_FRAMES - 1):
        assert auto_vol.update(make_noise_frame(loud_rms, rng)) is None


def test_auto_volume_tracks_up_for_a_loud_room():
    rng = np.random.default_rng(5)
    auto_vol = AmbientAutoVolume(current_volume=VOLUME_AT_QUIET)
    loud_rms = dbfs_to_rms(NOISE_FLOOR_LOUD_DBFS)
    suggestion = _run_until_next_suggestion(auto_vol, loud_rms, rng)
    assert suggestion is not None
    assert suggestion > VOLUME_AT_QUIET


def test_auto_volume_tracks_down_for_a_quiet_room():
    rng = np.random.default_rng(6)
    auto_vol = AmbientAutoVolume(current_volume=VOLUME_AT_LOUD)
    quiet_rms = dbfs_to_rms(NOISE_FLOOR_QUIET_DBFS)
    suggestion = _run_until_next_suggestion(auto_vol, quiet_rms, rng)
    assert suggestion is not None
    assert suggestion < VOLUME_AT_LOUD


def test_hysteresis_suppresses_further_updates_once_settled():
    rng = np.random.default_rng(7)
    auto_vol = AmbientAutoVolume(current_volume=VOLUME_AT_QUIET)
    loud_rms = dbfs_to_rms(NOISE_FLOOR_LOUD_DBFS)
    # let it fully converge and emit its first suggestion(s)
    for _ in range(10 * UPDATE_INTERVAL_FRAMES):
        auto_vol.update(make_noise_frame(loud_rms, rng))
    # steady-state noise at the same level should stop producing suggestions
    settled_results = [
        auto_vol.update(make_noise_frame(loud_rms, rng))
        for _ in range(3 * UPDATE_INTERVAL_FRAMES)
    ]
    assert all(r is None for r in settled_results)


def test_pause_suppresses_suggestions_until_resumed():
    rng = np.random.default_rng(8)
    auto_vol = AmbientAutoVolume(current_volume=VOLUME_AT_QUIET)
    auto_vol.pause()
    loud_rms = dbfs_to_rms(NOISE_FLOOR_LOUD_DBFS)
    for _ in range(10 * UPDATE_INTERVAL_FRAMES):
        assert auto_vol.update(make_noise_frame(loud_rms, rng)) is None

    auto_vol.resume()
    suggestion = _run_until_next_suggestion(auto_vol, loud_rms, rng)
    assert suggestion is not None
