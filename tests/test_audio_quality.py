import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akantts.audio_quality import clip_quality, flags, frame_db, pitch_stats  # noqa: E402

SR = 16000


def _bursts(noise_amp, tone_amp=0.3, f=150.0, seconds=6, seed=0):
    """Alternating 0.3 s tone bursts and 0.3 s of noise only, like speech and pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    tone = tone_amp * np.sin(2 * np.pi * f * t) * (np.floor(t / 0.3) % 2 == 0)
    return (tone + noise_amp * rng.standard_normal(len(t))).astype(np.float32)


def test_frame_db_handles_short_input():
    assert frame_db(np.zeros(10), SR).shape == (1,)


def test_snr_tracks_noise_level():
    clean = clip_quality(_bursts(0.001), SR)["snr_db"]
    noisy = clip_quality(_bursts(0.03), SR)["snr_db"]
    assert clean > noisy + 15
    assert 35 < clean < 60


def test_clipping_fraction():
    wav = np.concatenate([np.ones(100), np.zeros(900)])
    assert clip_quality(wav, SR)["clipping_fraction"] == pytest.approx(0.1)


def test_silence_fraction_about_half():
    assert 0.35 < clip_quality(_bursts(0.001), SR)["silence_fraction"] < 0.65


def test_pitch_recovers_tone_frequency():
    pytest.importorskip("librosa")
    t = np.arange(2 * SR) / SR
    stats = pitch_stats((0.3 * np.sin(2 * np.pi * 120 * t)).astype(np.float32), SR)
    assert stats is not None and abs(stats["f0_median_hz"] - 120) < 6


def test_flags():
    assert "low SNR (<25 dB)" in flags({"snr_db_median": 10})
    assert flags({"snr_db_median": 40, "f0_median_hz": 110}, "Male") == []
    assert any("male" in f for f in flags({"snr_db_median": 40, "f0_median_hz": 220}, "Male"))
    assert any("mixed" in f for f in flags({"f0_clip_median_std_hz": 40}))
