"""Cheap recording-quality statistics for choosing a TTS speaker.

These are heuristics for ranking and sanity checks, not calibrated measurements.
The SNR is a speech-to-quiet-frames ratio: it is only a lower bound when a clip
has no real pauses, and it cannot tell noise from reverberation.
"""
import numpy as np

FRAME_SEC, HOP_SEC = 0.025, 0.010
CLIP_LEVEL = 0.99


def frame_db(wav: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame mean-square level in dB (25 ms frames, 10 ms hop)."""
    n, h = int(FRAME_SEC * sr), int(HOP_SEC * sr)
    wav = np.asarray(wav, dtype=np.float64)
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    csum = np.concatenate([[0.0], np.cumsum(wav ** 2)])  # avoids materialising every frame
    starts = np.arange(0, len(wav) - n + 1, h)
    return 10 * np.log10((csum[starts + n] - csum[starts]) / n + 1e-10)


def clip_quality(wav: np.ndarray, sr: int) -> dict:
    """Level, SNR-like, silence and clipping statistics for one clip."""
    db = frame_db(wav, sr)
    quiet, loud = np.percentile(db, [10, 90])
    return {
        "noise_floor_db": float(quiet),
        "speech_level_db": float(loud),
        "snr_db": float(loud - quiet),
        "silence_fraction": float(np.mean(db < loud - 30)),
        "clipping_fraction": float(np.mean(np.abs(wav) >= CLIP_LEVEL)),
    }


def pitch_stats(wav: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0):
    """Median F0 (Hz), F0 spread (semitone std) and voiced fraction, or None if too little voicing."""
    import librosa

    if sr != 16000:
        wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    f0, voiced, _ = librosa.pyin(np.asarray(wav, dtype=np.float32), fmin=fmin, fmax=fmax, sr=sr,
                                 frame_length=1024)
    f0v = f0[voiced & ~np.isnan(f0)]
    if len(f0v) < 20:
        return None
    return {
        "f0_median_hz": float(np.median(f0v)),
        "f0_std_semitones": float(np.std(12 * np.log2(f0v))),
        "voiced_fraction": float(np.mean(voiced)),
    }


def flags(summary: dict, gender: str = "") -> list:
    """Human-readable warnings for a speaker summary (heuristic thresholds)."""
    out = []
    if summary.get("snr_db_median", 99) < 25:
        out.append("low SNR (<25 dB)")
    if summary.get("clipping_fraction_max", 0) > 1e-3:
        out.append("some clips are clipped")
    f0 = summary.get("f0_median_hz")
    if f0 is not None:
        if gender == "Male" and f0 > 170:
            out.append("F0 high for a male label")
        if gender == "Female" and f0 < 150:
            out.append("F0 low for a female label")
    if summary.get("f0_clip_median_std_hz", 0) > 25:
        out.append("F0 varies a lot between clips: possible mixed voices")
    return out
