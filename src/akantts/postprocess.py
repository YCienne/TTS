"""Light cleanup for synthesized speech: tame vocoder buzz, normalize level.

A VITS checkpoint fine-tuned on well under an hour of audio (our speaker-3 run:
0.58h) does not converge as cleanly as one trained on the many hours a production
TTS system uses -- the generator/discriminator pair leaves audible high-frequency
ringing and a faint noise floor. This cannot be removed by signal processing
without re-training on more data, but a gentle spectral gate and a high-shelf
cut measurably reduce how harsh/synthetic it sounds, without touching the model.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, stft, istft


def _spectral_gate(wav: np.ndarray, sr: int, strength: float = 0.6) -> np.ndarray:
    """Subtract a per-frequency noise floor estimated from the quietest frames.

    Classic spectral-subtraction denoise: the floor is the 10th-percentile
    magnitude in each frequency bin across the clip (assumes the artifact is
    roughly stationary, which vocoder ringing is). `strength` is how much of
    that floor to remove (0 = untouched, 1 = full subtraction).
    """
    f, t, Z = stft(wav, fs=sr, nperseg=512, noverlap=384)
    mag, phase = np.abs(Z), np.angle(Z)
    floor = np.percentile(mag, 10, axis=1, keepdims=True)
    gated = np.maximum(mag - strength * floor, 0.05 * mag)
    _, cleaned = istft(gated * np.exp(1j * phase), fs=sr, nperseg=512, noverlap=384)
    return cleaned[: len(wav)].astype(np.float32)


def _high_shelf_cut(wav: np.ndarray, sr: int, cutoff_hz: float = 6500, order: int = 4) -> np.ndarray:
    """Gentle low-pass above `cutoff_hz` -- most Akan speech content sits below this;
    vocoder buzz concentrates near the Nyquist edge and is the first thing a listener
    notices as "synthetic"."""
    nyquist = sr / 2
    if cutoff_hz >= nyquist:
        return wav
    sos = butter(order, cutoff_hz / nyquist, btype="low", output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def clean_audio(wav: np.ndarray, sr: int, target_peak: float = 0.89) -> np.ndarray:
    """Spectral-gate denoise + gentle high-shelf cut + peak normalize to `target_peak`
    (~-1 dBFS). Safe no-op on silence (an all-zero or near-zero clip is returned as-is)."""
    wav = np.asarray(wav, dtype=np.float32)
    peak = np.max(np.abs(wav)) if len(wav) else 0.0
    if peak < 1e-6:
        return wav
    cleaned = _spectral_gate(wav, sr)
    cleaned = _high_shelf_cut(cleaned, sr)
    cleaned_peak = np.max(np.abs(cleaned))
    if cleaned_peak > 1e-6:
        cleaned = cleaned * (target_peak / cleaned_peak)
    return cleaned.astype(np.float32)
