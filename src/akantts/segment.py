"""Split long read-aloud clips into sentence-level training segments.

WAXAL ``twi_tts`` clips have a median length of 16 s but a 99 s p90, and VITS
trains on clips of about 20 s or less. The speakers read sentence by sentence, so
a clip with N sentences is cut at its N-1 longest internal pauses. There are no
word timestamps to check this against, so callers must validate each result
(``plausible``) and discard a clip whose segments look wrong.
"""
import re

import numpy as np

from .audio_quality import HOP_SEC, frame_db
from .normalize import normalize

PAD_SEC = 0.15  # silence kept on each side of a segment
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


def sentences(raw: str) -> list:
    """Normalized sentences of ``raw``, split on line breaks and ``. ? !``."""
    out = []
    for line in raw.splitlines():
        line = normalize(line)
        if line:
            out.extend(s for s in _SENTENCE_END.split(line) if s)
    return out


def speech_activity(wav: np.ndarray, sr: int) -> np.ndarray:
    """Boolean per 10 ms frame: louder than a quarter of the way from noise floor to speech level."""
    db = frame_db(wav, sr)
    noise, loud = np.percentile(db, [10, 90])
    return db >= noise + 0.25 * (loud - noise)


def find_pauses(active: np.ndarray, min_pause: float):
    """Internal pauses of at least ``min_pause`` s as (start_s, end_s), and the (first, last) speech span."""
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return [], None
    first, last = int(idx[0]), int(idx[-1])
    pauses, f = [], first
    while f <= last:
        if active[f]:
            f += 1
            continue
        g = f
        while g <= last and not active[g]:
            g += 1
        if (g - f) * HOP_SEC >= min_pause:
            pauses.append((f * HOP_SEC, g * HOP_SEC))
        f = g
    return pauses, (first * HOP_SEC, (last + 1) * HOP_SEC)


def split_by_pauses(wav: np.ndarray, sr: int, n_parts: int, min_pause: float = 0.2):
    """(start_s, end_s) for ``n_parts`` pieces cut at the longest pauses, or None if there are too few."""
    pauses, span = find_pauses(speech_activity(wav, sr), min_pause)
    if span is None:
        return None
    start, end = max(0.0, span[0] - PAD_SEC), min(len(wav) / sr, span[1] + PAD_SEC)
    if n_parts == 1:
        return [(start, end)]
    if len(pauses) < n_parts - 1:
        return None
    cuts = sorted(sorted(pauses, key=lambda p: p[1] - p[0], reverse=True)[: n_parts - 1])
    parts, seg_start = [], start
    for p_start, p_end in cuts:
        mid = (p_start + p_end) / 2
        parts.append((seg_start, min(mid, p_start + PAD_SEC)))
        seg_start = max(mid, p_end - PAD_SEC)
    parts.append((seg_start, end))
    return parts


def plausible(parts, texts, median_rate: float, band=(0.5, 1.8), min_sec=1.0, max_sec=20.0) -> bool:
    """True if every segment has a sane length and a characters-per-second rate near the speaker's median."""
    for (a, b), text in zip(parts, texts):
        dur = b - a
        if not (min_sec <= dur <= max_sec):
            return False
        if not (band[0] * median_rate <= len(text) / dur <= band[1] * median_rate):
            return False
    return True
