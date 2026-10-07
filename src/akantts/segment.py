"""Split long read-aloud clips into sentence-level training segments.

WAXAL ``twi_tts`` clips have a median length of 16 s but a 99 s p90, and VITS
trains on clips of about 20 s or less. The speakers read sentence by sentence, so
a clip with N sentences is cut at N-1 of its internal pauses. Choosing the N-1
*longest* pauses failed on a real speaker (100 of ~117 long clips rejected): speakers
also pause inside sentences. ``choose_cuts`` therefore picks the pauses that make
each segment's duration proportional to its sentence's length in characters, with a
small preference for longer pauses. There are still no word timestamps to check
against, so callers must validate each result (``check_segments``) and discard a
clip whose segments look wrong.
"""
import re

import numpy as np

from .audio_quality import HOP_SEC, frame_db
from .normalize import normalize

PAD_SEC = 0.15  # silence kept on each side of a segment
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


_CLAUSE_END = re.compile(r"(?<=[.?!,;:])\s+")


def _split(raw: str, pattern) -> list:
    out = []
    for line in raw.splitlines():
        line = normalize(line)
        if line:
            out.extend(s for s in pattern.split(line) if s)
    return out


def sentences(raw: str) -> list:
    """Normalized sentences of ``raw``, split on line breaks and ``. ? !``."""
    return _split(raw, _SENTENCE_END)


def clauses(raw: str) -> list:
    """Like ``sentences`` but also split after ``, ; :``, for sentences too long for one segment."""
    return _split(raw, _CLAUSE_END)


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


def choose_cuts(pauses, span, chars, bonus_per_sec: float = 0.15, bonus_cap: float = 1.5):
    """Pick ``len(chars) - 1`` pauses so each segment's length matches its character count.

    Minimises the sum over segments of (log duration - log expected duration)^2, where the
    expected duration is ``chars / rate`` and ``rate`` is the clip's overall characters per
    second, minus a small bonus for longer pauses. Dynamic programming over the candidate
    pauses. Returns the chosen pauses in time order, or None if there are too few.
    """
    n, m = len(chars), len(pauses)
    if m < n - 1:
        return None
    rate = sum(chars) / (span[1] - span[0])
    mids = np.array([(a + b) / 2 for a, b in pauses])
    bonus = bonus_per_sec * np.minimum(np.array([b - a for a, b in pauses]), bonus_cap)

    def seg_cost(t0, t1, c):
        return (np.log(np.maximum(t1 - t0, 0.2)) - np.log(c / rate)) ** 2

    inf = 1e18
    later = np.triu(np.ones((m, m), dtype=bool), k=1)  # [i, j] true when pause j comes after pause i
    dp = np.full((n - 1, m), inf)
    back = np.zeros((n - 1, m), dtype=int)
    dp[0] = seg_cost(span[0], mids, chars[0]) - bonus
    for k in range(1, n - 1):
        total = np.where(later, dp[k - 1][:, None] + seg_cost(mids[:, None], mids[None, :], chars[k]), inf)
        back[k] = total.argmin(axis=0)
        dp[k] = total.min(axis=0) - bonus
    j = int((dp[-1] + seg_cost(mids, span[1], chars[-1])).argmin())
    chosen = [j]
    for k in range(n - 2, 0, -1):
        j = int(back[k][j])
        chosen.append(j)
    return [pauses[i] for i in reversed(chosen)]


def split_by_pauses(wav: np.ndarray, sr: int, n_parts: int, min_pause: float = 0.2, chars=None):
    """(start_s, end_s) for ``n_parts`` pieces, or None if there are too few pauses.

    With ``chars`` (character count of each sentence) the cuts come from ``choose_cuts``;
    without it they are the ``n_parts - 1`` longest pauses.
    """
    pauses, span = find_pauses(speech_activity(wav, sr), min_pause)
    if span is None:
        return None
    start, end = max(0.0, span[0] - PAD_SEC), min(len(wav) / sr, span[1] + PAD_SEC)
    if n_parts == 1:
        return [(start, end)]
    if len(pauses) < n_parts - 1:
        return None
    if chars is not None:
        cuts = choose_cuts(pauses, span, chars)
    else:
        cuts = sorted(sorted(pauses, key=lambda p: p[1] - p[0], reverse=True)[: n_parts - 1])
    parts, seg_start = [], start
    for p_start, p_end in cuts:
        mid = (p_start + p_end) / 2
        parts.append((seg_start, min(mid, p_start + PAD_SEC)))
        seg_start = max(mid, p_end - PAD_SEC)
    parts.append((seg_start, end))
    return parts


def check_segments(parts, texts, median_rate: float, band=(0.5, 1.8), min_sec=1.0, max_sec=20.0):
    """None if every segment has a sane length and speaking rate, else the first problem found."""
    for (a, b), text in zip(parts, texts):
        dur = b - a
        if dur < min_sec:
            return "segment too short"
        if dur > max_sec:
            return "segment too long"
        rate = len(text) / dur
        if rate < band[0] * median_rate:
            return "speaking rate too low"
        if rate > band[1] * median_rate:
            return "speaking rate too high"
    return None


def plausible(parts, texts, median_rate: float, **kwargs) -> bool:
    return check_segments(parts, texts, median_rate, **kwargs) is None
