"""Error rates between a reference text and an ASR transcript.

Both sides go through the same normalization (case, ɛ/ɔ spelling, punctuation) so the
rates measure content, not formatting. ASR for Akan is itself imperfect, so these are
upper bounds on the true error of the audio: use them to rank and to find outliers.
"""
import re

from .normalize import normalize

_NOT_WORD = re.compile(r"[^\w\s]|_")
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Lowercase, canonical ɛ/ɔ, no punctuation or bracketed annotations, single spaces."""
    return _WS.sub(" ", _NOT_WORD.sub("", normalize(text))).strip()


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate of ``hyp`` against ``ref`` (can exceed 1.0)."""
    ref, hyp = clean(ref), clean(hyp)
    return edit_distance(ref, hyp) / max(len(ref), 1)


def wer(ref: str, hyp: str) -> float:
    """Word error rate of ``hyp`` against ``ref`` (can exceed 1.0)."""
    ref, hyp = clean(ref).split(), clean(hyp).split()
    return edit_distance(ref, hyp) / max(len(ref), 1)
