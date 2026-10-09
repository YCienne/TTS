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


def align(a, b):
    """Levenshtein alignment: ``[(op, x, y), ...]`` covering all of ``a`` and ``b`` in order.

    ``op`` is ``"match"``, ``"sub"`` (``x`` from ``a`` read as ``y`` from ``b``), ``"del"``
    (``x`` with no counterpart, ``y`` is ``None``) or ``"ins"`` (``y`` with no counterpart,
    ``x`` is ``None``). Same cost model as :func:`edit_distance`, with the traceback kept.
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j - 1] if a[i - 1] == b[j - 1] else 1 + min(
                dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", a[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, b[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def vowel_harmony_confusions(ref: str, hyp: str) -> dict:
    """Counts of e<->ɛ and o<->ɔ substitutions between (cleaned) ``ref`` and ``hyp``.

    These are the two ATR (tongue-root) vowel-harmony pairs in Akan orthography, and the main
    segmental risk this project's text normalization has to get right (see
    ``akantts.normalize``). Everything else a wrong vowel height could cause is already in
    plain ``cer``/``wer``; this isolates the harmony-specific confusion so it can be reported
    on its own. Tone is not marked in this orthography, so neither ``ref`` nor an ASR ``hyp``
    carries it -- this cannot check tone; use the native-speaker listening test for that.
    """
    counts = {"e_to_ɛ": 0, "ɛ_to_e": 0, "o_to_ɔ": 0, "ɔ_to_o": 0}
    pairs = {("e", "ɛ"): "e_to_ɛ", ("ɛ", "e"): "ɛ_to_e", ("o", "ɔ"): "o_to_ɔ", ("ɔ", "o"): "ɔ_to_o"}
    for op, x, y in align(list(clean(ref)), list(clean(hyp))):
        if op == "sub" and (x, y) in pairs:
            counts[pairs[(x, y)]] += 1
    return counts
