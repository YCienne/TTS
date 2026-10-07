import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akantts.segment import plausible, sentences, split_by_pauses  # noqa: E402

SR = 16000


def _clip(pieces, noise=0.002, seed=0):
    """pieces: list of (seconds, is_speech). Speech is a 150 Hz tone, silence is low noise."""
    rng = np.random.default_rng(seed)
    out = []
    for secs, speech in pieces:
        t = np.arange(int(secs * SR)) / SR
        out.append((0.3 * np.sin(2 * np.pi * 150 * t) if speech else np.zeros_like(t)))
    wav = np.concatenate(out)
    return (wav + noise * rng.standard_normal(len(wav))).astype(np.float32)


def test_sentences_split_on_punctuation_and_newlines():
    assert sentences("Aprɛ yɛ dɛ. Wɔkɔ he?\n\nMe kɔ!") == ["aprɛ yɛ dɛ.", "wɔkɔ he?", "me kɔ!"]
    assert sentences("[10] du.\n") == ["du."]
    assert sentences("") == []


def test_clauses_split_at_commas_too():
    from akantts.segment import clauses

    raw = "Sɛ wo kɔ fie a, ka kyerɛ me; Mehunu mununkum. Ɛyɛ dɛ?"
    assert clauses(raw) == ["sɛ wo kɔ fie a,", "ka kyerɛ me;", "mehunu mununkum.", "ɛyɛ dɛ?"]
    assert len(sentences(raw)) == 2  # ";" and "," do not end a sentence


def test_cuts_at_longest_pauses_not_short_ones():
    # three sentences; a 0.25 s comma pause inside the second one must not become a cut
    wav = _clip([(0.5, 0), (1.0, 1), (0.8, 0), (0.7, 1), (0.25, 0), (0.8, 1), (0.6, 0), (1.2, 1), (0.5, 0)])
    parts = split_by_pauses(wav, SR, 3)
    assert len(parts) == 3
    # Speech runs 0.5-1.5, 2.3-3.0, 3.25-4.05, 4.65-5.85 s. The cuts are the 0.8 s pause (1.5-2.3)
    # and the 0.6 s pause (4.05-4.65); each side keeps 0.15 s of silence (PAD_SEC).
    expected = [(0.35, 1.65), (2.15, 4.20), (4.50, 6.00)]
    for (a, b), (ea, eb) in zip(parts, expected):
        assert abs(a - ea) < 0.1 and abs(b - eb) < 0.1


def test_text_lengths_beat_a_long_pause_inside_a_sentence():
    # Sentences of 20, 60 and 30 characters. The middle one has a 0.9 s breath pause that is
    # longer than both real sentence gaps (0.5 s), which fools "cut at the longest pauses".
    wav = _clip([(0.3, 0), (2.0, 1), (0.5, 0), (3.0, 1), (0.9, 0), (3.0, 1), (0.5, 0), (3.0, 1), (0.3, 0)])
    naive = split_by_pauses(wav, SR, 3)
    assert any(abs(b - 5.95) < 0.1 for _, b in naive)  # the breath pause (5.8-6.7) became a cut
    parts = split_by_pauses(wav, SR, 3, chars=[20, 60, 30])
    # real gaps: 2.3-2.8 s and 9.7-10.2 s, each keeps 0.15 s of silence per side
    expected = [(0.15, 2.45), (2.65, 9.85), (10.05, 13.35)]
    assert len(parts) == 3
    for (a, b), (ea, eb) in zip(parts, expected):
        assert abs(a - ea) < 0.12 and abs(b - eb) < 0.12


def test_choose_cuts_none_when_too_few_pauses():
    from akantts.segment import choose_cuts

    assert choose_cuts([(1.0, 1.5)], (0.0, 10.0), [10, 10, 10]) is None


def test_single_sentence_is_trimmed_not_cut():
    wav = _clip([(1.0, 0), (2.0, 1), (1.0, 0)])
    (a, b), = split_by_pauses(wav, SR, 1)
    assert 0.7 < a < 1.0 and 3.0 < b < 3.3


def test_too_few_pauses_returns_none():
    assert split_by_pauses(_clip([(0.5, 0), (3.0, 1), (0.5, 0)]), SR, 3) is None


def test_plausible_checks_rate_and_length():
    parts, texts = [(0.0, 4.0), (4.0, 8.0)], ["a" * 32, "a" * 32]  # 8 chars/s
    assert plausible(parts, texts, median_rate=8.0)
    assert not plausible(parts, ["a" * 4, "a" * 32], median_rate=8.0)  # 1 char/s
    assert not plausible([(0.0, 30.0)], ["a" * 240], median_rate=8.0)  # too long
    from akantts.segment import check_segments

    assert check_segments(parts, ["a" * 4, "a" * 32], 8.0) == "speaking rate too low"
    assert check_segments([(0.0, 0.5)], ["a" * 4], 8.0) == "segment too short"
