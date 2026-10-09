import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akantts.metrics import align, cer, clean, edit_distance, vowel_harmony_confusions, wer  # noqa: E402


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance(["a", "b"], ["a", "c"]) == 1


def test_clean_ignores_case_punctuation_and_letter_variants():
    assert clean("Ɛyɛ, [5] dɛ!") == "ɛyɛ dɛ"
    assert clean("εyε dɛ") == "ɛyɛ dɛ"


def test_cer_and_wer():
    assert cer("Ɛyɛ dɛ.", "ɛyɛ dɛ") == 0.0  # formatting differences are free
    assert cer("abcd", "abxd") == pytest.approx(0.25)
    assert cer("abc", "") == 1.0
    assert wer("me kɔ fie", "me kɔ fi") == pytest.approx(1 / 3)
    assert cer("", "abc") == 3.0  # empty reference does not divide by zero


def test_align_matches_and_substitutes():
    assert align("abc", "abc") == [("match", "a", "a"), ("match", "b", "b"), ("match", "c", "c")]
    assert align("abc", "axc") == [("match", "a", "a"), ("sub", "b", "x"), ("match", "c", "c")]
    assert align("ab", "abx") == [("match", "a", "a"), ("match", "b", "b"), ("ins", None, "x")]
    assert align("abx", "ab") == [("match", "a", "a"), ("match", "b", "b"), ("del", "x", None)]


def test_vowel_harmony_confusions_counts_only_the_atr_pairs():
    assert vowel_harmony_confusions("ɛyɛ", "eye") == {"e_to_ɛ": 0, "ɛ_to_e": 2, "o_to_ɔ": 0, "ɔ_to_o": 0}
    assert vowel_harmony_confusions("do", "dɔ") == {"e_to_ɛ": 0, "ɛ_to_e": 0, "o_to_ɔ": 1, "ɔ_to_o": 0}
    assert vowel_harmony_confusions("kitten", "sitting") == {
        "e_to_ɛ": 0, "ɛ_to_e": 0, "o_to_ɔ": 0, "ɔ_to_o": 0}
    # Greek lookalikes and case are normalized away first, same as cer/wer.
    assert vowel_harmony_confusions("Ɛyɛ", "εyε") == {"e_to_ɛ": 0, "ɛ_to_e": 0, "o_to_ɔ": 0, "ɔ_to_o": 0}
