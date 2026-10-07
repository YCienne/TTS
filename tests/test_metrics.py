import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akantts.metrics import cer, clean, edit_distance, wer  # noqa: E402


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
