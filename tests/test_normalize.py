import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akantts.normalize import charset, find_digits, normalize  # noqa: E402


def test_greek_lookalikes_become_akan_letters():
    assert normalize("εyε", lowercase=False) == "ɛyɛ"
    assert normalize("Εkɔ", lowercase=False) == "Ɛkɔ"
    assert normalize("ͻ", lowercase=False) == "ɔ"


def test_output_is_nfc():
    decomposed = unicodedata.normalize("NFD", "ɛ́")
    assert unicodedata.is_normalized("NFC", normalize(decomposed))


def test_lowercase_and_whitespace():
    assert normalize("  Ɛyɛ   Hɔ \n") == "ɛyɛ hɔ"
    assert normalize("Ɛyɛ", lowercase=False) == "Ɛyɛ"


def test_punctuation_and_zero_width():
    assert normalize("w’ani​") == "w'ani"


def test_idempotent():
    s = "Εyε hɔ, ɔkɔ’"
    assert normalize(normalize(s)) == normalize(s)


def test_find_digits():
    assert find_digits("yɛn ho 2026")
    assert not find_digits("yɛn ho")


def test_charset_counts_akan_letters():
    cs = charset(["ɛyɛ", "ɔkɔ"])
    assert cs["ɛ"] == 2 and cs["ɔ"] == 2
