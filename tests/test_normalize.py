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


def test_reversed_c_becomes_open_o():
    assert normalize("ↄkↄ Ↄ", lowercase=False) == "ɔkɔ Ɔ"


def test_bracketed_annotations_are_dropped():
    s = "[10] Du [×] ahodoɔ [10] du [=] ma yɛn [100] ɔha."
    assert normalize(s) == "du ahodoɔ du ma yɛn ɔha."
    assert not find_digits(normalize(s))
    assert normalize("Abɔ [5:00pm] awia\n\nnnɔn") == "abɔ awia nnɔn"
    assert "[10]" in normalize("[10] du", drop_brackets=False)


def test_to_model_text():
    from akantts.normalize import to_model_text

    allowed = set("abdefghiklmnoprstuwy ɛɔ'-.,?!")
    assert to_model_text('Ɛyɛ "dɛ": yie; (ana)?', allowed) == "ɛyɛ dɛ, yie, ana?"
    assert to_model_text("Ɛyɛ [5] dɛ", allowed) == "ɛyɛ dɛ"  # bracketed digits dropped first
    assert to_model_text("afe 2006", allowed) is None  # bare digits
    assert to_model_text("Jesu yɛ", allowed) is None  # j is outside the vocabulary
    assert to_model_text("[1]", allowed) is None  # nothing left


def test_find_digits():
    assert find_digits("yɛn ho 2026")
    assert not find_digits("yɛn ho")


def test_charset_counts_akan_letters():
    cs = charset(["ɛyɛ", "ɔkɔ"])
    assert cs["ɛ"] == 2 and cs["ɔ"] == 2
