"""Text normalization for Akan (Twi/Fante) TTS.

Scope is deliberately small: Unicode hygiene, canonical spelling of the Akan
letters ɛ and ɔ, and removal of bracketed annotations such as ``[10]``. In the
WAXAL Twi scripts the spoken number words are written next to the bracketed
digits, so the brackets are not read aloud (checked on a handful of samples;
``scripts/audit_data.py`` counts what is left over). Number-word expansion is
not implemented.

Kept in sync by hand with src/akantts/normalize.py in the main repository
(https://github.com/YCienne/TTS) -- this Space is a standalone deployment, so
it carries its own copy instead of a package dependency.
"""
import re
import unicodedata

# Characters typed, OCR'd or mis-encoded in place of the Akan letters.
_LETTER_MAP = {
    "ε": "ɛ",  # Greek small epsilon
    "Ε": "Ɛ",  # Greek capital epsilon
    "ͻ": "ɔ",  # Greek small reversed lunate sigma
    "Ͻ": "Ɔ",  # Greek capital reversed lunate sigma
    "ↄ": "ɔ",  # Latin small reversed c (65 times in WAXAL twi_tts train)
    "Ↄ": "Ɔ",  # Latin capital reversed C
}

_PUNCT_MAP = {
    "‘": "'",
    "’": "'",
    "ʼ": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
}

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"))
_BRACKETED = re.compile(r"\[[^\]]*\]")
_WS = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")


def normalize(text: str, lowercase: bool = True, drop_brackets: bool = True) -> str:
    """Return ``text`` in the canonical form used for training and inference.

    NFC first, so combining marks are composed consistently. Tone diacritics,
    when present, are kept; the orthography does not normally mark tone.
    """
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    if drop_brackets:
        text = _BRACKETED.sub(" ", text)
    text = "".join(_LETTER_MAP.get(c, c) for c in text)
    text = "".join(_PUNCT_MAP.get(c, c) for c in text)
    if lowercase:
        text = text.lower()
    return _WS.sub(" ", text).strip()


_TO_COMMA = {ord(";"): ",", ord(":"): ","}
_DROPPED = {ord(c): None for c in '"()`\\/'}


def to_inference_text(text: str, vocab):
    """Forgiving counterpart of ``to_model_text`` for synthesis: ``(text, removed)``.

    Applies the same normalization, then drops every character the model's vocabulary cannot
    represent instead of rejecting the text. ``removed`` lists the dropped characters (digits
    among them: numbers are not expanded) so the caller can warn.
    """
    text = normalize(text).translate(_TO_COMMA).translate(_DROPPED)
    removed = sorted({c for c in text if c not in vocab})
    text = _WS.sub(" ", "".join(c for c in text if c in vocab)).strip()
    return text, removed


def find_digits(text: str) -> bool:
    """True if the text still contains digits, i.e. needs number expansion."""
    return bool(_DIGIT.search(text))
