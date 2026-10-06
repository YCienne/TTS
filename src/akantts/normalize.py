"""Text normalization for Akan (Twi/Fante) TTS.

Scope is deliberately small: Unicode hygiene and canonical spelling of the Akan
letters ɛ and ɔ. Number and abbreviation expansion is not implemented, so
``find_digits`` lets the data audit measure how often it would matter.
"""
import re
import unicodedata

# Characters people type or OCR in place of the Akan letters.
_LETTER_MAP = {
    "ε": "ɛ",  # Greek small epsilon
    "Ε": "Ɛ",  # Greek capital epsilon
    "ͻ": "ɔ",  # Greek small reversed lunate sigma
    "Ͻ": "Ɔ",  # Greek capital reversed lunate sigma
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
_WS = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")


def normalize(text: str, lowercase: bool = True) -> str:
    """Return ``text`` in the canonical form used for training and inference.

    NFC first, so combining marks are composed consistently. Tone diacritics,
    when present, are kept; the orthography does not normally mark tone.
    """
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    text = "".join(_LETTER_MAP.get(c, c) for c in text)
    text = "".join(_PUNCT_MAP.get(c, c) for c in text)
    if lowercase:
        text = text.lower()
    return _WS.sub(" ", text).strip()


def find_digits(text: str) -> bool:
    """True if the text still contains digits, i.e. needs number expansion."""
    return bool(_DIGIT.search(text))


def charset(texts) -> dict:
    """Character frequency over ``texts`` after normalization."""
    counts: dict = {}
    for t in texts:
        for c in normalize(t):
            counts[c] = counts.get(c, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
