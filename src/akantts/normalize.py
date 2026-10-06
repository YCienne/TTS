"""Text normalization for Akan (Twi/Fante) TTS.

Scope is deliberately small: Unicode hygiene, canonical spelling of the Akan
letters ɛ and ɔ, and removal of bracketed annotations such as ``[10]``. In the
WAXAL Twi scripts the spoken number words are written next to the bracketed
digits, so the brackets are not read aloud (checked on a handful of samples;
``scripts/audit_data.py`` counts what is left over). Number-word expansion is
not implemented.
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


def to_model_text(text: str, allowed):
    """``normalize`` plus punctuation clean-up for the model, or None if the text is unusable.

    ``;`` and ``:`` become commas, quotes and brackets are dropped. Text is rejected if it still
    contains a character outside ``allowed`` (the model vocabulary plus kept punctuation) or any
    digit, because digits are not spoken as written in these recordings.
    """
    text = _WS.sub(" ", normalize(text).translate(_TO_COMMA).translate(_DROPPED)).strip()
    if not text or find_digits(text) or any(c not in allowed for c in text):
        return None
    return text


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
