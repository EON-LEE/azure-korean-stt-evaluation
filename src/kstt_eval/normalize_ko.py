"""Korean text normalization for scoring.

The same pipeline is applied to references and hypotheses. Raw texts are kept elsewhere.
Changing any rule here requires bumping ``NORMALIZER_VERSION``.

Rules (ko-norm-v1):
1. Unicode NFC.
2. KsponSpeech transcription markup, only if present:
   - dual transcription ``(spelling)/(pronunciation)`` -> one side, chosen by ``dual``
   - noise/event tags ``b/ l/ o/ n/ u/`` -> removed
   - filler marker ``word/`` and markers ``+ *`` -> marker removed, word kept
3. Lowercase (Latin letters).
4. Remove every character in Unicode categories P* (punctuation) and S* (symbols).
5. Collapse whitespace to single spaces and strip.

Fillers that remain in the reference are kept. No alias dictionary, number conversion
or reference-derived rules are applied.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

NORMALIZER_VERSION = "ko-norm-v1"

DualMode = Literal["spelling", "pronunciation"]

_DUAL_RE = re.compile(r"\(([^()]*)\)\s*/\s*\(([^()]*)\)")
_NOISE_TAG_RE = re.compile(r"(?<![0-9A-Za-z])[blonu]/")
_MARKUP_HINT_RE = re.compile(r"[()/+*]|(?<![0-9A-Za-z])[blonu]/")
_WS_RE = re.compile(r"\s+")


def has_kspon_markup(text: str) -> bool:
    """True if the text still contains KsponSpeech transcription markup characters."""
    return bool(_MARKUP_HINT_RE.search(text))


def strip_kspon_markup(text: str, dual: DualMode = "spelling") -> str:
    group = 1 if dual == "spelling" else 2
    text = _DUAL_RE.sub(lambda m: m.group(group), text)
    text = _NOISE_TAG_RE.sub(" ", text)
    return text


def _remove_punct_and_symbols(text: str) -> str:
    return "".join(
        " " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text
    )


def normalize(text: str | None, dual: DualMode = "spelling") -> str:
    """Normalize text to the word-level form used for CER/WER."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = strip_kspon_markup(text, dual)
    text = text.lower()
    text = _remove_punct_and_symbols(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def remove_spaces(text: str) -> str:
    return _WS_RE.sub("", text)
