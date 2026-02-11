"""
Input safety for LLM-facing content: normalization, hidden characters, prompt injection mitigation.

Protects against:
- Zero-width and invisible characters (hiding content, bypassing filters)
- Mixed scripts (e.g. Cyrillic lookalikes for ASCII)
- Excessive or abusive whitespace
- Unicode normalization issues (NFKC)
- Optional: emoji overload, control characters

Use for any user-provided or external strings that are interpolated into LLM prompts.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Zero-width and invisible character ranges/codepoints (common ones)
# https://util.unicode.org/UnicodeJsps/list-unicodeset.jsp?a=%5B%3AInvisible%3DYes%3A%5D
ZERO_WIDTH_OR_INVISIBLE = (
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER (invisible)
    "\u2061",  # FUNCTION APPLICATION
    "\u2062",  # INVISIBLE TIMES
    "\u2063",  # INVISIBLE SEPARATOR
    "\u2064",  # INVISIBLE PLUS
    "\ufeff",  # BOM / ZERO WIDTH NO-BREAK SPACE
    "\u00ad",  # SOFT HYPHEN (invisible)
    "\u034f",  # COMBINING GRAPHEME JOINER
    "\u061c",  # ARABIC LETTER MARK
    "\u115f",  # HANGUL CHOSEONG FILLER
    "\u1160",  # HANGUL JUNGSEONG FILLER
    "\u17b4",  # KHMER VOWEL INHERENT AQ
    "\u17b5",  # KHMER VOWEL INHERENT AA
    "\u180e",  # MONGOLIAN VOWEL SEPARATOR
)
ZERO_WIDTH_PATTERN = re.compile("|".join(re.escape(c) for c in ZERO_WIDTH_OR_INVISIBLE))

# Control characters (Cc, Cf, Co, Cn categories that are controls)
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Excessive whitespace (collapse to single space)
WHITESPACE_PATTERN = re.compile(r"\s+")

# ASCII lookalikes in Cyrillic (common homoglyphs)
CYRILLIC_LOOKALIKES = str.maketrans(
    "аеорсухАВЕКМНОРСТУХ",  # Cyrillic
    "aeopcyxABEKMHOPCTYX",  # ASCII (best-effort)
)
# Add more if needed; this covers common Latin lookalikes.


@dataclass
class SafetyReport:
    """Result of checking text for hidden or abusive content."""

    ok: bool
    normalized: str
    findings: List[str] = field(default_factory=list)
    replaced_emoji_count: int = 0


def strip_zero_width_and_invisible(text: str) -> str:
    """Remove zero-width and invisible characters."""
    if not text:
        return text
    return ZERO_WIDTH_PATTERN.sub("", text)


def strip_control_characters(text: str) -> str:
    """Remove ASCII and common control characters."""
    if not text:
        return text
    return CONTROL_CHARS_PATTERN.sub("", text)


def normalize_whitespace(text: str, collapse: bool = True) -> str:
    """Collapse runs of whitespace to a single space; strip leading/trailing."""
    if not text:
        return text
    if collapse:
        return WHITESPACE_PATTERN.sub(" ", text).strip()
    return text.strip()


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Unicode normalize (NFKC by default: compatibility decomposition + canonical composition)."""
    if not text:
        return text
    return unicodedata.normalize(form, text)


def replace_cyrillic_lookalikes(text: str) -> str:
    """Replace common Cyrillic letters that look like ASCII (reduce homograph abuse)."""
    if not text:
        return text
    return text.translate(CYRILLIC_LOOKALIKES)


def _is_emoji_like(c: str) -> bool:
    """Heuristic: emoji blocks and common symbol ranges."""
    o = ord(c)
    if o >= 0x1F300 and o <= 0x1F9FF:
        return True
    if o >= 0x2600 and o <= 0x26FF:
        return True
    if o >= 0x2700 and o <= 0x27BF:
        return True
    cat = unicodedata.category(c)
    return cat in ("So", "Sk")


def limit_emoji(text: str, max_emoji: int = 10, replacement: str = " ") -> Tuple[str, int]:
    """
    Replace emoji beyond max_emoji with replacement; return (new_string, count_replaced).
    """
    if not text or max_emoji < 0:
        return text, 0
    kept = 0
    replaced = 0
    result = []
    for c in text:
        if _is_emoji_like(c):
            if kept < max_emoji:
                result.append(c)
                kept += 1
            else:
                result.append(replacement)
                replaced += 1
        else:
            result.append(c)
    return "".join(result), replaced


def normalize_for_llm(
    text: str,
    *,
    normalize_unicode_form: str = "NFKC",
    strip_invisible: bool = True,
    strip_control: bool = True,
    collapse_whitespace: bool = True,
    apply_cyrillic_replacement: bool = True,
    max_emoji: Optional[int] = 10,
) -> str:
    """
    Normalize a string before sending to the LLM or using in prompts.

    Applies (in order): unicode normalize, strip invisible/control,
    collapse whitespace, optional Cyrillic lookalike replacement, optional emoji limit.
    """
    if not text or not isinstance(text, str):
        return "" if text is None else str(text)
    s = text
    s = normalize_unicode(s, form=normalize_unicode_form)
    if strip_invisible:
        s = strip_zero_width_and_invisible(s)
    if strip_control:
        s = strip_control_characters(s)
    if collapse_whitespace:
        s = normalize_whitespace(s, collapse=True)
    if apply_cyrillic_replacement:
        s = replace_cyrillic_lookalikes(s)
    if max_emoji is not None and max_emoji >= 0:
        s, _ = limit_emoji(s, max_emoji=max_emoji, replacement=" ")
        s = normalize_whitespace(s, collapse=True)
    return s


def detect_hidden_or_abusive(text: str) -> List[str]:
    """
    Detect potential hidden or abusive content (zero-width, control chars, excessive emoji).
    Returns a list of short findings for logging or rejection.
    """
    findings: List[str] = []
    if not text:
        return findings
    if ZERO_WIDTH_PATTERN.search(text):
        findings.append("zero_width_or_invisible_chars")
    if CONTROL_CHARS_PATTERN.search(text):
        findings.append("control_characters")
    _, emoji_replaced = limit_emoji(text, max_emoji=20)
    if emoji_replaced > 0:
        findings.append("excessive_emoji")
    # Mixed script: very rough check (e.g. Cyrillic + Latin in same word could be homograph)
    if replace_cyrillic_lookalikes(text) != text:
        findings.append("cyrillic_lookalikes_present")
    return findings


def sanitize_string_for_prompt(
    text: str,
    *,
    max_length: Optional[int] = 50_000,
    reject_if_abusive: bool = False,
) -> Tuple[str, SafetyReport]:
    """
    Sanitize a string for safe use in an LLM prompt. Optionally reject if findings are severe.

    Returns (normalized_string, safety_report). If reject_if_abusive and findings exist,
    still returns normalized string but report.ok is False (caller can reject).
    """
    normalized = normalize_for_llm(text)
    if max_length is not None and len(normalized) > max_length:
        normalized = normalized[:max_length]
        findings = ["truncated_to_max_length"]
    else:
        findings = detect_hidden_or_abusive(text)
    ok = not (reject_if_abusive and findings)
    return normalized, SafetyReport(ok=ok, normalized=normalized, findings=findings)


# Delimiter to wrap user/content in prompts to reduce prompt injection surface
PROMPT_CONTENT_DELIMITER = "\n---END CONTENT---\n"


def wrap_content_for_prompt(content: str, prefix: str = "Content:") -> str:
    """Wrap content with a clear delimiter so the model treats it as data, not instructions."""
    sanitized, _ = sanitize_string_for_prompt(content, max_length=30_000)
    return f"{prefix}\n{sanitized}{PROMPT_CONTENT_DELIMITER}"
