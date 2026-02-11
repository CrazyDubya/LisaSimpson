"""Tests for input_safety: normalization, zero-width, hidden chars, emoji limit."""

import unicodedata

import pytest

from deliberative_agent.input_safety import (
    normalize_for_llm,
    strip_zero_width_and_invisible,
    detect_hidden_or_abusive,
    sanitize_string_for_prompt,
    SafetyReport,
)


class TestNormalize:
    def test_strip_zero_width(self):
        zw = "\u200bhello\u200cworld\u2060"
        assert strip_zero_width_and_invisible(zw) == "helloworld"

    def test_normalize_unicode(self):
        out = normalize_for_llm("café")
        assert "é" in out or unicodedata.normalize("NFKC", "é") in out

    def test_collapse_whitespace(self):
        out = normalize_for_llm("  a   b\n\t  c  ")
        assert out == "a b c"

    def test_cyrillic_lookalikes(self):
        # а is Cyrillic, should be replaced with ASCII a
        out = normalize_for_llm("heаllo")  # second 'a' is Cyrillic U+0430
        assert "а" not in out or out != "heаllo"

    def test_control_chars_removed(self):
        out = normalize_for_llm("hello\x00world\x1f")
        assert "\x00" not in out and "\x1f" not in out

    def test_empty_and_none(self):
        assert normalize_for_llm("") == ""
        assert normalize_for_llm(None) == ""


class TestDetect:
    def test_detect_zero_width(self):
        findings = detect_hidden_or_abusive("hi\u200bthere")
        assert "zero_width" in " ".join(findings).lower()

    def test_detect_control(self):
        findings = detect_hidden_or_abusive("hi\x01")
        assert "control" in " ".join(findings).lower()

    def test_sanitize_returns_report(self):
        s, report = sanitize_string_for_prompt("  normal  text  ")
        assert isinstance(report, SafetyReport)
        assert report.normalized == "normal text"
        assert s == "normal text"
