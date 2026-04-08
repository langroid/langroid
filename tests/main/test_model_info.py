"""Tests for langroid.language_models.model_info helpers."""

import pytest

from langroid.language_models.model_info import (
    GeminiModel,
    _normalize_gemini_model_name,
)


# ---------------------------------------------------------------------------
# _normalize_gemini_model_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        # ---- canonical names are returned unchanged ----
        ("gemini-2.5-flash", "gemini-2.5-flash"),
        ("gemini-2.5-pro", "gemini-2.5-pro"),
        ("gemini-2.0-flash", "gemini-2.0-flash"),
        ("gemini-1.5-pro", "gemini-1.5-pro"),
        # canonical name that already ends with "-exp"
        (
            GeminiModel.GEMINI_2_FLASH_THINKING.value,
            GeminiModel.GEMINI_2_FLASH_THINKING.value,
        ),
        # ---- provider-prefixed variants ----
        ("google/gemini-2.5-flash", "gemini-2.5-flash"),
        ("vertex_ai/gemini-2.5-pro", "gemini-2.5-pro"),
        # ---- simple keyword suffixes ----
        ("gemini-2.5-flash-preview", "gemini-2.5-flash"),
        ("gemini-2.5-pro-preview", "gemini-2.5-pro"),
        ("gemini-2.5-flash-latest", "gemini-2.5-flash"),
        ("gemini-2.5-flash-experimental", "gemini-2.5-flash"),
        # ---- dated preview variants (keyword + date) ----
        ("gemini-2.5-flash-preview-05-20", "gemini-2.5-flash"),
        ("gemini-2.5-flash-lite-preview-06-17", "gemini-2.5-flash-lite"),
        ("gemini-2.5-pro-preview-03-25", "gemini-2.5-pro"),
        # ---- date-only suffix on a canonical name that contains "-exp" ----
        # "gemini-2.0-flash-thinking-exp" is canonical; "-01-21" is the date variant
        (
            "gemini-2.0-flash-thinking-exp-01-21",
            GeminiModel.GEMINI_2_FLASH_THINKING.value,
        ),
        (
            "gemini-2.0-flash-thinking-exp-12-31",
            GeminiModel.GEMINI_2_FLASH_THINKING.value,
        ),
        # ---- non-Gemini models return None ----
        ("gpt-4o", None),
        ("claude-3-opus-latest", None),
        # ---- completely unknown Gemini variant returns None ----
        ("gemini-99-ultra", None),
    ],
)
def test_normalize_gemini_model_name(model: str, expected: str | None) -> None:
    assert _normalize_gemini_model_name(model) == expected


def test_normalize_gemini_all_canonical_names_are_stable() -> None:
    """Every canonical GeminiModel value must normalize to itself."""
    for member in GeminiModel:
        result = _normalize_gemini_model_name(member.value)
        assert (
            result == member.value
        ), f"Canonical name {member.value!r} normalized to {result!r}"
