"""Tests for langroid.language_models.model_info helpers."""

import pytest

from langroid.language_models.model_info import (
    GeminiModel,
    _normalize_gemini_model_name,
)


@pytest.mark.parametrize(
    "model, expected",
    [
        # Canonical names round-trip.
        ("gemini-2.5-flash", "gemini-2.5-flash"),
        ("gemini-2.5-pro", "gemini-2.5-pro"),
        ("gemini-2.0-flash", "gemini-2.0-flash"),
        ("gemini-1.5-pro", "gemini-1.5-pro"),
        # Canonical name that already ends in "-exp".
        (
            GeminiModel.GEMINI_2_FLASH_THINKING.value,
            GeminiModel.GEMINI_2_FLASH_THINKING.value,
        ),
        # Canonical name that contains both "-exp" and a date.
        (
            GeminiModel.GEMINI_2_PRO.value,
            GeminiModel.GEMINI_2_PRO.value,
        ),
        # Provider-prefixed.
        ("google/gemini-2.5-flash", "gemini-2.5-flash"),
        ("vertex_ai/gemini-2.5-pro", "gemini-2.5-pro"),
        # Plain keyword suffixes.
        ("gemini-2.5-flash-preview", "gemini-2.5-flash"),
        ("gemini-2.5-pro-preview", "gemini-2.5-pro"),
        ("gemini-2.5-flash-latest", "gemini-2.5-flash"),
        ("gemini-2.5-flash-experimental", "gemini-2.5-flash"),
        # Keyword + trailing date.
        ("gemini-2.5-flash-preview-05-20", "gemini-2.5-flash"),
        ("gemini-2.5-flash-lite-preview-06-17", "gemini-2.5-flash-lite"),
        ("gemini-2.5-pro-preview-03-25", "gemini-2.5-pro"),
        # Date-only suffix on a canonical "-exp" name (the bug from #995).
        (
            "gemini-2.0-flash-thinking-exp-01-21",
            GeminiModel.GEMINI_2_FLASH_THINKING.value,
        ),
        (
            "gemini-2.0-flash-thinking-exp-12-31",
            GeminiModel.GEMINI_2_FLASH_THINKING.value,
        ),
        # Non-Gemini -> None.
        ("gpt-4o", None),
        ("claude-3-opus-latest", None),
        # Unknown Gemini variants -> None. Only the 02-05 dated pro-exp is
        # canonical; we don't guess at the nearest match.
        ("gemini-99-ultra", None),
        ("gemini-2.0-pro-exp-03-07", None),
        # Bare-dated unknown names (date but no -exp/-preview/-experimental/
        # -latest keyword) must NOT be guessed as the nearest canonical model.
        ("gemini-2.5-pro-03-25", None),
        ("gemini-2.0-flash-lite-01-21", None),
        ("gemini-3-pro-12-01", None),
        # Hostile lookalikes must NOT normalize: only strict ASCII "-MM-DD"
        # dates at the true end of the name count as dated variants.
        # Fullwidth (unicode) date digits.
        ("gemini-2.5-flash-preview-０５-２０", None),
        # Trailing newline / control characters.
        ("gemini-2.5-flash-preview-05-20\n", None),
        ("gemini-2.5-flash-preview\n", None),
        ("gemini-2.0-flash-thinking-exp-01-21\n", None),
        ("gemini-2.5-flash-preview-05-20\x00", None),
        # Arbitrary junk after the keyword suffix.
        ("gemini-2.5-flash-preview-junk", None),
        ("gemini-2.5-flash-previewer", None),
        ("gemini-2.5-flash-preview-05-20-99-99", None),
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
