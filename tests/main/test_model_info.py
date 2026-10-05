"""Tests for langroid.language_models.model_info helpers."""

import pytest

from langroid.language_models.model_info import (
    AnthropicModel,
    GeminiModel,
    ModelProvider,
    _normalize_gemini_model_name,
    _strip_dated_snapshot,
    get_model_info,
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
        # Malformed calendar dates must not expose canonical model metadata.
        ("gemini-2.0-flash-thinking-exp-00-00", None),
        ("gemini-2.0-flash-thinking-exp-13-01", None),
        ("gemini-2.0-flash-thinking-exp-12-32", None),
        ("gemini-2.0-flash-thinking-exp-02-30", None),
        ("gemini-2.0-flash-thinking-exp-04-31", None),
        (
            "gemini-2.0-flash-thinking-exp-02-29",
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
        # Lookalike dates (unicode digits, trailing newline/control chars)
        # must NOT take the date-stripping path: only a strict ASCII
        # "-MM-DD" at the true end of the name counts. A lookalike date on
        # an "-exp" canonical name therefore stays unrecognized.
        ("gemini-2.0-flash-thinking-exp-０１-２１", None),
        ("gemini-2.0-flash-thinking-exp-01-21\n", None),
        # Outside the date-stripped case, the first-occurrence keyword
        # split keeps parity with historical behavior: any name with a
        # canonical prefix before the first keyword still normalizes,
        # regardless of what follows the keyword.
        ("gemini-2.5-flash-preview-０５-２０", "gemini-2.5-flash"),
        ("gemini-2.5-flash-preview-05-20\n", "gemini-2.5-flash"),
        ("gemini-2.5-flash-preview\n", "gemini-2.5-flash"),
        ("gemini-2.5-flash-preview-05-20\x00", "gemini-2.5-flash"),
        ("gemini-2.5-flash-preview-junk", "gemini-2.5-flash"),
        ("gemini-2.5-flash-previewer", "gemini-2.5-flash"),
        ("gemini-2.5-flash-preview-05-20-99-99", "gemini-2.5-flash"),
        # Empty input and case variants are rejected: matching is exact
        # and case-sensitive.
        ("", None),
        ("Gemini-2.5-flash-preview-05-20", None),
        ("gemini-2.5-FLASH-preview-05-20", None),
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


@pytest.mark.parametrize(
    "dated,base",
    [
        # Anthropic: -YYYYMMDD
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        # OpenAI: -YYYY-MM-DD
        ("gpt-4o-2024-08-06", "gpt-4o"),
        # behind a provider prefix
        ("anthropic/claude-haiku-4-5-20251001", "claude-haiku-4-5"),
    ],
)
def test_dated_snapshot_resolves_to_base_model(dated: str, base: str):
    """A dated snapshot of a KNOWN model inherits that model's info.

    Providers ship dated snapshots constantly; a hard-coded table cannot keep
    up, and the old behaviour silently gave them a 16k context.
    """
    base_info = get_model_info(base)
    # Anchor the comparison: two UNKNOWN names both resolve to the same
    # default ModelInfo, so comparing them alone would pass vacuously if the
    # base entry were removed or renamed.
    assert base_info.provider != ModelProvider.UNKNOWN
    assert base_info.name == base
    assert get_model_info(dated) == base_info


@pytest.mark.parametrize(
    "unknown",
    [
        "some-new-model-20260101",  # base is not a known model
        "gpt-4o-2024-13-45",  # not a real date
        "claude-haiku-4-5-2025100",  # too few digits
        # Unicode digit lookalikes: Python's \d matches these, so a naive
        # regex would strip "-20２4-08-06" and hand this ID gpt-4o's limits
        # and prices. The digit classes are ASCII-only to prevent that.
        "gpt-4o-20２４-08-06",  # fullwidth TWO, FOUR in the year
        "gpt-4o-2024-08-1０",  # fullwidth ZERO in the day
        # Shaped like a date but impossible, so not a snapshot of anything.
        "gpt-4o-2024-02-31",  # February has no 31st
        "gpt-4o-2023-02-29",  # 2023 is not a leap year
        "gpt-4o-20240230",  # same, in the compact form
        # Mixed separators: no provider ships these, so treating them as
        # dates would only let malformed IDs inherit real prices.
        "gpt-4o-2024-0806",
        "gpt-4o-202408-06",
    ],
)
def test_unknown_dated_model_does_not_borrow_info(unknown: str):
    """Stripping a date must not GUESS: an unknown base stays unknown."""
    info = get_model_info(unknown)
    assert info.provider == ModelProvider.UNKNOWN
    assert info.input_cost_per_million == 0.0


def test_leap_day_snapshot_still_resolves():
    """The calendar check must not reject a date that is real."""
    assert get_model_info("gpt-4o-20240229") == get_model_info("gpt-4o")


def test_dated_gemini_name_keeps_the_gemini_policy():
    """The generic stripper must not override the Gemini normalizer.

    `_normalize_gemini_model_name` deliberately refuses to guess a bare-dated
    name (`gemini-2.5-pro-03-25` must not become `gemini-2.5-pro`). A generic
    date-stripper that resolved `gemini-2.5-pro-2025-03-25` would reverse that
    ruling by the back door -- and silently, since it also flips the model to
    "known", which turns on `rename_params` and extra-body filtering.
    """
    assert _strip_dated_snapshot("gemini-2.5-pro-2025-03-25") is None
    info = get_model_info("gemini-2.5-pro-2025-03-25")
    assert info.provider == ModelProvider.UNKNOWN
    # the canonical name is of course still resolved
    assert get_model_info("gemini-2.5-pro").provider == ModelProvider.GOOGLE


def test_haiku_4_5_cached_price_is_discounted():
    """A cached read must not be billed at the full input rate.

    `OpenAIGPT.chat_cost()` treats a zero `cached_cost_per_million` as
    "missing" and falls back to `input_cost_per_million`, so leaving it unset
    would overstate the cached portion of every Haiku 4.5 request by 10x.
    """
    info = get_model_info(AnthropicModel.CLAUDE_4_5_HAIKU)
    assert info.input_cost_per_million == 1.00
    assert info.cached_cost_per_million == 0.10
    # and a dated snapshot inherits the discount
    assert get_model_info("claude-haiku-4-5-20251001").cached_cost_per_million == 0.10
