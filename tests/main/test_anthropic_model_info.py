"""
Tests for Anthropic Claude ModelInfo registry entries.

Guards against the same class of bug as #997/#1001 (Gemini) and #1015
(Claude 3.7 / 4 / 4.5): an enum value with no MODEL_INFO entry silently
falls back to the 16k-context, $0-cost defaults.
"""

import pytest

from langroid.language_models.model_info import (
    MODEL_INFO,
    AnthropicModel,
    ModelInfo,
    ModelProvider,
    get_model_info,
)


@pytest.mark.parametrize("model", list(AnthropicModel))
def test_anthropic_model_info_registered(model: AnthropicModel) -> None:
    assert model.value in MODEL_INFO, (
        f"{model.name} ({model.value}) has no MODEL_INFO entry; "
        f"users would silently get 16k-context / $0-cost defaults."
    )
    info = MODEL_INFO[model.value]
    assert isinstance(info, ModelInfo)
    assert info.provider == ModelProvider.ANTHROPIC
    assert info.name == model.value
    assert info.context_length >= 200_000
    assert info.max_output_tokens > 0
    assert info.input_cost_per_million > 0
    assert info.output_cost_per_million > 0


def test_claude_3_7_sonnet_extended_thinking_output() -> None:
    info = get_model_info(AnthropicModel.CLAUDE_3_7_SONNET.value)
    assert info.context_length == 200_000
    assert info.max_output_tokens == 64_000


@pytest.mark.parametrize(
    "model",
    [
        AnthropicModel.CLAUDE_4_SONNET,
        AnthropicModel.CLAUDE_4_5_SONNET,
    ],
)
def test_claude_4x_sonnet_pricing(model: AnthropicModel) -> None:
    info = MODEL_INFO[model.value]
    assert info.input_cost_per_million == 3.0
    assert info.output_cost_per_million == 15.0
    assert info.max_output_tokens == 64_000


@pytest.mark.parametrize(
    "model",
    [
        AnthropicModel.CLAUDE_4_OPUS,
        AnthropicModel.CLAUDE_4_5_OPUS,
    ],
)
def test_claude_4x_opus_pricing(model: AnthropicModel) -> None:
    info = MODEL_INFO[model.value]
    assert info.input_cost_per_million == 15.0
    assert info.output_cost_per_million == 75.0
    assert info.max_output_tokens == 32_000


@pytest.mark.parametrize(
    "model",
    [
        AnthropicModel.CLAUDE_4_HAIKU,
        AnthropicModel.CLAUDE_4_5_HAIKU,
    ],
)
def test_claude_4x_haiku_pricing(model: AnthropicModel) -> None:
    info = MODEL_INFO[model.value]
    assert info.input_cost_per_million == 1.0
    assert info.output_cost_per_million == 5.0
    assert info.max_output_tokens == 64_000
