from enum import Enum
from typing import Dict, List, Optional

from langroid.pydantic_v1 import BaseModel


class ModelProvider(str, Enum):
    """Enum for model providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    UNKNOWN = "unknown"


class ModelName(str, Enum):
    """Parent class for all model name enums"""

    pass


class OpenAIChatModel(ModelName):
    """Enum for OpenAI Chat models"""

    GPT3_5_TURBO = "gpt-3.5-turbo-1106"
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4o = "gpt-4o"
    GPT4o_MINI = "gpt-4o-mini"
    O1 = "o1"
    O1_MINI = "o1-mini"
    O3_MINI = "o3-mini"


class OpenAICompletionModel(str, Enum):
    """Enum for OpenAI Completion models"""

    DAVINCI = "davinci-002"
    BABBAGE = "babbage-002"


class AnthropicModel(ModelName):
    """Enum for Anthropic models"""

    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_3_OPUS = "claude-3-opus-latest"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"


class DeepSeekModel(ModelName):
    """Enum for DeepSeek models direct from DeepSeek API"""

    DEEPSEEK = "deepseek/deepseek-chat"
    DEEPSEEK_R1 = "deepseek/deepseek-reasoner"
    OPENROUTER_DEEPSEEK_R1 = "openrouter/deepseek/deepseek-r1"


class GeminiModel(ModelName):
    """Enum for Gemini models"""

    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_FLASH_8B = "gemini-1.5-flash-8b"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_2_PRO = "gemini-2.0-pro-exp-02-05"
    GEMINI_2_FLASH = "gemini-2.0-flash"
    GEMINI_2_FLASH_LITE = "gemini-2.0-flash-lite-preview"
    GEMINI_2_FLASH_THINKING = "gemini-2.0-flash-thinking-exp"


class OpenAI_API_ParamInfo(BaseModel):
    """
    Parameters exclusive to some models, when using OpenAI API
    """

    # model-specific params at top level
    params: Dict[str, List[str]] = dict(
        reasoning_effort=[
            OpenAIChatModel.O3_MINI.value,
        ],
    )
    # model-specific params in extra_body
    extra_parameters: Dict[str, List[str]] = dict(
        include_reasoning=[
            DeepSeekModel.OPENROUTER_DEEPSEEK_R1.value,
        ]
    )


class ModelInfo(BaseModel):
    """
    Consolidated information about LLM, related to capacity, cost and API
    idiosyncrasies. Reasonable defaults for all params in case there's no
    specific info available.
    """

    name: str = "unknown"
    provider: ModelProvider = ModelProvider.UNKNOWN
    context_length: int = 16_000
    max_cot_tokens: int = 0  # max chain of thought (thinking) tokens where applicable
    max_output_tokens: int = 8192  # Maximum number of output tokens - model dependent
    input_cost_per_million: float = 0.0  # Cost in USD per million input tokens
    output_cost_per_million: float = 0.0  # Cost in USD per million output tokens
    allows_streaming: bool = True  # Whether model supports streaming output
    allows_system_message: bool = True  # Whether model supports system messages
    rename_params: Dict[str, str] = {}  # Rename parameters for OpenAI API
    unsupported_params: List[str] = []
    has_structured_output: bool = False  # Does model API support structured output?
    has_tools: bool = True  # Does model API support tools/function-calling?
    needs_first_user_message: bool = False  # Does API need first msg to be from user?
    description: Optional[str] = None


# Model information registry
MODEL_INFO: Dict[str, ModelInfo] = {
    # OpenAI Models
    OpenAICompletionModel.DAVINCI.value: ModelInfo(
        name=OpenAICompletionModel.DAVINCI.value,
        provider=ModelProvider.OPENAI,
        context_length=4096,
        max_output_tokens=4096,
        input_cost_per_million=2.0,
        output_cost_per_million=2.0,
        description="Davinci-002",
    ),
    OpenAICompletionModel.BABBAGE.value: ModelInfo(
        name=OpenAICompletionModel.BABBAGE.value,
        provider=ModelProvider.OPENAI,
        context_length=4096,
        max_output_tokens=4096,
        input_cost_per_million=0.40,
        output_cost_per_million=0.40,
        description="Babbage-002",
    ),
    OpenAIChatModel.GPT3_5_TURBO.value: ModelInfo(
        name=OpenAIChatModel.GPT3_5_TURBO.value,
        provider=ModelProvider.OPENAI,
        context_length=16_385,
        max_output_tokens=4096,
        input_cost_per_million=0.50,
        output_cost_per_million=1.50,
        description="GPT-3.5 Turbo",
    ),
    OpenAIChatModel.GPT4.value: ModelInfo(
        name=OpenAIChatModel.GPT4.value,
        provider=ModelProvider.OPENAI,
        context_length=8192,
        max_output_tokens=8192,
        input_cost_per_million=30.0,
        output_cost_per_million=60.0,
        description="GPT-4 (8K context)",
    ),
    OpenAIChatModel.GPT4_TURBO.value: ModelInfo(
        name=OpenAIChatModel.GPT4_TURBO.value,
        provider=ModelProvider.OPENAI,
        context_length=128_000,
        max_output_tokens=4096,
        input_cost_per_million=10.0,
        output_cost_per_million=30.0,
        description="GPT-4 Turbo",
    ),
    OpenAIChatModel.GPT4o.value: ModelInfo(
        name=OpenAIChatModel.GPT4o.value,
        provider=ModelProvider.OPENAI,
        context_length=128_000,
        max_output_tokens=16_384,
        input_cost_per_million=2.5,
        output_cost_per_million=10.0,
        has_structured_output=True,
        description="GPT-4o (128K context)",
    ),
    OpenAIChatModel.GPT4o_MINI.value: ModelInfo(
        name=OpenAIChatModel.GPT4o_MINI.value,
        provider=ModelProvider.OPENAI,
        context_length=128_000,
        max_output_tokens=16_384,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        has_structured_output=True,
        description="GPT-4o Mini",
    ),
    OpenAIChatModel.O1.value: ModelInfo(
        name=OpenAIChatModel.O1.value,
        provider=ModelProvider.OPENAI,
        context_length=200_000,
        max_output_tokens=100_000,
        input_cost_per_million=15.0,
        output_cost_per_million=60.0,
        allows_streaming=True,
        allows_system_message=False,
        unsupported_params=["temperature"],
        rename_params={"max_tokens": "max_completion_tokens"},
        has_tools=False,
        description="O1 Reasoning LM",
    ),
    OpenAIChatModel.O1_MINI.value: ModelInfo(
        name=OpenAIChatModel.O1_MINI.value,
        provider=ModelProvider.OPENAI,
        context_length=128_000,
        max_output_tokens=65_536,
        input_cost_per_million=1.1,
        output_cost_per_million=4.4,
        allows_streaming=False,
        allows_system_message=False,
        unsupported_params=["temperature", "stream"],
        rename_params={"max_tokens": "max_completion_tokens"},
        has_tools=False,
        description="O1 Mini Reasoning LM",
    ),
    OpenAIChatModel.O3_MINI.value: ModelInfo(
        name=OpenAIChatModel.O3_MINI.value,
        provider=ModelProvider.OPENAI,
        context_length=200_000,
        max_output_tokens=100_000,
        input_cost_per_million=1.1,
        output_cost_per_million=4.4,
        allows_streaming=False,
        allows_system_message=False,
        unsupported_params=["temperature", "stream"],
        rename_params={"max_tokens": "max_completion_tokens"},
        has_tools=False,
        description="O3 Mini Reasoning LM",
    ),
    # Anthropic Models
    AnthropicModel.CLAUDE_3_5_SONNET.value: ModelInfo(
        name=AnthropicModel.CLAUDE_3_5_SONNET.value,
        provider=ModelProvider.ANTHROPIC,
        context_length=200_000,
        max_output_tokens=8192,
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        description="Claude 3.5 Sonnet",
    ),
    AnthropicModel.CLAUDE_3_OPUS.value: ModelInfo(
        name=AnthropicModel.CLAUDE_3_OPUS.value,
        provider=ModelProvider.ANTHROPIC,
        context_length=200_000,
        max_output_tokens=4096,
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
        description="Claude 3 Opus",
    ),
    AnthropicModel.CLAUDE_3_SONNET.value: ModelInfo(
        name=AnthropicModel.CLAUDE_3_SONNET.value,
        provider=ModelProvider.ANTHROPIC,
        context_length=200_000,
        max_output_tokens=4096,
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        description="Claude 3 Sonnet",
    ),
    AnthropicModel.CLAUDE_3_HAIKU.value: ModelInfo(
        name=AnthropicModel.CLAUDE_3_HAIKU.value,
        provider=ModelProvider.ANTHROPIC,
        context_length=200_000,
        max_output_tokens=4096,
        input_cost_per_million=0.25,
        output_cost_per_million=1.25,
        description="Claude 3 Haiku",
    ),
    # DeepSeek Models
    DeepSeekModel.DEEPSEEK.value: ModelInfo(
        name=DeepSeekModel.DEEPSEEK.value,
        provider=ModelProvider.DEEPSEEK,
        context_length=64_000,
        max_output_tokens=8_000,
        input_cost_per_million=0.27,
        output_cost_per_million=1.10,
        description="DeepSeek Chat",
    ),
    DeepSeekModel.DEEPSEEK_R1.value: ModelInfo(
        name=DeepSeekModel.DEEPSEEK_R1.value,
        provider=ModelProvider.DEEPSEEK,
        context_length=64_000,
        max_output_tokens=8_000,
        input_cost_per_million=0.55,
        output_cost_per_million=2.19,
        description="DeepSeek-R1 Reasoning LM",
    ),
    # Gemini Models
    GeminiModel.GEMINI_2_FLASH.value: ModelInfo(
        name=GeminiModel.GEMINI_2_FLASH.value,
        provider=ModelProvider.GOOGLE,
        context_length=1_056_768,
        max_output_tokens=8192,
        input_cost_per_million=0.10,
        output_cost_per_million=0.40,
        rename_params={"max_tokens": "max_completion_tokens"},
        description="Gemini 2.0 Flash",
    ),
    GeminiModel.GEMINI_2_FLASH_LITE.value: ModelInfo(
        name=GeminiModel.GEMINI_2_FLASH_LITE.value,
        provider=ModelProvider.GOOGLE,
        context_length=1_056_768,
        max_output_tokens=8192,
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
        rename_params={"max_tokens": "max_completion_tokens"},
        description="Gemini 2.0 Flash Lite Preview",
    ),
    GeminiModel.GEMINI_1_5_FLASH.value: ModelInfo(
        name=GeminiModel.GEMINI_1_5_FLASH.value,
        provider=ModelProvider.GOOGLE,
        context_length=1_056_768,
        max_output_tokens=8192,
        rename_params={"max_tokens": "max_completion_tokens"},
        description="Gemini 1.5 Flash",
    ),
    GeminiModel.GEMINI_1_5_FLASH_8B.value: ModelInfo(
        name=GeminiModel.GEMINI_1_5_FLASH_8B.value,
        provider=ModelProvider.GOOGLE,
        context_length=1_000_000,
        max_output_tokens=8192,
        rename_params={"max_tokens": "max_completion_tokens"},
        description="Gemini 1.5 Flash 8B",
    ),
    GeminiModel.GEMINI_1_5_PRO.value: ModelInfo(
        name=GeminiModel.GEMINI_1_5_PRO.value,
        provider=ModelProvider.GOOGLE,
        context_length=2_000_000,
        max_output_tokens=8192,
        rename_params={"max_tokens": "max_completion_tokens"},
        description="Gemini 1.5 Pro",
    ),
    GeminiModel.GEMINI_2_PRO.value: ModelInfo(
        name=GeminiModel.GEMINI_2_PRO.value,
        provider=ModelProvider.GOOGLE,
        context_length=2_000_000,
        max_output_tokens=8192,
        rename_params={"max_tokens": "max_completion_tokens"},
        description="Gemini 2 Pro Exp 02-05",
    ),
    GeminiModel.GEMINI_2_FLASH_THINKING.value: ModelInfo(
        name=GeminiModel.GEMINI_2_FLASH_THINKING.value,
        provider=ModelProvider.GOOGLE,
        context_length=1_000_000,
        max_output_tokens=64_000,
        rename_params={"max_tokens": "max_completion_tokens"},
        description="Gemini 2.0 Flash Thinking",
    ),
}


def get_model_info(
    model: str | ModelName,
    fallback_model: str | ModelName = "",
) -> ModelInfo:
    """Get model information by name or enum value"""
    return _get_model_info(model) or _get_model_info(fallback_model) or ModelInfo()


def _get_model_info(model: str | ModelName) -> ModelInfo | None:
    if isinstance(model, str):
        return MODEL_INFO.get(model)
    return MODEL_INFO.get(model.value)
