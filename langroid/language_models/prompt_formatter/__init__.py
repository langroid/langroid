from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import base
    from . import llama2_formatter
    from .base import PromptFormatter
    from .llama2_formatter import Llama2Formatter
    from ..config import PromptFormatterConfig
    from ..config import Llama2FormatterConfig
else:
    base = LazyLoad("langroid.language_models.prompt_formatter.base")
    llama2_formatter = LazyLoad(
        "langroid.language_models.prompt_formatter.llama2_formatter"
    )
    PromptFormatter = LazyLoad(
        "langroid.language_models.prompt_formatter.base.PromptFormatter"
    )
    Llama2Formatter = LazyLoad(
        "langroid.language_models.prompt_formatter.llama2_formatter.Llama2Formatter"
    )

    PromptFormatterConfig = LazyLoad(
        "langroid.language_models.config.PromptFormatterConfig"
    )
    Llama2FormatterConfig = LazyLoad(
        "langroid.language_models.config.Llama2FormatterConfig"
    )


__all__ = [
    "PromptFormatter",
    "Llama2Formatter",
    "PromptFormatterConfig",
    "Llama2FormatterConfig",
    "base",
    "llama2_formatter",
]
