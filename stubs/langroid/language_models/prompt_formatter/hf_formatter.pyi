from typing import Any

from _typeshed import Incomplete

from langroid.language_models.base import (
    LanguageModel as LanguageModel,
)
from langroid.language_models.base import (
    LLMMessage as LLMMessage,
)
from langroid.language_models.base import (
    Role as Role,
)
from langroid.language_models.config import (
    HFPromptFormatterConfig as HFPromptFormatterConfig,
)
from langroid.language_models.prompt_formatter.base import (
    PromptFormatter as PromptFormatter,
)

logger: Incomplete

def try_import_hf_modules() -> tuple[type[Any], type[Any], type[Any]]: ...
def find_hf_formatter(model_name: str) -> str: ...

class HFFormatter(PromptFormatter):
    models: set[str]
    config: Incomplete
    tokenizer: Incomplete
    def __init__(self, config: HFPromptFormatterConfig) -> None: ...
    def format(self, messages: list[LLMMessage]) -> str: ...
