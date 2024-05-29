from _typeshed import Incomplete

from langroid.language_models.base import (
    LanguageModel as LanguageModel,
)
from langroid.language_models.base import (
    LLMMessage as LLMMessage,
)
from langroid.language_models.config import (
    Llama2FormatterConfig as Llama2FormatterConfig,
)
from langroid.language_models.prompt_formatter.base import (
    PromptFormatter as PromptFormatter,
)

logger: Incomplete
BOS: str
EOS: str
B_INST: str
E_INST: str
B_SYS: str
E_SYS: str
SPECIAL_TAGS: list[str]

class Llama2Formatter(PromptFormatter):
    config: Incomplete
    def __int__(self, config: Llama2FormatterConfig) -> None: ...
    def format(self, messages: list[LLMMessage]) -> str: ...
