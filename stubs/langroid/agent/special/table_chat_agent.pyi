import pandas as pd
from _typeshed import Incomplete

from langroid.agent import ChatDocument as ChatDocument
from langroid.agent.chat_agent import (
    ChatAgent as ChatAgent,
)
from langroid.agent.chat_agent import (
    ChatAgentConfig as ChatAgentConfig,
)
from langroid.agent.tool_message import ToolMessage as ToolMessage
from langroid.language_models.openai_gpt import (
    OpenAIChatModel as OpenAIChatModel,
)
from langroid.language_models.openai_gpt import (
    OpenAIGPTConfig as OpenAIGPTConfig,
)
from langroid.parsing.table_loader import read_tabular_data as read_tabular_data
from langroid.prompts.prompts_config import PromptsConfig as PromptsConfig
from langroid.utils.constants import DONE as DONE
from langroid.utils.constants import PASS as PASS
from langroid.vector_store.base import VectorStoreConfig as VectorStoreConfig

logger: Incomplete
console: Incomplete
DEFAULT_TABLE_CHAT_SYSTEM_MESSAGE: Incomplete

def dataframe_summary(df): ...

class TableChatAgentConfig(ChatAgentConfig):
    system_message: str
    user_message: None | str
    cache: bool
    debug: bool
    stream: bool
    data: str | pd.DataFrame
    separator: None | str
    vecdb: None | VectorStoreConfig
    llm: OpenAIGPTConfig
    prompts: PromptsConfig

class PandasEvalTool(ToolMessage):
    request: str
    purpose: str
    expression: str
    @classmethod
    def examples(cls) -> list["ToolMessage"]: ...
    @classmethod
    def instructions(cls) -> str: ...

class TableChatAgent(ChatAgent):
    sent_expression: bool
    df: Incomplete
    config: Incomplete
    def __init__(self, config: TableChatAgentConfig) -> None: ...
    def user_response(
        self, msg: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
    def pandas_eval(self, msg: PandasEvalTool) -> str: ...
    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None: ...
