import pandas as pd
from _typeshed import Incomplete

from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent as Neo4jChatAgent,
)
from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgentConfig as Neo4jChatAgentConfig,
)
from langroid.agent.tool_message import ToolMessage as ToolMessage
from langroid.language_models.openai_gpt import (
    OpenAIChatModel as OpenAIChatModel,
)
from langroid.language_models.openai_gpt import (
    OpenAIGPTConfig as OpenAIGPTConfig,
)
from langroid.parsing.table_loader import read_tabular_data as read_tabular_data
from langroid.utils.output import status as status
from langroid.vector_store.base import VectorStoreConfig as VectorStoreConfig

app: Incomplete
BUILD_KG_INSTRUCTIONS: str
DEFAULT_CSV_KG_CHAT_SYSTEM_MESSAGE: str

class CSVGraphAgentConfig(Neo4jChatAgentConfig):
    system_message: str
    data: str | pd.DataFrame | None
    separator: None | str
    vecdb: None | VectorStoreConfig
    llm: OpenAIGPTConfig

class PandasToKGTool(ToolMessage):
    request: str
    purpose: str
    cypherQuery: str
    args: list[str]
    @classmethod
    def examples(cls) -> list["ToolMessage"]: ...

class CSVGraphAgent(Neo4jChatAgent):
    df: Incomplete
    config: Incomplete
    def __init__(self, config: CSVGraphAgentConfig) -> None: ...
    def pandas_to_kg(self, msg: PandasToKGTool) -> str: ...
