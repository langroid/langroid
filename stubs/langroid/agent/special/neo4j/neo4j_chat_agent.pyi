from typing import Any

from _typeshed import Incomplete
from pydantic import BaseModel, BaseSettings

from langroid.agent import ToolMessage as ToolMessage
from langroid.agent.chat_agent import (
    ChatAgent as ChatAgent,
)
from langroid.agent.chat_agent import (
    ChatAgentConfig as ChatAgentConfig,
)
from langroid.agent.chat_document import (
    ChatDocMetaData as ChatDocMetaData,
)
from langroid.agent.chat_document import (
    ChatDocument as ChatDocument,
)
from langroid.agent.special.neo4j.utils.system_message import (
    DEFAULT_NEO4J_CHAT_SYSTEM_MESSAGE as DEFAULT_NEO4J_CHAT_SYSTEM_MESSAGE,
)
from langroid.agent.special.neo4j.utils.system_message import (
    DEFAULT_SYS_MSG as DEFAULT_SYS_MSG,
)
from langroid.agent.special.neo4j.utils.system_message import (
    SCHEMA_TOOLS_SYS_MSG as SCHEMA_TOOLS_SYS_MSG,
)
from langroid.mytypes import Entity as Entity

logger: Incomplete
console: Incomplete
NEO4J_ERROR_MSG: str

class CypherRetrievalTool(ToolMessage):
    request: str
    purpose: str
    cypher_query: str

class CypherCreationTool(ToolMessage):
    request: str
    purpose: str
    cypher_query: str

class GraphSchemaTool(ToolMessage):
    request: str
    purpose: str

class Neo4jSettings(BaseSettings):
    uri: str
    username: str
    password: str
    database: str

    class Config:
        env_prefix: str

class QueryResult(BaseModel):
    success: bool
    data: str | list[dict[Any, Any]] | None

class Neo4jChatAgentConfig(ChatAgentConfig):
    neo4j_settings: Neo4jSettings
    system_message: str
    kg_schema: list[dict[str, Any]] | None
    database_created: bool
    use_schema_tools: bool
    use_functions_api: bool
    use_tools: bool

class Neo4jChatAgent(ChatAgent):
    config: Incomplete
    def __init__(self, config: Neo4jChatAgentConfig) -> None: ...
    def close(self) -> None: ...
    def retry_query(self, e: Exception, query: str) -> str: ...
    def read_query(
        self, query: str, parameters: dict[Any, Any] | None = None
    ) -> QueryResult: ...
    def write_query(
        self, query: str, parameters: dict[Any, Any] | None = None
    ) -> QueryResult: ...
    def remove_database(self) -> None: ...
    def retrieval_query(self, msg: CypherRetrievalTool) -> str: ...
    def create_query(self, msg: CypherCreationTool) -> str: ...
    def get_schema(self, msg: GraphSchemaTool | None) -> str: ...
    def agent_response(
        self, msg: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
