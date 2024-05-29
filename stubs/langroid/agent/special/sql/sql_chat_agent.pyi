from _typeshed import Incomplete
from sqlalchemy import Row as Row
from sqlalchemy.orm import Session as Session

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
from langroid.agent.special.sql.utils.description_extractors import (
    extract_schema_descriptions as extract_schema_descriptions,
)
from langroid.agent.special.sql.utils.populate_metadata import (
    populate_metadata as populate_metadata,
)
from langroid.agent.special.sql.utils.populate_metadata import (
    populate_metadata_with_schema_tools as populate_metadata_with_schema_tools,
)
from langroid.agent.special.sql.utils.system_message import (
    DEFAULT_SYS_MSG as DEFAULT_SYS_MSG,
)
from langroid.agent.special.sql.utils.system_message import (
    SCHEMA_TOOLS_SYS_MSG as SCHEMA_TOOLS_SYS_MSG,
)
from langroid.agent.special.sql.utils.tools import (
    GetColumnDescriptionsTool as GetColumnDescriptionsTool,
)
from langroid.agent.special.sql.utils.tools import (
    GetTableNamesTool as GetTableNamesTool,
)
from langroid.agent.special.sql.utils.tools import (
    GetTableSchemaTool as GetTableSchemaTool,
)
from langroid.agent.special.sql.utils.tools import (
    RunQueryTool as RunQueryTool,
)
from langroid.exceptions import LangroidImportError as LangroidImportError
from langroid.mytypes import Entity as Entity
from langroid.vector_store.base import VectorStoreConfig as VectorStoreConfig

logger: Incomplete
console: Incomplete
DEFAULT_SQL_CHAT_SYSTEM_MESSAGE: str
SQL_ERROR_MSG: str

class SQLChatAgentConfig(ChatAgentConfig):
    system_message: str
    user_message: None | str
    cache: bool
    debug: bool
    stream: bool
    database_uri: str
    database_session: None | Session
    vecdb: None | VectorStoreConfig
    context_descriptions: dict[str, dict[str, str | dict[str, str]]]
    use_schema_tools: bool
    multi_schema: bool

class SQLChatAgent(ChatAgent):
    config: Incomplete
    def __init__(self, config: SQLChatAgentConfig) -> None: ...
    def agent_response(
        self, msg: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
    def retry_query(self, e: Exception, query: str) -> str: ...
    def run_query(self, msg: RunQueryTool) -> str: ...
    def get_table_names(self, msg: GetTableNamesTool) -> str: ...
    def get_table_schema(self, msg: GetTableSchemaTool) -> str: ...
    def get_column_descriptions(self, msg: GetColumnDescriptionsTool) -> str: ...
