from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import sql_chat_agent, utils
    from .sql_chat_agent import SQLChatAgentConfig, SQLChatAgent
else:
    sql_chat_agent = LazyLoad("langroid.agent.special.sql.sql_chat_agent")
    SQLChatAgentConfig = LazyLoad(
        "langroid.agent.special.sql.sql_chat_agent.SQLChatAgentConfig"
    )
    SQLChatAgent = LazyLoad("langroid.agent.special.sql.sql_chat_agent.SQLChatAgent")

    utils = LazyLoad("langroid.agent.special.sql.utils")


__all__ = [
    "SQLChatAgentConfig",
    "SQLChatAgent",
    "sql_chat_agent",
    "utils",
]
