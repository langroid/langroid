"""
Main langroid package
"""

from . import mytypes
from . import utils

from . import parsing
from . import prompts
from . import cachedb

from . import language_models
from . import embedding_models

from . import vector_store
from . import agent

from .agent.base import (
    Agent,
    AgentConfig,
)

from .agent.chat_document import (
    ChatDocument,
    ChatDocMetaData,
)

from .agent.tool_message import (
    ToolMessage,
)

from .agent.chat_agent import (
    ChatAgent,
    ChatAgentConfig,
)

from .agent.task import Task

try:
    from .agent.callbacks.chainlit import (
        ChainlitAgentCallbacks,
        ChainlitTaskCallbacks,
    )

    chainlit_available = True
    ChainlitAgentCallbacks
    ChainlitTaskCallbacks
except ImportError:
    chainlit_available = False


from .mytypes import (
    DocMetaData,
    Document,
    Entity,
)

__all__ = [
    "mytypes",
    "utils",
    "parsing",
    "prompts",
    "cachedb",
    "language_models",
    "embedding_models",
    "vector_store",
    "agent",
    "Agent",
    "AgentConfig",
    "ChatAgent",
    "ChatAgentConfig",
    "ChatDocument",
    "ChatDocMetaData",
    "Task",
    "DocMetaData",
    "Document",
    "Entity",
    "ToolMessage",
]
if chainlit_available:
    __all__.extend(
        [
            "ChainlitAgentCallbacks",
            "ChainlitTaskCallbacks",
        ]
    )
