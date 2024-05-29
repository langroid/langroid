"""
Main langroid package
"""

from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    from .agent.batch import (
        run_batch_tasks,
        llm_response_batch,
        agent_response_batch,
    )

    from .agent.chat_document import (
        StatusCode,
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

    from .agent.task import Task, TaskConfig

    try:
        from .agent.callbacks.chainlit import (
            ChainlitAgentCallbacks,
            ChainlitTaskCallbacks,
            ChainlitCallbackConfig,
        )

        chainlit_available = True
        ChainlitAgentCallbacks
        ChainlitTaskCallbacks
        ChainlitCallbackConfig
    except ImportError:
        chainlit_available = False

    from .mytypes import (
        DocMetaData,
        Document,
        Entity,
    )

    from .exceptions import InfiniteLoopException
    from .exceptions import LangroidImportError

    from . import exceptions

else:
    mytypes = LazyLoad("langroid.mytypes")
    utils = LazyLoad("langroid.utils")
    exceptions = LazyLoad("langroid.exceptions")
    parsing = LazyLoad("langroid.parsing")
    prompts = LazyLoad("langroid.prompts")
    cachedb = LazyLoad("langroid.cachedb")
    language_models = LazyLoad("langroid.language_models")
    embedding_models = LazyLoad("langroid.embedding_models")
    vector_store = LazyLoad("langroid.vector_store")
    agent = LazyLoad("langroid.agent")

    Agent = LazyLoad("langroid.agent.base.Agent")
    AgentConfig = LazyLoad("langroid.agent.base.AgentConfig")

    run_batch_tasks = LazyLoad("langroid.agent.batch.run_batch_tasks")
    llm_response_batch = LazyLoad("langroid.agent.batch.llm_response_batch")
    agent_response_batch = LazyLoad("langroid.agent.batch.agent_response_batch")

    StatusCode = LazyLoad("langroid.agent.chat_document.StatusCode")
    ChatDocument = LazyLoad("langroid.agent.chat_document.ChatDocument")
    ChatDocMetaData = LazyLoad("langroid.agent.chat_document.ChatDocMetaData")

    ToolMessage = LazyLoad("langroid.agent.tool_message.ToolMessage")
    ChatAgent = LazyLoad("langroid.agent.chat_agent.ChatAgent")
    ChatAgentConfig = LazyLoad("langroid.agent.chat_agent.ChatAgentConfig")

    Task = LazyLoad("langroid.agent.task.Task")
    TaskConfig = LazyLoad("langroid.agent.task.TaskConfig")

    ChainlitAgentCallbacks = LazyLoad(
        "langroid.agent.callbacks.chainlit.ChainlitAgentCallbacks"
    )
    ChainlitTaskCallbacks = LazyLoad(
        "langroid.agent.callbacks.chainlit.ChainlitTaskCallbacks"
    )

    ChainlitCallbackConfig = LazyLoad(
        "langroid.agent.callbacks.chainlit.ChainlitCallbackConfig"
    )

    DocMetaData = LazyLoad("langroid.mytypes.DocMetaData")
    Document = LazyLoad("langroid.mytypes.Document")
    Entity = LazyLoad("langroid.mytypes.Entity")

    InfiniteLoopException = LazyLoad("langroid.exceptions.InfiniteLoopException")
    LangroidImportError = LazyLoad("langroid.exceptions.LangroidImportError")


__all__ = [
    "mytypes",
    "exceptions",
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
    "StatusCode",
    "ChatDocument",
    "ChatDocMetaData",
    "Task",
    "TaskConfig",
    "DocMetaData",
    "Document",
    "Entity",
    "ToolMessage",
    "run_batch_tasks",
    "llm_response_batch",
    "agent_response_batch",
    "InfiniteLoopException",
    "LangroidImportError",
    "ChainlitAgentCallbacks",
    "ChainlitTaskCallbacks",
    "ChainlitCallbackConfig",
]
