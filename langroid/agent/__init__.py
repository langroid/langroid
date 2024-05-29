from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Agent, AgentConfig
    from .chat_document import (
        ChatDocAttachment,
        ChatDocMetaData,
        ChatDocLoggerFields,
        ChatDocument,
    )
    from .chat_agent import ChatAgentConfig, ChatAgent
    from .tool_message import ToolMessage
    from .task import Task

    from . import base
    from . import chat_document
    from . import chat_agent
    from . import task
    from . import batch
    from . import tool_message
    from . import tools
    from . import special
else:
    base = LazyLoad("langroid.agent.base")
    Agent = LazyLoad("langroid.agent.base.Agent")
    AgentConfig = LazyLoad("langroid.agent.base.AgentConfig")

    chat_document = LazyLoad("langroid.agent.chat_document")
    ChatDocAttachment = LazyLoad("langroid.agent.chat_document.ChatDocAttachment")
    ChatDocMetaData = LazyLoad("langroid.agent.chat_document.ChatDocMetaData")
    ChatDocLoggerFields = LazyLoad("langroid.agent.chat_document.ChatDocLoggerFields")
    ChatDocument = LazyLoad("langroid.agent.chat_document.ChatDocument")

    chat_agent = LazyLoad("langroid.agent.chat_agent")
    ChatAgent = LazyLoad("langroid.agent.chat_agent.ChatAgent")
    ChatAgentConfig = LazyLoad("langroid.agent.chat_agent.ChatAgentConfig")

    tool_message = LazyLoad("langroid.agent.tool_message")
    ToolMessage = LazyLoad("langroid.agent.tool_message.ToolMessage")

    task = LazyLoad("langroid.agent.task")
    Task = LazyLoad("langroid.agent.task.Task")

    batch = LazyLoad("langroid.agent.batch")
    tools = LazyLoad("langroid.agent.tools")
    special = LazyLoad("langroid.agent.special")


__all__ = [
    "Agent",
    "AgentConfig",
    "ChatDocAttachment",
    "ChatDocMetaData",
    "ChatDocLoggerFields",
    "ChatDocument",
    "ChatAgent",
    "ChatAgentConfig",
    "ToolMessage",
    "Task",
    "base",
    "chat_document",
    "chat_agent",
    "task",
    "batch",
    "tool_message",
    "tools",
    "special",
]
