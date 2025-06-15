"""
Test various async handler method signatures for ToolMessage classes.
Tests the new flexible handle_async method that can accept optional agent parameter.
"""

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage


class HandleAsyncNoArgsMsg(ToolMessage):
    """Tool with handle_async() method - no arguments"""

    request: str = "handle_async_no_args"
    purpose: str = "Test async handle with no arguments"

    async def handle_async(self) -> str:
        return "async handled with no args"


class HandleAsyncChatDocMsg(ToolMessage):
    """Tool with handle_async(chat_doc) method"""

    request: str = "handle_async_chat_doc"
    purpose: str = "Test async handle with chat_doc"
    data: str

    async def handle_async(self, chat_doc: ChatDocument) -> str:
        # Actually use the chat_doc parameter
        return (
            f"async handled with chat_doc content '{chat_doc.content}' and "
            f"data: {self.data}"
        )


class HandleAsyncAgentMsg(ToolMessage):
    """Tool with handle_async(agent) method"""

    request: str = "handle_async_agent"
    purpose: str = "Test async handle with agent"

    async def handle_async(self, agent: ChatAgent) -> str:
        return f"async handled with agent: {agent.__class__.__name__}"


class HandleAsyncAgentChatDocMsg(ToolMessage):
    """Tool with handle_async(agent, chat_doc) method"""

    request: str = "handle_async_agent_chat_doc"
    purpose: str = "Test async handle with agent and chat_doc"
    data: str

    async def handle_async(self, agent: ChatAgent, chat_doc: ChatDocument) -> str:
        # Use both agent and chat_doc
        return (
            f"async handled with agent {agent.__class__.__name__}, "
            f"chat_doc '{chat_doc.content}', data: {self.data}"
        )


class HandleAsyncChatDocAgentMsg(ToolMessage):
    """Tool with handle_async(chat_doc, agent) method - reversed order"""

    request: str = "handle_async_chat_doc_agent"
    purpose: str = "Test async handle with chat_doc and agent in reverse order"
    data: str

    async def handle_async(self, chat_doc: ChatDocument, agent: ChatAgent) -> str:
        # Use both parameters in the order they're defined
        return (
            f"async chat_doc '{chat_doc.content}' first, "
            f"then agent {agent.__class__.__name__}, data: {self.data}"
        )


class HandleAsyncNoAnnotationsMsg(ToolMessage):
    """Tool with async handle method but no type annotations - single param"""

    request: str = "handle_async_no_annotations"
    purpose: str = "Test async handle without type annotations"
    data: str

    async def handle_async(self, chat_doc) -> str:
        # Should fall back to parameter name - assume single arg is chat_doc
        # Access content attribute to verify it's actually a ChatDocument
        return (
            f"async handled without annotations - "
            f"chat_doc: '{chat_doc.content}', data: {self.data}"
        )


class HandleAsyncNoAnnotationsAgentMsg(ToolMessage):
    """Tool with async handle method expecting agent but no type annotations"""

    request: str = "handle_async_no_annotations_agent"
    purpose: str = "Test async handle with agent but no annotations"

    async def handle_async(self, agent) -> str:
        # Parameter name 'agent' should be recognized even without annotations
        return f"async handled agent without annotations: {agent.__class__.__name__}"


class HandleAsyncNoAnnotationsBothMsg(ToolMessage):
    """Tool with async handle method expecting both params but no type annotations"""

    request: str = "handle_async_no_annotations_both"
    purpose: str = "Test async handle with both params but no annotations"
    data: str

    async def handle_async(self, agent, chat_doc) -> str:
        # Parameter names 'agent' and 'chat_doc' should be recognized
        return (
            f"async handled with agent {agent.__class__.__name__}, "
            f"chat_doc '{chat_doc.content}', data: {self.data}"
        )


class HandleAsyncNoAnnotationsBothReversedMsg(ToolMessage):
    """Tool with async handle method with reversed parameter order but no type annotations"""  # noqa: E501

    request: str = "handle_async_no_annotations_both_reversed"
    purpose: str = "Test async handle with reversed params but no annotations"
    data: str

    async def handle_async(self, chat_doc, agent) -> str:
        # Parameter order should be respected based on names
        return (
            f"async chat_doc '{chat_doc.content}' first, "
            f"agent {agent.__class__.__name__}, data: {self.data}"
        )


class TestToolHandlerAsync:
    """Test the flexible async tool handler extraction"""

    @pytest.mark.asyncio
    async def test_handle_async_no_args(self):
        """Test handle_async() with no arguments"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAsyncNoArgsMsg)

        msg = HandleAsyncNoArgsMsg()
        result = await agent.handle_async_no_args_async(msg)
        assert result == "async handled with no args"

    @pytest.mark.asyncio
    async def test_handle_async_chat_doc(self):
        """Test handle_async(chat_doc) with type annotation"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAsyncChatDocMsg)

        msg = HandleAsyncChatDocMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = await agent.handle_async_chat_doc_async(msg, chat_doc)
        assert result == (
            "async handled with chat_doc content 'test' and " "data: test data"
        )

    @pytest.mark.asyncio
    async def test_handle_async_agent(self):
        """Test handle_async(agent) with type annotation"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAsyncAgentMsg)

        msg = HandleAsyncAgentMsg()
        result = await agent.handle_async_agent_async(msg)
        assert result == "async handled with agent: ChatAgent"

    @pytest.mark.asyncio
    async def test_handle_async_agent_chat_doc(self):
        """Test handle_async(agent, chat_doc) with type annotations"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAsyncAgentChatDocMsg)

        msg = HandleAsyncAgentChatDocMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = await agent.handle_async_agent_chat_doc_async(msg, chat_doc)
        assert result == (
            "async handled with agent ChatAgent, " "chat_doc 'test', data: test data"
        )

    @pytest.mark.asyncio
    async def test_handle_async_chat_doc_agent(self):
        """Test handle_async(chat_doc, agent) with reversed parameter order"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAsyncChatDocAgentMsg)

        msg = HandleAsyncChatDocAgentMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = await agent.handle_async_chat_doc_agent_async(msg, chat_doc)
        assert result == (
            "async chat_doc 'test' first, " "then agent ChatAgent, data: test data"
        )

    @pytest.mark.asyncio
    async def test_handle_async_no_annotations(self):
        """Test async handle with no type annotations - should use parameter name"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAsyncNoAnnotationsMsg)

        msg = HandleAsyncNoAnnotationsMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = await agent.handle_async_no_annotations_async(msg, chat_doc)
        assert result == (
            "async handled without annotations - " "chat_doc: 'test', data: test data"
        )

    @pytest.mark.asyncio
    async def test_handle_async_no_annotations_agent(self):
        """Test async handle with agent param but no type annotations"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAsyncNoAnnotationsAgentMsg)

        msg = HandleAsyncNoAnnotationsAgentMsg()
        # Parameter name 'agent' should be recognized, so no chat_doc needed
        result = await agent.handle_async_no_annotations_agent_async(msg)
        assert result == "async handled agent without annotations: ChatAgent"

    @pytest.mark.asyncio
    async def test_handle_async_no_annotations_both(self):
        """Test async handle with both params but no type annotations"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAsyncNoAnnotationsBothMsg)

        msg = HandleAsyncNoAnnotationsBothMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = await agent.handle_async_no_annotations_both_async(msg, chat_doc)
        assert result == (
            "async handled with agent ChatAgent, " "chat_doc 'test', data: test data"
        )

    @pytest.mark.asyncio
    async def test_handle_async_no_annotations_both_reversed(self):
        """Test async handle with reversed params but no type annotations"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAsyncNoAnnotationsBothReversedMsg)

        msg = HandleAsyncNoAnnotationsBothReversedMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = await agent.handle_async_no_annotations_both_reversed_async(
            msg, chat_doc
        )
        assert result == (
            "async chat_doc 'test' first, " "agent ChatAgent, data: test data"
        )


# Test that sync and async can coexist
class HandleBothSyncAsyncMsg(ToolMessage):
    """Tool with both sync and async handle methods"""

    request: str = "handle_both_sync_async"
    purpose: str = "Test tool with both sync and async handlers"

    def handle(self, agent: ChatAgent) -> str:
        return f"sync handled with agent: {agent.__class__.__name__}"

    async def handle_async(self, agent: ChatAgent) -> str:
        return f"async handled with agent: {agent.__class__.__name__}"


@pytest.mark.asyncio
async def test_handle_both_sync_async():
    """Test that a tool can have both sync and async handlers"""
    agent = ChatAgent(ChatAgentConfig())
    agent.enable_message(HandleBothSyncAsyncMsg)

    msg = HandleBothSyncAsyncMsg()

    # Test sync handler
    sync_result = agent.handle_both_sync_async(msg)
    assert sync_result == "sync handled with agent: ChatAgent"

    # Test async handler
    async_result = await agent.handle_both_sync_async_async(msg)
    assert async_result == "async handled with agent: ChatAgent"
