"""
Test various handler method signatures for ToolMessage classes.
Tests the new flexible handle method that can accept optional agent parameter.
"""

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage


class HandleNoArgsMsg(ToolMessage):
    """Tool with handle() method - no arguments"""

    request: str = "handle_no_args"
    purpose: str = "Test handle with no arguments"

    def handle(self) -> str:
        return "handled with no args"


class HandleChatDocMsg(ToolMessage):
    """Tool with handle(chat_doc) method"""

    request: str = "handle_chat_doc"
    purpose: str = "Test handle with chat_doc"
    data: str

    def handle(self, chat_doc: ChatDocument) -> str:
        # Actually use the chat_doc parameter
        return (
            f"handled with chat_doc content '{chat_doc.content}' and "
            f"data: {self.data}"
        )


class HandleAgentMsg(ToolMessage):
    """Tool with handle(agent) method"""

    request: str = "handle_agent"
    purpose: str = "Test handle with agent"

    def handle(self, agent: ChatAgent) -> str:
        return f"handled with agent: {agent.__class__.__name__}"


class HandleAgentChatDocMsg(ToolMessage):
    """Tool with handle(agent, chat_doc) method"""

    request: str = "handle_agent_chat_doc"
    purpose: str = "Test handle with agent and chat_doc"
    data: str

    def handle(self, agent: ChatAgent, chat_doc: ChatDocument) -> str:
        # Use both agent and chat_doc
        return (
            f"handled with agent {agent.__class__.__name__}, "
            f"chat_doc '{chat_doc.content}', data: {self.data}"
        )


class HandleChatDocAgentMsg(ToolMessage):
    """Tool with handle(chat_doc, agent) method - reversed order"""

    request: str = "handle_chat_doc_agent"
    purpose: str = "Test handle with chat_doc and agent in reverse order"
    data: str

    def handle(self, chat_doc: ChatDocument, agent: ChatAgent) -> str:
        # Use both parameters in the order they're defined
        return (
            f"chat_doc '{chat_doc.content}' first, "
            f"then agent {agent.__class__.__name__}, data: {self.data}"
        )


class HandleNoAnnotationsMsg(ToolMessage):
    """Tool with handle method but no type annotations - single param"""

    request: str = "handle_no_annotations"
    purpose: str = "Test handle without type annotations"
    data: str

    def handle(self, chat_doc) -> str:
        # Should fall back to duck typing - assume single arg is chat_doc
        # Access content attribute to verify it's actually a ChatDocument
        return (
            f"handled without annotations - "
            f"chat_doc: '{chat_doc.content}', data: {self.data}"
        )


class HandleNoAnnotationsAgentMsg(ToolMessage):
    """Tool with handle method expecting agent but no type annotations"""

    request: str = "handle_no_annotations_agent"
    purpose: str = "Test handle with agent but no annotations"

    def handle(self, agent) -> str:
        # Parameter name 'agent' should be recognized even without annotations
        return f"handled agent without annotations: {agent.__class__.__name__}"


class HandleNoAnnotationsBothMsg(ToolMessage):
    """Tool with handle method expecting both params but no type annotations"""

    request: str = "handle_no_annotations_both"
    purpose: str = "Test handle with both params but no annotations"
    data: str

    def handle(self, agent, chat_doc) -> str:
        # Parameter names 'agent' and 'chat_doc' should be recognized
        return (
            f"handled with agent {agent.__class__.__name__}, "
            f"chat_doc '{chat_doc.content}', data: {self.data}"
        )


class HandleNoAnnotationsBothReversedMsg(ToolMessage):
    """Tool with handle method with reversed parameter order but no type annotations"""

    request: str = "handle_no_annotations_both_reversed"
    purpose: str = "Test handle with reversed params but no annotations"
    data: str

    def handle(self, chat_doc, agent) -> str:
        # Parameter order should be respected based on names
        return (
            f"chat_doc '{chat_doc.content}' first, "
            f"agent {agent.__class__.__name__}, data: {self.data}"
        )


class TestToolHandler:
    """Test the flexible tool handler extraction"""

    def test_handle_no_args(self):
        """Test handle() with no arguments"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleNoArgsMsg)

        msg = HandleNoArgsMsg()
        result = agent.handle_no_args(msg)
        assert result == "handled with no args"

    def test_handle_chat_doc(self):
        """Test handle(chat_doc) with type annotation"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleChatDocMsg)

        msg = HandleChatDocMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = agent.handle_chat_doc(msg, chat_doc)
        assert result == "handled with chat_doc content 'test' and data: test data"

    def test_handle_agent(self):
        """Test handle(agent) with type annotation"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAgentMsg)

        msg = HandleAgentMsg()
        result = agent.handle_agent(msg)
        assert result == "handled with agent: ChatAgent"

    def test_handle_agent_chat_doc(self):
        """Test handle(agent, chat_doc) with type annotations"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAgentChatDocMsg)

        msg = HandleAgentChatDocMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = agent.handle_agent_chat_doc(msg, chat_doc)
        assert (
            result == "handled with agent ChatAgent, chat_doc 'test', data: test data"
        )

    def test_handle_chat_doc_agent(self):
        """Test handle(chat_doc, agent) with reversed parameter order"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleChatDocAgentMsg)

        msg = HandleChatDocAgentMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = agent.handle_chat_doc_agent(msg, chat_doc)
        assert result == "chat_doc 'test' first, then agent ChatAgent, data: test data"

    def test_handle_no_annotations(self):
        """Test handle with no type annotations - should use duck typing"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleNoAnnotationsMsg)

        msg = HandleNoAnnotationsMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = agent.handle_no_annotations(msg, chat_doc)
        assert result == (
            "handled without annotations - " "chat_doc: 'test', data: test data"
        )

    def test_handle_no_annotations_agent(self):
        """Test handle with agent param but no type annotations"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleNoAnnotationsAgentMsg)

        msg = HandleNoAnnotationsAgentMsg()
        # When called from agent, it won't pass chat_doc if the handler
        # only expects one parameter
        result = agent.handle_no_annotations_agent(msg)
        assert result == "handled agent without annotations: ChatAgent"

    def test_handle_no_annotations_both(self):
        """Test handle with both params but no type annotations"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleNoAnnotationsBothMsg)

        msg = HandleNoAnnotationsBothMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = agent.handle_no_annotations_both(msg, chat_doc)
        assert result == (
            "handled with agent ChatAgent, " "chat_doc 'test', data: test data"
        )

    def test_handle_no_annotations_both_reversed(self):
        """Test handle with reversed params but no type annotations"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleNoAnnotationsBothReversedMsg)

        msg = HandleNoAnnotationsBothReversedMsg(data="test data")
        chat_doc = agent.create_agent_response(content="test")
        result = agent.handle_no_annotations_both_reversed(msg, chat_doc)
        assert result == "chat_doc 'test' first, agent ChatAgent, data: test data"

    def test_backward_compatibility(self):
        """Test that existing tools with response() method still work"""
        from langroid.agent.tools.orchestration import DoneTool

        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(DoneTool)

        # DoneTool uses response(agent) method
        msg = DoneTool(content="task complete")
        result = agent.done_tool(msg)
        assert result is not None

    def test_agent_response_with_tool_message(self):
        """Test agent_response method with various tool messages"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAgentChatDocMsg)

        # Create a tool message
        tool_msg = HandleAgentChatDocMsg(data="response test")

        # When using a handler that expects both agent and chat_doc,
        # we need to provide the tool message within a ChatDocument
        # so that chat_doc is available
        chat_doc = agent.create_agent_response(content=tool_msg.json())
        response = agent.agent_response(chat_doc)
        assert response is not None
        assert isinstance(response, ChatDocument)
        assert "response test" in response.content

    def test_agent_response_with_chat_document(self):
        """Test agent_response with ChatDocument containing tool message"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleChatDocAgentMsg)

        # Create a tool message
        tool_msg = HandleChatDocAgentMsg(data="chat doc test")

        # Create a ChatDocument with the tool message
        chat_doc = agent.create_agent_response(content=tool_msg.json())

        # Process through agent_response
        response = agent.agent_response(chat_doc)
        assert response is not None
        assert isinstance(response, ChatDocument)
        # Check that our handle method was called with both agent and chat_doc
        assert "chat_doc" in response.content
        assert "ChatAgent" in response.content

    def test_agent_response_no_annotations(self):
        """Test agent_response with handler that has no type annotations"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleNoAnnotationsBothMsg)

        # Create a tool message
        tool_msg = HandleNoAnnotationsBothMsg(data="no annotations test")

        # Since the handler expects both agent and chat_doc,
        # we need to provide it within a ChatDocument
        chat_doc = agent.create_agent_response(content=tool_msg.json())
        response = agent.agent_response(chat_doc)
        assert response is not None
        assert isinstance(response, ChatDocument)
        assert "no annotations test" in response.content
        assert "ChatAgent" in response.content

    def test_agent_response_agent_only(self):
        """Test agent_response with handler that only takes agent parameter"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleAgentMsg)

        # Create a tool message
        tool_msg = HandleAgentMsg()

        # Test with tool message as JSON string
        json_msg = tool_msg.json()
        response = agent.agent_response(json_msg)
        assert response is not None
        assert isinstance(response, ChatDocument)
        assert "handled with agent: ChatAgent" in response.content

    def test_agent_response_chat_doc_only(self):
        """Test agent_response with handler that only takes chat_doc parameter"""
        agent = ChatAgent(ChatAgentConfig())
        agent.enable_message(HandleChatDocMsg)

        # Create a tool message
        tool_msg = HandleChatDocMsg(data="chat doc only test")

        # For handlers that only need chat_doc, we can pass as ChatDocument
        chat_doc = agent.create_agent_response(content=tool_msg.json())
        response = agent.agent_response(chat_doc)
        assert response is not None
        assert isinstance(response, ChatDocument)
        assert "chat doc only test" in response.content
