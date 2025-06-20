"""Tests for done sequences DSL integration with Task."""

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task, TaskConfig
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.mock_lm import MockLMConfig
from langroid.utils.configuration import Settings, set_global


class SimpleTool(ToolMessage):
    request: str = "simple_tool"
    purpose: str = "A simple tool for testing"
    value: str

    def handle(self) -> str:
        """Handle the tool and return a response"""
        return f"Processed value: {self.value}"


def test_dsl_simple_pattern(test_settings: Settings):
    """Test that DSL pattern 'T, A' works like full DoneSequence."""
    set_global(test_settings)

    # Mock LLM that always generates a tool
    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(
                response_fn=lambda x: '{"request": "simple_tool", "value": "test"}'
            ),
        )
    )
    agent.enable_message(SimpleTool)

    # Use DSL string pattern
    config = TaskConfig(done_sequences=["T, A"])
    task = Task(agent, config=config, interactive=False)
    result = task.run("Generate a tool", turns=10)

    assert result is not None
    assert len(agent.message_history) == 3


def test_dsl_specific_tool(test_settings: Settings):
    """Test DSL pattern with specific tool name."""
    set_global(test_settings)

    class AnotherTool(ToolMessage):
        request: str = "another_tool"
        purpose: str = "Another tool"
        data: str

        def handle(self) -> str:
            return f"Processed data: {self.data}"

    # Mock LLM that generates specific tool
    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(
                response_fn=lambda x: '{"request": "simple_tool", "value": "test"}'
            ),
        )
    )
    agent.enable_message(SimpleTool)
    agent.enable_message(AnotherTool)

    # Use DSL with specific tool
    config = TaskConfig(done_sequences=["T[simple_tool], A"])
    task = Task(agent, config=config, interactive=False)
    result = task.run("Generate tool", turns=10)

    assert result is not None
    assert "simple_tool" in agent.message_history[-1].content


def test_dsl_content_match(test_settings: Settings):
    """Test DSL pattern with content matching."""
    set_global(test_settings)

    # Mock LLM that says "quit"
    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(response_fn=lambda x: "I quit now"),
        )
    )

    # Use DSL with content match
    config = TaskConfig(done_sequences=["C[quit|exit]"])
    task = Task(agent, config=config, interactive=False)
    result = task.run("Do something", turns=10)

    assert result is not None
    assert "quit" in result.content.lower()


def test_dsl_complex_pattern(test_settings: Settings):
    """Test complex DSL pattern."""
    set_global(test_settings)

    responses = [
        "Let me help",
        '{"request": "simple_tool", "value": "calc"}',
        "All done",
    ]
    response_idx = 0

    def mock_response(x):
        nonlocal response_idx
        resp = responses[response_idx % len(responses)]
        response_idx += 1
        return resp

    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(response_fn=mock_response),
        )
    )
    agent.enable_message(SimpleTool)

    # Complex pattern: LLM, Tool, Agent, LLM
    config = TaskConfig(done_sequences=["L, T, A, L"])
    task = Task(
        agent,
        config=config,
        interactive=False,
        single_round=False,
        allow_null_result=True,
    )
    result = task.run("Help me", turns=10)

    assert result is not None


def test_dsl_mixed_with_done_sequence(test_settings: Settings):
    """Test mixing DSL strings with DoneSequence objects."""
    set_global(test_settings)

    from langroid.agent.task import AgentEvent, DoneSequence, EventType

    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(
                response_fn=lambda x: '{"request": "simple_tool", "value": "test"}'
            ),
        )
    )
    agent.enable_message(SimpleTool)

    # Mix DSL string and DoneSequence object
    config = TaskConfig(
        done_sequences=[
            "T, A",  # DSL string
            DoneSequence(  # Full object
                name="specific_pattern",
                events=[
                    AgentEvent(event_type=EventType.LLM_RESPONSE),
                    AgentEvent(event_type=EventType.LLM_RESPONSE),
                ],
            ),
        ]
    )
    task = Task(agent, config=config, interactive=False)
    result = task.run("Do something", turns=10)

    assert result is not None


def test_dsl_without_spaces(test_settings: Settings):
    """Test DSL works without spaces."""
    set_global(test_settings)

    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(
                response_fn=lambda x: '{"request": "simple_tool", "value": "test"}'
            ),
        )
    )
    agent.enable_message(SimpleTool)

    # DSL without spaces
    config = TaskConfig(done_sequences=["T,A"])
    task = Task(agent, config=config, interactive=False)
    result = task.run("Generate tool", turns=10)

    assert result is not None
    assert len(agent.message_history) == 3


def test_dsl_word_tokens(test_settings: Settings):
    """Test DSL with full word tokens."""
    set_global(test_settings)

    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(
                response_fn=lambda x: '{"request": "simple_tool", "value": "test"}'
            ),
        )
    )
    agent.enable_message(SimpleTool)

    # Full word tokens
    config = TaskConfig(done_sequences=["TOOL, AGENT"])
    task = Task(agent, config=config, interactive=False)
    result = task.run("Generate tool", turns=10)

    assert result is not None
    assert len(agent.message_history) == 3
