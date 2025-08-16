"""
Tests for the done_sequences feature in Task.
"""

from pydantic import Field

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import (
    AgentEvent,
    DoneSequence,
    EventType,
    Task,
    TaskConfig,
)
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


class CalculatorTool(ToolMessage):
    request: str = "calculator"
    purpose: str = "Calculate math expressions"
    expression: str = Field(..., description="Math expression")

    def handle(self) -> str:
        return f"Result: {eval(self.expression)}"


def test_done_sequence_tool_then_agent(test_settings: Settings):
    """Test that task terminates after tool followed by agent response"""
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

    # Configure task to be done after tool -> agent response
    config = TaskConfig(
        done_sequences=[
            DoneSequence(
                name="tool_then_agent",
                events=[
                    AgentEvent(event_type=EventType.TOOL),
                    AgentEvent(event_type=EventType.AGENT_RESPONSE),
                ],
            )
        ]
    )

    task = Task(agent, config=config, interactive=False)
    result = task.run("Generate a tool", turns=10)

    # Task should complete after tool generation and agent response
    assert result is not None
    # Should have: system, user, llm (with tool)
    # Note: agent response is not added to message history
    assert len(agent.message_history) == 3


def test_done_sequence_specific_tool(test_settings: Settings):
    """Test that task terminates only after specific tool"""
    set_global(test_settings)

    class AnotherTool(ToolMessage):
        request: str = "another_tool"
        purpose: str = "Another tool"
        data: str

        def handle(self) -> str:
            return f"Processed data: {self.data}"

    # Mock LLM that alternates between tools
    call_count = 0

    def mock_response(x):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '{"request": "another_tool", "data": "test"}'
        else:
            return '{"request": "simple_tool", "value": "test"}'

    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(response_fn=mock_response),
        )
    )
    agent.enable_message(SimpleTool)
    agent.enable_message(AnotherTool)

    # Configure to be done only after simple_tool
    config = TaskConfig(
        done_sequences=[
            DoneSequence(
                name="specific_tool",
                events=[
                    AgentEvent(
                        event_type=EventType.SPECIFIC_TOOL, tool_name="simple_tool"
                    ),
                    AgentEvent(event_type=EventType.AGENT_RESPONSE),
                ],
            )
        ]
    )

    task = Task(agent, config=config, interactive=False)
    result = task.run("Generate tools", turns=10)

    assert result is not None
    # Verify simple_tool was generated (it's in the last assistant message)
    last_assistant_msg = agent.message_history[-1]
    assert "simple_tool" in last_assistant_msg.content


def test_done_sequence_llm_agent_llm(test_settings: Settings):
    """Test sequence: LLM -> Agent -> LLM"""
    set_global(test_settings)

    # Mock LLM: first plain text, then tool, then plain text again
    responses = [
        "Let me process this",
        '{"request": "simple_tool", "value": "processed"}',
        "Processing complete",
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

    # Done after: Tool -> Agent -> LLM (plain)
    # The first LLM response becomes a USER message in the conversation
    config = TaskConfig(
        done_sequences=[
            DoneSequence(
                name="process_complete",
                events=[
                    AgentEvent(event_type=EventType.TOOL),
                    AgentEvent(event_type=EventType.AGENT_RESPONSE),
                    AgentEvent(event_type=EventType.LLM_RESPONSE),
                ],
            )
        ]
    )

    task = Task(
        agent,
        config=config,
        interactive=False,
        single_round=False,
        allow_null_result=True,  # Allow conversation to continue
    )
    result = task.run("Process data", turns=10)

    assert result is not None
    assert result.content == "Processing complete"


def test_done_sequence_no_match(test_settings: Settings):
    """Test that task continues when sequence doesn't match"""
    set_global(test_settings)

    # Mock LLM that only generates plain text
    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(response_fn=lambda x: f"Response to: {x}"),
        )
    )

    # Configure sequence that won't match (looking for tools)
    config = TaskConfig(
        done_sequences=[
            DoneSequence(
                name="tool_sequence",
                events=[
                    AgentEvent(event_type=EventType.TOOL),
                    AgentEvent(event_type=EventType.AGENT_RESPONSE),
                ],
            )
        ]
    )

    task = Task(agent, config=config, interactive=False, allow_null_result=True)
    result = task.run("Say something", turns=3)

    # Task should run for all 3 turns since sequence doesn't match
    assert result is not None
    # Should have at least: system, user, assistant, user, assistant (2 turns minimum)
    assert len(agent.message_history) >= 5


def test_done_sequence_multiple_sequences(test_settings: Settings):
    """Test multiple done sequences"""
    set_global(test_settings)

    # Mock LLM that responds based on input
    def mock_response(x):
        if "urgent" in x.lower():
            return "I quit immediately"
        else:
            return '{"request": "simple_tool", "value": "normal"}'

    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(response_fn=mock_response),
        )
    )
    agent.enable_message(SimpleTool)

    # Multiple ways to be done
    config = TaskConfig(
        done_sequences=[
            # Quick exit on "quit"
            DoneSequence(
                name="quit_pattern",
                events=[
                    AgentEvent(
                        event_type=EventType.CONTENT_MATCH, content_pattern=r"\bquit\b"
                    )
                ],
            ),
            # Normal tool completion
            DoneSequence(
                name="tool_complete",
                events=[
                    AgentEvent(event_type=EventType.TOOL),
                    AgentEvent(event_type=EventType.AGENT_RESPONSE),
                ],
            ),
        ]
    )

    # Test quick exit
    task1 = Task(agent, config=config, interactive=False)
    result1 = task1.run("This is urgent!", turns=5)
    assert result1 is not None
    assert "quit" in result1.content.lower()
    history_len_1 = len(agent.message_history)

    # Test tool completion
    agent.clear_history()
    task2 = Task(agent, config=config, interactive=False)
    result2 = task2.run("Do something normal", turns=5)
    assert result2 is not None
    history_len_2 = len(agent.message_history)

    # Both should complete but potentially with different message counts
    assert history_len_1 == 3  # Quick exit: system, user, assistant
    assert history_len_2 == 3  # Tool completion: system, user, assistant (with tool)


def test_done_sequence_with_done_if_tool(test_settings: Settings):
    """Test that done_sequences works alongside done_if_tool"""
    set_global(test_settings)

    # First response is plain text, second is tool
    response_count = 0

    def mock_response(x):
        nonlocal response_count
        response_count += 1
        if response_count == 1:
            return "Thinking about it"
        else:
            return '{"request": "simple_tool", "value": "done"}'

    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(response_fn=mock_response),
        )
    )
    agent.enable_message(SimpleTool)

    # Both done_if_tool and done_sequences
    config = TaskConfig(
        done_if_tool=True,  # Should trigger first
        done_sequences=[
            DoneSequence(
                name="never_reached",
                events=[
                    AgentEvent(event_type=EventType.LLM_RESPONSE),
                    AgentEvent(event_type=EventType.LLM_RESPONSE),
                    AgentEvent(event_type=EventType.LLM_RESPONSE),
                ],
            )
        ],
    )

    task = Task(agent, config=config, interactive=False, allow_null_result=True)
    result = task.run("Do something", turns=10)

    assert result is not None
    # Should terminate after second LLM response (which has tool)
    # System, user, llm (plain), user (DO-NOT-KNOW), llm (tool)
    assert len(agent.message_history) == 5


def test_done_sequence_simulates_done_if_tool(test_settings: Settings):
    """Test that done_if_tool behavior can be approximated with a done sequence"""
    set_global(test_settings)

    # Mock LLM that generates a tool immediately
    def mock_response(x):
        return '{"request": "simple_tool", "value": "calculated"}'

    # Create two identical agents
    agent1 = ChatAgent(
        ChatAgentConfig(
            name="TestAgent1",
            llm=MockLMConfig(response_fn=mock_response),
        )
    )
    agent1.enable_message(SimpleTool)

    agent2 = ChatAgent(
        ChatAgentConfig(
            name="TestAgent2",
            llm=MockLMConfig(response_fn=mock_response),
        )
    )
    agent2.enable_message(SimpleTool)

    # Task 1: Using done_if_tool
    config1 = TaskConfig(done_if_tool=True)
    task1 = Task(agent1, config=config1, interactive=False)

    # Task 2: Using done_sequences to simulate done_if_tool
    config2 = TaskConfig(
        done_sequences=[
            DoneSequence(
                name="tool_generated",
                events=[
                    AgentEvent(event_type=EventType.TOOL),
                ],
            )
        ]
    )
    task2 = Task(agent2, config=config2, interactive=False)

    # Run both tasks
    result1 = task1.run("Calculate something", turns=10)
    result2 = task2.run("Calculate something", turns=10)

    # Both should complete successfully
    assert result1 is not None
    assert result2 is not None

    # Both approaches are equivalent - they check done conditions at the same point
    # in the task execution flow (in the done() method), so they produce identical
    # message histories
    assert len(agent1.message_history) == 3  # system, user, llm (with tool)
    assert len(agent2.message_history) == 3

    # Both should have the tool in their final LLM message
    assert "simple_tool" in agent1.message_history[-1].content
    assert "simple_tool" in agent2.message_history[-1].content

    # Verify they are truly equivalent by checking the exact same number of messages
    assert len(agent1.message_history) == len(agent2.message_history)


def test_done_sequence_tool_class_reference(test_settings: Settings):
    """Test using tool class names in done sequences"""
    set_global(test_settings)

    # Mock LLM that generates calculator tool
    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(
                response_fn=lambda x: '{"request": "calculator", "expression": "2+2"}'
            ),
        )
    )
    agent.enable_message([SimpleTool, CalculatorTool])

    # Use tool class name in done sequence
    config = TaskConfig(done_sequences=["T[CalculatorTool], A"])  # Using class name

    task = Task(agent, config=config, interactive=False)
    result = task.run("Calculate something")

    # The sequence is: LLM generates calculator tool -> Agent handles it -> done
    # Check that result contains the calculated result,  from handling the tool
    assert "4" in result.content

    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(
                response_fn=lambda x: '{"request": "calculator", "expression": "5*5"}'
            ),
        )
    )
    agent.enable_message([CalculatorTool])

    # Test with tool name
    config1 = TaskConfig(done_sequences=["T[calculator], A"])
    task1 = Task(agent, config=config1, interactive=False)
    result = task1.run("Calculate")
    # Result should contain the calculator tool's result
    assert "25" in result.content

    # Test with class name
    agent2 = ChatAgent(
        ChatAgentConfig(
            name="TestAgent2",
            llm=MockLMConfig(
                response_fn=lambda x: '{"request": "calculator", "expression": "5*5"}'
            ),
        )
    )
    agent2.enable_message([CalculatorTool])
    config2 = TaskConfig(done_sequences=["T[CalculatorTool], A"])
    task2 = Task(agent2, config=config2, interactive=False)
    result2 = task2.run("Calculate")
    assert "25" in result2.content

    # set up task to end as soon as the tool is generated, using the class name
    config3 = TaskConfig(done_sequences=["T[CalculatorTool]"])
    # ... and specialize the task to return the tool itself
    task3 = Task(agent2, config=config3, interactive=False)[CalculatorTool]
    result3: CalculatorTool | None = task3.run("Calculate")
    assert isinstance(result3, CalculatorTool)
    assert result3.expression == "5*5"
