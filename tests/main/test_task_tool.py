import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task, TaskConfig
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import DoneTool
from langroid.agent.tools.task_tool import TaskTool
from langroid.language_models.mock_lm import MockLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig


class MultiplierTool(ToolMessage):
    """A simple calculator tool for testing."""

    request: str = "multiplier_tool"
    purpose: str = "To calculate the product of two numbers."
    a: int
    b: int

    def handle(self) -> str:
        return self.a * self.b


def test_task_tool_mock_main_agent():
    """
    Test that when MockAgent uses TaskTool, it  properly spawns a sub-agent
    that can use tools and complete tasks.
    """

    # Configure the main agent to use TaskTool:
    # The MockLM has a fixed response, which is the TaskTool request
    main_config = ChatAgentConfig(
        llm=MockLMConfig(
            default_response=TaskTool(
                system_message=f"""
                    You are a calculator assistant. When asked to 
                    calculate, use the TOOL `{MultiplierTool.name()}` to multiply the 
                    numbers, then use the TOOL `{DoneTool.name()}` to return the result
                    """,
                prompt="Multiply 5 and 7",
                model="gpt-4.1-mini",
                tools=["multiplier_tool"],
                max_iterations=5,
            ).json()
        ),
        name="MainAgent",
    )
    main_agent = ChatAgent(main_config)

    # Enable TaskTool and MultiplierTool for the main agent.
    # The MultiplierTool must be enabled for the main agent,
    # since the TaskTool handler will create a sub-agent that uses it,
    # and the handler only has access to tools enabled for the main agent.
    main_agent.enable_message([TaskTool, MultiplierTool], use=True, handle=True)

    # Create main task that stops after handling the task_tool
    task = Task(
        main_agent,
        name="MainTask",
        interactive=False,
        config=TaskConfig(
            done_sequences=["T,A"],  # LLM (Tool (TaskTool)), Agent(Handled) -> done
        ),
    )

    # Run the task
    result = task.run(msg="Please calculate something")

    # The result should be from the sub-agent's execution
    assert result is not None, "Task should return a result"
    assert "35" in result.content, "Result should contain the multiplication result"


class NebrowskiTool(ToolMessage):
    """A tool that computes the exotic Nebrowski operation."""

    request: str = "nebrowski_tool"
    purpose: str = """
        To compute the Nebrowski operation of two numbers: 
        neb(a,b) = 3a + b
    """
    a: int
    b: int

    def handle(self) -> str:
        # The Nebrowski operation: 3a + b
        result = 3 * self.a + self.b
        return f"Nebrowski({self.a}, {self.b}) = {result}"


def _create_nebrowski_task():
    """
    Helper function to create a Nebrowski task for both sync and async tests.
    Returns a configured Task ready to run.
    """
    # Configure the main agent with a real LLM
    main_config = ChatAgentConfig(
        llm=OpenAIGPTConfig(),  # Uses default model
        handle_llm_no_tool="you forgot to use one of your TOOLs!",
        system_message=f"""
        You are a Nebrowski operation specialist. The Nebrowski operation is an exotic 
        mathematical function that takes two numbers and produces a result.
        BUT you do NOT know how to compute it yourself!
        
        When the user asks you to compute nested Nebrowski operations like 
        Nebrowski(a, Nebrowski(b, c)), you MUST:
        
        1. Break it down into individual Nebrowski operations
        2. Use the TOOL `{TaskTool.name()}` to delegate each Nebrowski 
            operation to a sub-agent
        3. The sub-agent knows how to use the `{NebrowskiTool.name()}` tool
        
        For example, to compute Nebrowski(10, Nebrowski(3, 2)):
        - First compute inner: Nebrowski(3, 2) = result1 (using TaskTool)
        - Then compute outer: Nebrowski(10, result1) (using TaskTool)
        - Return the final result
        
        IMPORTANT: You must use TaskTool for EACH Nebrowski operation.
        Configure the TaskTool with:
        - system_message: Instructions for the sub-agent to compute Nebrowski
        - prompt: The specific Nebrowski task (e.g., "Compute Nebrowski(3, 2)")
        - tools: ["nebrowski_tool"]
        - model: "gpt-4o-mini"
        
        Remember: You cannot compute Nebrowski operations yourself - you must 
        delegate to sub-agents!
        
        You MUST use the TOOL `{DoneTool.name()}` to return the final result!
        """,
        name="NebrowskiAgent",
    )
    main_agent = ChatAgent(main_config)

    # Enable TaskTool and NebrowskiTool
    main_agent.enable_message(
        [DoneTool, TaskTool, NebrowskiTool], use=True, handle=True
    )

    # Create task with appropriate configuration
    task = Task(
        main_agent,
        name="NebrowskiTask",
        interactive=False,
    )

    return task


def test_task_tool_real_llm_nebrowski():
    """
    Test that a real LLM agent can compute nested Nebrowski operations
    by using TaskTool to delegate each Nebrowski computation to sub-agents.
    """
    task = _create_nebrowski_task()

    # Run the task - compute Nebrowski(10, Nebrowski(3, 2))
    # Expected: Nebrowski(3, 2) = 11, then Nebrowski(10, 11) = 41
    result = task.run("Compute Nebrowski(10, Nebrowski(3, 2))", turns=15)

    # Verify the result
    assert result is not None, "Task should return a result"
    assert "41" in result.content, "Result should contain the final Nebrowski result"


@pytest.mark.asyncio
async def test_task_tool_real_llm_nebrowski_async():
    """
    Async version: Test that a real LLM agent can compute nested Nebrowski operations
    by using TaskTool to delegate each Nebrowski computation to sub-agents.
    """
    task = _create_nebrowski_task()

    # Run the task asynchronously - compute Nebrowski(10, Nebrowski(3, 2))
    # Expected: Nebrowski(3, 2) = 11, then Nebrowski(10, 11) = 41
    result = await task.run_async("Compute Nebrowski(10, Nebrowski(3, 2))", turns=15)

    # Verify the result
    assert result is not None, "Task should return a result"
    assert "41" in result.content, "Result should contain the final Nebrowski result"
