#!/usr/bin/env python3
"""
Example demonstrating the new done_sequences feature in Langroid Tasks.

This feature allows you to specify sequences of events that trigger task completion,
providing more flexibility than simple done conditions.

You can use either:
1. DSL string patterns for convenience: "T, A" (tool then agent)
2. Full DoneSequence objects for more control

DSL Pattern Syntax:
- T = Any tool
- T[name] = Specific tool
- A = Agent response
- L = LLM response
- U = User response
- N = No response
- C[pattern] = Content matching regex

Note: Sequences use strict matching - events must occur consecutively in the message
chain without intervening messages. This ensures predictable behavior and efficient
matching.
"""

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import (
    AgentEvent,
    DoneSequence,
    EventType,
    Task,
    TaskConfig,
)
from langroid.agent.tool_message import ToolMessage


# Define a simple calculator tool
class CalculatorTool(ToolMessage):
    request: str = "calculator"
    purpose: str = "Perform arithmetic calculations"
    expression: str

    def handle(self) -> str:
        try:
            result = eval(self.expression)
            return f"The result is: {result}"
        except Exception as e:
            return f"Error: {str(e)}"


# Define a search tool
class SearchTool(ToolMessage):
    request: str = "search"
    purpose: str = "Search for information"
    query: str

    def handle(self) -> str:
        # Mock search implementation
        return f"Search results for '{self.query}': [Mock results here]"


def example0_dsl_syntax():
    """Example 0: Using DSL string patterns (recommended for simple cases)"""
    print("\n=== Example 0: DSL String Patterns ===")

    agent = ChatAgent(
        ChatAgentConfig(
            name="Assistant",
            system_message="""
            You are a helpful assistant with access to calculator and search tools.
            Use the appropriate tool when asked to calculate or search for something.
            """,
        )
    )
    agent.enable_message(CalculatorTool, use=True, handle=True)
    agent.enable_message(SearchTool, use=True, handle=True)

    # Using DSL string patterns - much more concise!
    config = TaskConfig(
        done_sequences=[
            "T, A",  # Any tool then agent response
            "T[calculator], A",  # Specific calculator tool
            "C[quit|exit|bye]",  # Content matching pattern
            "L, T, A, L",  # Complex sequence
        ]
    )

    _ = Task(agent, config=config)
    print("Task configured with multiple DSL patterns.")
    print(
        "Will complete on any of: tool use, calculator use, quit words, or L->T->A->L sequence"
    )
    # _ = task.run("What is 25 * 4?")
    # print(f"Final result: {result.content}")


def example1_tool_then_agent():
    """Example 1: Task completes after any tool is generated and handled by agent"""
    print("\n=== Example 1: Tool -> Agent Response ===")

    agent = ChatAgent(
        ChatAgentConfig(
            name="Assistant",
            system_message="""
            You are a helpful assistant with access to calculator and search tools.
            Use the appropriate tool when asked to calculate or search for something.
            """,
        )
    )
    agent.enable_message(CalculatorTool, use=True, handle=True)
    agent.enable_message(SearchTool, use=True, handle=True)

    # Task completes after: Tool -> Agent Response
    # Using DSL (recommended for simple patterns):
    # config = TaskConfig(done_sequences=["T, A"])

    # Using full syntax (for more control):
    config = TaskConfig(
        done_sequences=[
            DoneSequence(
                name="tool_handled",
                events=[
                    AgentEvent(event_type=EventType.TOOL),
                    AgentEvent(event_type=EventType.AGENT_RESPONSE),
                ],
            )
        ]
    )

    task = Task(agent, config=config)
    print("Task will complete after any tool is used and handled.")
    _ = task.run("What is 25 * 4?")
    # print(f"Final result: {_.content}")


def example2_specific_tool_sequence():
    """Example 2: Task completes only after specific tool (calculator) is used"""
    print("\n=== Example 2: Specific Tool Sequence ===")

    agent = ChatAgent(
        ChatAgentConfig(
            name="Assistant",
            system_message="""
            You help users with calculations and searches.
            Always use the appropriate tool.
            """,
        )
    )
    agent.enable_message(CalculatorTool, use=True, handle=True)
    agent.enable_message(SearchTool, use=True, handle=True)

    # Task completes only after calculator tool is used
    config = TaskConfig(
        done_sequences=[
            DoneSequence(
                name="calculation_done",
                events=[
                    AgentEvent(
                        event_type=EventType.SPECIFIC_TOOL, tool_name="calculator"
                    ),
                    AgentEvent(event_type=EventType.AGENT_RESPONSE),
                ],
            )
        ]
    )

    task = Task(agent, config=config)
    print("Task will complete only after calculator tool is used.")
    print("Try: 'Search for Python tutorials' (won't complete task)")
    print("Then try: 'Calculate 15 + 27' (will complete task)")
    _ = task.run()


def example3_conversation_pattern():
    """Example 3: Task completes after specific conversation pattern"""
    print("\n=== Example 3: Conversation Pattern ===")

    agent = ChatAgent(
        ChatAgentConfig(
            name="Assistant",
            system_message="""
            You are a step-by-step assistant. When asked to solve a problem:
            1. First acknowledge the request
            2. Then use the calculator tool
            3. Finally provide a summary of the result
            """,
        )
    )
    agent.enable_message(CalculatorTool, use=True, handle=True)

    # Task completes after: LLM -> Tool -> Agent -> LLM pattern
    config = TaskConfig(
        done_sequences=[
            DoneSequence(
                name="problem_solved",
                events=[
                    AgentEvent(event_type=EventType.LLM_RESPONSE),  # Acknowledgment
                    AgentEvent(event_type=EventType.TOOL),  # Calculator use
                    AgentEvent(event_type=EventType.AGENT_RESPONSE),  # Tool handled
                    AgentEvent(event_type=EventType.LLM_RESPONSE),  # Summary
                ],
            )
        ]
    )

    task = Task(agent, config=config)
    print("Task will complete after: acknowledgment -> tool use -> handling -> summary")
    _ = task.run(
        "I need to calculate the area of a rectangle with width 12 and height 8"
    )


def example4_multiple_completion_paths():
    """Example 4: Multiple ways to complete a task"""
    print("\n=== Example 4: Multiple Completion Paths ===")

    agent = ChatAgent(
        ChatAgentConfig(
            name="Assistant",
            system_message="""
            You help users with various tasks. 
            If they say 'quit' or 'exit', acknowledge and stop.
            Otherwise, help them with calculations or searches.
            """,
        )
    )
    agent.enable_message(CalculatorTool, use=True, handle=True)
    agent.enable_message(SearchTool, use=True, handle=True)

    # Multiple ways to complete the task
    config = TaskConfig(
        done_sequences=[
            # Path 1: User says quit/exit
            DoneSequence(
                name="user_quit",
                events=[
                    AgentEvent(
                        event_type=EventType.CONTENT_MATCH,
                        content_pattern=r"\b(quit|exit|bye|goodbye)\b",
                    ),
                ],
            ),
            # Path 2: Calculator tool used
            DoneSequence(
                name="calculation_done",
                events=[
                    AgentEvent(
                        event_type=EventType.SPECIFIC_TOOL, tool_name="calculator"
                    ),
                    AgentEvent(event_type=EventType.AGENT_RESPONSE),
                ],
            ),
            # Path 3: Two searches performed
            DoneSequence(
                name="double_search",
                events=[
                    AgentEvent(event_type=EventType.SPECIFIC_TOOL, tool_name="search"),
                    AgentEvent(event_type=EventType.AGENT_RESPONSE),
                    AgentEvent(event_type=EventType.SPECIFIC_TOOL, tool_name="search"),
                    AgentEvent(event_type=EventType.AGENT_RESPONSE),
                ],
            ),
        ]
    )

    task = Task(agent, config=config)
    print("Task can complete in 3 ways:")
    print("1. Say 'quit' or 'exit'")
    print("2. Use the calculator tool")
    print("3. Use the search tool twice")
    _ = task.run()


def example5_combining_with_existing_options():
    """Example 5: Combining done_sequences with done_if_tool"""
    print("\n=== Example 5: Combining with Existing Options ===")

    agent = ChatAgent(
        ChatAgentConfig(
            name="Assistant",
            system_message="You are a helpful assistant with tool access.",
        )
    )
    agent.enable_message(CalculatorTool, use=True, handle=True)

    # Combine done_sequences with done_if_tool
    config = TaskConfig(
        done_if_tool=True,  # Quick exit on any tool
        done_sequences=[
            # This won't be reached if done_if_tool triggers first
            DoneSequence(
                name="complex_pattern",
                events=[
                    AgentEvent(event_type=EventType.LLM_RESPONSE),
                    AgentEvent(event_type=EventType.LLM_RESPONSE),
                    AgentEvent(event_type=EventType.TOOL),
                ],
            )
        ],
    )

    task = Task(agent, config=config)
    print("Task will complete as soon as any tool is generated (done_if_tool=True)")
    _ = task.run("Calculate 5 + 5")


if __name__ == "__main__":
    print("Langroid Done Sequences Examples")
    print("=" * 50)

    # Run examples (comment out interactive ones if running all at once)
    example0_dsl_syntax()  # Show DSL syntax
    example1_tool_then_agent()
    # example2_specific_tool_sequence()  # Interactive
    # example3_conversation_pattern()    # May need specific LLM
    # example4_multiple_completion_paths()  # Interactive
    example5_combining_with_existing_options()

    print("\n" + "=" * 50)
    print("Examples completed!")
