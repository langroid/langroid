"""
Illustrates a Planner agent orchestrating a multi-step workflow by using tools that
invoke other specialized agents.

- The PlannerAgent is instructed to first increment a number by 3, and then
  multiply the result by 8.
- To do this, it repeatedly uses two tools: `IncrementTool` and `DoublingTool`.
- The key idea is that these tools are stateful: their `handle_async` methods
  don't perform the simple math themselves, but instead run other `Task` objects
  (`increment_task`, `doubling_task`).
- These tasks are handled by simple, specialized agents (`IncrementAgent`,
  `DoublingAgent`) that only know how to perform a single, small step.

This example showcases a powerful pattern where a high-level agent delegates complex
sub-processes to other agents via the tool mechanism.

Run like this from the repo root, once you are in a virtual environment with
langroid installed:

    uv run examples/basic/planner-workflow-simple.py

To use a different model, for example, run like this:

    uv run examples/basic/planner-workflow-simple.py --model gpt-4.1-mini

"""

import logging

from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import DoneTool
from pydantic import Field

logger = logging.getLogger(__name__)
MODEL = lm.OpenAIChatModel.GPT4_1


class IncrementAgentConfig(lr.ChatAgentConfig):
    name: str = "Incrementer"
    system_message: str = "Given a number, return the next number"


class DoublingAgentConfig(lr.ChatAgentConfig):
    name: str = "Doubler"
    system_message: str = "Given a number, return the number multiplied by 2"


async def main(model: str = ""):

    increment_agent = lr.ChatAgent(
        IncrementAgentConfig(
            llm=lm.OpenAIGPTConfig(
                chat_model=model or MODEL,
                async_stream_quiet=False,
            )
        )
    )
    increment_task = lr.Task(
        increment_agent,
        interactive=False,
        single_round=True,
    )

    doubling_agent = lr.ChatAgent(
        DoublingAgentConfig(
            llm=lm.OpenAIGPTConfig(
                chat_model=model or MODEL,
                async_stream_quiet=False,
            )
        )
    )

    doubling_task = lr.Task(
        doubling_agent,
        interactive=False,
        single_round=True,
    )

    class IncrementTool(lr.ToolMessage):
        request: str = "increment_tool"
        purpose: str = "To increment a <number> by 1"
        number: int = Field(..., description="The number (int) to Increment")

        async def handle_async(self) -> str:
            # stateful tool: handler runs the increment_task
            result = await increment_task.run_async(f"{self.number}")
            return result.content

    class DoublingTool(lr.ToolMessage):
        request: str = "doubling_tool"
        purpose: str = "To double a <number>"
        number: int = Field(..., description="The number (int) to Double")

        async def handle_async(self) -> str:
            # stateful tool: handler runs the doubling_task
            result = await doubling_task.run_async(self.number)
            return result.content

    class PlannerConfig(lr.ChatAgentConfig):
        name: str = "Planner"
        handle_llm_no_tool: str = "You FORGOT to use one of your TOOLs!"
        llm: lm.OpenAIGPTConfig = lm.OpenAIGPTConfig(
            chat_model=model or MODEL,
            async_stream_quiet=False,
        )
        system_message: str = f"""
        You are a Planner in charge of PROCESSING the user's input number
        (an integer) through a SEQUENCE of two steps:
        
        1. Increment the number by 3 -- use the `{IncrementTool.name()}` tool,
            as many times as needed, until the number is incremented by 3.
        2. Multiply the number by 8 -- use the `{DoublingTool.name()}` tool,
            as many times as needed, until the number is multiplied by 8.
            
        Note That even though these tasks sound trivial, you cannot and must not do them 
        yourself. You must use the tools as many times as needed for each step and then 
        proceed to the next step. 
        
        CRITICAL: You must call ONE TOOL only and wait for its result, 
        and then call another tool. 
        NEVER EVER call multiple tools at the same time.  
        
        Once you are done, use the TOOL `{DoneTool.name()}` to return the final result.
        """

    planner = lr.ChatAgent(PlannerConfig())

    planner.enable_message([IncrementTool, DoublingTool, DoneTool])

    planner_task = lr.Task(planner, interactive=False)

    result = await planner_task.run_async("Process this number: 3")
    assert "48" in result.content, f"Expected 48, got {result.content}"


if __name__ == "__main__":
    Fire(main)
