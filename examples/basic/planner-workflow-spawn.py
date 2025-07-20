"""
Illustrates a Planner agent orchestrating a multi-step workflow by using the `TaskTool`
to dynamically spawn specialized sub-agents for each step.

- The PlannerAgent is instructed to first increment a number by 3, and then
  multiply the result by 8.
- To do this, it uses the `TaskTool` to dynamically create and run sub-tasks.
- For the incrementing part, it spawns a simple `IncrementAgent` three times.
- For the multiplication part, it spawns a simple `DoublingAgent` three times.

This example showcases a powerful pattern where a high-level agent can delegate
complex sub-processes to dynamically created, specialized agents without needing
them to be pre-defined in the main script.

Run like this from the repo root:

    uv run examples/basic/planner-workflow-spawn.py

To use a different model, for example gpt-4-turbo, run:

    uv run examples/basic/planner-workflow-spawn.py --model gpt-4-turbo

"""

import logging

from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import DoneTool, ResultTool
from langroid.agent.tools.task_tool import TaskTool

logger = logging.getLogger(__name__)
MODEL = lm.OpenAIChatModel.GPT4_1


async def main(model: str = ""):

    class PlannerConfig(lr.ChatAgentConfig):
        name: str = "Planner"
        handle_llm_no_tool: str = "You FORGOT to use one of your TOOLs!"
        llm: lm.OpenAIGPTConfig = lm.OpenAIGPTConfig(
            chat_model=model or MODEL,
        )
        system_message: str = f""" 
        You are a Planner that has ZERO knowledge about MATH/ARITHMETIC!
        
        Your job is to process a number given by the user through a sequence of 2 steps:

        1.  **Increment the number by 3.**
        2.  **Multiply the resulting number by 8.**

        HOWEVER, you CANNOT do these steps yourself, so you instead 
        MUST use the `{TaskTool.name()}` to spawn a sub-agent for one of
        the following tasks as you see fit:
        
        - Increment a given number by 1
        - Double a given number
        
        The sub-agent can use "gpt-4.1-mini" as the model,
        and does not need any tools enabled.
        
        Keep track of the intermediate results.

        Once you have the final result, you MUST use the `{DoneTool.name()}` to return it.
        """

    planner = lr.ChatAgent(PlannerConfig())

    planner.enable_message([TaskTool, DoneTool])

    planner_task = lr.Task(planner, interactive=False)

    # Initial number is 3.
    # After incrementing 3 times: 3 + 3 = 6
    # After doubling 3 times: 6 * 2 * 2 * 2 = 48
    result = await planner_task.run_async("Process this number: 3")
    assert "48" in result.content, f"Expected 48, got {result.content}"


if __name__ == "__main__":
    Fire(main)
