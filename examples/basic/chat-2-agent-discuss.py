# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langroid",
# ]
# ///

"""
Give a problem statement, two agents Alice and Bob will discuss it,
and EITHER of them may return a final result via MyFinalResultTool.

Run like this (Omit model to default to GPT4o):

python3 examples/basic/chat-2-agent-discuss.py --model gemini/gemini-2.0-flash-exp

For example, try giving his problem:
What is the prime number that comes after 17?

"""

import logging

from fire import Fire
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
from langroid.agent.task import TaskConfig
from langroid.agent.tools.orchestration import FinalResultTool

# set info level
logging.basicConfig(level=logging.INFO)


# Any tool subclassed from FinalResultTool can be used to return the final result
# from any agent, and it will short-circuit the flow and return the result.
class MyFinalResultTool(FinalResultTool):
    request: str = "my_final_result_tool"
    purpose: str = "To present the final <result> of a discussion"
    # override this flag since it's False by default
    _allow_llm_use: bool = True

    result: str


def main(model: str = ""):
    problem = Prompt.ask(
        """
        [blue]Alice and Bob will discuss a problem.
        Please enter the problem statement:[/blue]
        """
    )

    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=128_000,
        timeout=60,
    )

    logging.warning("Setting up Alice, Bob agents...")

    alice = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm_config,
            name="Alice",
            system_message=f"""
            Here is a problem the user wants to solve:
            <problem>
            {problem}
            </problem>
            To solve this, you will engage in a discussion with your colleague Bob.
            At any point, if you decide the problem is solved,
            you must use the TOOL `{MyFinalResultTool.name()}` to
            return the FINAL answer to the problem. 

            In each round of the discussion, limit yourself to a CONCISE
            message.
            """,
        )
    )

    alice.enable_message(MyFinalResultTool)
    # Set `inf_loop_cycle_len` to 0, to turn OFF inf loop detection
    alice_task_config = TaskConfig(inf_loop_cycle_len=10)
    # set up alice_task to return a result of type MyFinalResultTool
    alice_task = lr.Task(alice, config=alice_task_config, interactive=False)[
        MyFinalResultTool
    ]

    bob = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm_config,
            name="Bob",
            system_message=f"""
            Here is a problem the user wants to solve:
            <problem>
            {problem}
            </problem>
            To solve this, you will engage in a discussion with your colleague Alice.
            At any point, if you decide the problem is solved,
            you must use the TOOL `{MyFinalResultTool.name()}` to
            return the FINAL answer to the problem. 

            In each round of the discussion, limit yourself to a CONCISE
            message. 
            
            You will first receive a message from Alice, and you can then follow up. 
            """,
        )
    )

    bob.enable_message(MyFinalResultTool)

    bob_task = lr.Task(bob, interactive=False, single_round=True)

    # make the Con agent the sub-task of the Pro agent, so
    # they go back and forth in the arguments
    alice_task.add_sub_task(bob_task)

    result = alice_task.run("get started")

    print(
        f"""
        FINAL RESULT:
        {result.result}
        """
    )


if __name__ == "__main__":
    Fire(main)
