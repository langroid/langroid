"""
Use Langroid to set up a collaboration among three agents:

- Processor: needs to transform a list of positive numbers, does not know how to
apply the transformations, and sends out each number so that one of two
specialized agents apply the transformation. It is instructed to avoid getting a
negative number.
- EvenHandler only transforms even numbers, otherwise returns a negative number
- OddHandler only transforms odd numbers, otherwise returns a negative number

Since the Processor must avoid getting a negative number, it needs to
specify a recipient for each number it sends out,
using the `recipient_message` tool/function-call, where the `content` field
is the number it wants to send, and the `recipient` field is the name
of the intended recipient, either "EvenHandler" or "OddHandler".

However, the Processor often forgets to use this syntax, and in this situation
the `handle_message_fallback` method of the RecipientTool class
asks the Processor to clarify the intended recipient using the
`add_recipient` tool, which allows the LLM to simply specify a recipient for
its last message, without having to repeat the message.

For more explanation, see the
[Getting Started guide](https://langroid.github.io/langroid/quick-start/three-agent-chat-num-router/)
"""

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import DoneTool
from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.language_models.mock_lm import MockLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.utils.constants import DONE

INPUT_NUMBERS = [1, 100, 12]
TRANSFORMED_NUMBERS = [4, 10000, 6]


class SquareTool(ToolMessage):
    request: str = "square"
    purpose: str = "To square a <number>, when it is a multiple of 10."
    number: int

    # this is a stateless tool, so we can define the handler here,
    # without having to define a `square` method in the agent.
    def handle(self) -> str | DoneTool:
        if self.number % 10 == 0:
            return DONE + str(self.number**2)
        else:
            # test that DoneTool works just like saying DONE
            return DoneTool(content="-1")


@pytest.mark.fallback
@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize("fn_api", [True, False])
@pytest.mark.parametrize("tools_api", [True, False])
@pytest.mark.parametrize("constrain_recipients", [True, False])
@pytest.mark.parametrize("done_tool", [True, False])
def test_agents_with_recipient_tool(
    fn_api: bool,
    tools_api: bool,
    constrain_recipients: bool,
    done_tool: bool,
):
    config = ChatAgentConfig(
        llm=OpenAIGPTConfig(),
        use_tools=not fn_api,
        use_tools_api=tools_api,
        use_functions_api=fn_api,
        vecdb=None,
    )
    processor_agent = ChatAgent(config)

    if constrain_recipients:
        processor_agent.enable_message(
            RecipientTool.create(recipients=["EvenHandler", "OddHandler"])
        )
    else:
        processor_agent.enable_message(RecipientTool)

    processor_agent.enable_message(
        SquareTool, require_recipient=True, use=True, handle=False
    )
    if done_tool:
        processor_agent.enable_message(DoneTool)
        done_tool_name = DoneTool.default_value("request")

    done_response = (
        f"use the TOOL: `{done_tool_name}` with `content` field set to the result"
        if done_tool
        else f"say {DONE} and show me the result"
    )

    processor_task = Task(
        processor_agent,
        name="Processor",
        interactive=False,
        system_message=f"""
        You are given this list of {len(INPUT_NUMBERS)} numbers:
        {INPUT_NUMBERS}. 
        You have to transform each number to a new POSITIVE value.
        However you do not know how to do this transformation.
        You can send the number to one of two people to do the 
        transformation: 
        - EvenHandler (who handles only even numbers),
        - OddHandler (who handles only odd numbers). 
        
        There are 3 cases, depending on the number n
        
        (a) If n is even:
         (a.1) if n is a multiple of 10, send it to EvenHandler,
             using the `square` tool/function-call, specifying the `recipient` 
             field as "EvenHandler".
         (a.2) if n is NOT a multiple of 10, send it to EvenHandler,
             
        (b) If n is odd, send it to OddHandler. 
        
        IMPORTANT: send the numbers ONE AT A TIME. Your message content
        should ONLY be numbers, do not say anything else, other than specifying
        recipients etc.
        
        The handlers will transform the number and give you the result.
        If you deviate from the above rules 
        (i.e. you send it to the wrong person or using the wrong tool/function), 
        you will receive a value of -10.
        Your task is to avoid getting negative values, by making sure you
        follow the above rules. If you ever get a negative value, correct yourself
        in the next step.
        
        Once all {len(INPUT_NUMBERS)} numbers in the given list have been transformed
        to positive values, 
        {done_response}, 
        showing only the positive transformations, 
        in the same order as the original list.
        
        Start by requesting a transformation for the first number.
        Be very concise in your messages, do not say anything unnecessary.
        """,
    )
    even_agent = ChatAgent(
        ChatAgentConfig(
            llm=MockLMConfig(
                response_fn=lambda x: (
                    str(int(x) // 2) if int(x) % 2 == 0 and int(x) % 10 != 0 else "-10"
                )
            )
        )
    )
    even_agent.enable_message(
        SquareTool,
        use=False,  # LLM of this agent does not need to generate this tool/fn-call
        handle=True,  # this agent needs to handle this tool/fn-call
        require_recipient=False,
    )
    even_task = Task(
        even_agent,
        name="EvenHandler",
        interactive=False,
        done_if_response=[Entity.LLM],  # done as soon as LLM responds
    )

    odd_agent = ChatAgent(
        ChatAgentConfig(
            llm=MockLMConfig(
                response_fn=lambda x: str(int(x) * 3 + 1) if int(x) % 2 else "-10"
            )
        )
    )
    odd_task = Task(
        odd_agent,
        name="OddHandler",
        interactive=False,
        done_if_response=[Entity.LLM],  # done as soon as LLM responds
    )

    processor_task.add_sub_task([even_task, odd_task])
    result = processor_task.run()
    assert all(str(i) in result.content for i in TRANSFORMED_NUMBERS)
