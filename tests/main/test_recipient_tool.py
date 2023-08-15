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
from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global

INPUT_NUMBERS = [1, 100, 2, 21, 4, 40, 33]
TRANSFORMED_NUMBERS = [4, 10000, 1, 64, 2, 1600, 100]


class SquareTool(ToolMessage):
    request: str = "square"
    purpose: str = "To square a <number>, when it is a multiple of 10."
    number: int

    # this is a stateless tool, so we can define the handler here,
    # without having to define a `square` method in the agent.
    def handle(self) -> str:
        if self.number % 10 == 0:
            return str(self.number**2)
        else:
            return "-1"


@pytest.mark.parametrize("fn_api", [True, False])
@pytest.mark.parametrize("constrain_recipients", [True, False])
def test_agents_with_recipient_tool(
    test_settings: Settings,
    fn_api: bool,
    constrain_recipients: bool,
):
    set_global(test_settings)
    config = ChatAgentConfig(
        llm=OpenAIGPTConfig(
            chat_model=OpenAIChatModel.GPT4,
        ),
        use_tools=not fn_api,
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
    processor_task = Task(
        processor_agent,
        name="Processor",
        default_human_response="",
        only_user_quits_root=False,
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
             using the `square` tool/function-call, specifying the `recipient` field 
             as "EvenHandler".
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
        to positive values, say DONE and show me the result, showing only the 
        positive transformations, in the same order as the original list.
        
        Start by requesting a transformation for the first number.
        Be very concise in your messages, do not say anything unnecessary.
        """,
        llm_delegate=True,
        single_round=False,
    )

    # You are also forgetful when it comes to numbers bigger than 10.
    # When a number is bigger than 10, just send your number without specifying
    # a recipient, as a simple non-JSON string, and do this without explaining why
    # you are doing this, and do not correct yourself until I remind you.
    # When I receive this message, I will remind you to correct this, and you can
    # then correct yourself. For example when trying to transform 25, since it is
    # bigger than 10, you will send me simply the  message "25", and say nothing else.

    even_agent = ChatAgent(config)
    even_agent.enable_message(
        SquareTool,
        use=False,  # LLM of this agent does not need to generate this tool/fn-call
        handle=True,  # this agent needs to handle this tool/fn-call
        require_recipient=False,  # this agent does not to specify recipient
    )
    even_task = Task(
        even_agent,
        name="EvenHandler",
        default_human_response="",
        system_message="""
        You will be given a number. 
        If it is not a multiple of 10, and even, 
        divide by 2 and say the result, nothing else.
        Otherwise, say -10
        """,
        single_round=True,  # task done after 1 step() with valid response
    )

    odd_agent = ChatAgent(config)
    odd_task = Task(
        odd_agent,
        name="OddHandler",
        default_human_response="",
        system_message="""
        You will be given a number n. 
        If it is odd, return (n*3+1), say nothing else. 
        If it is even, say -10
        """,
        single_round=True,  # task done after 1 step() with valid response
    )

    processor_task.add_sub_task([even_task, odd_task])
    result = processor_task.run()
    assert all(str(i) in result.content for i in TRANSFORMED_NUMBERS)
