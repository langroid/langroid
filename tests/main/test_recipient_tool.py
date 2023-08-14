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
from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global

INPUT_NUMBERS = [1, 100, 2, 21, 4, 40, 33]
TRANSFORMED_NUMBERS = [4, 50, 1, 64, 2, 20, 100]


@pytest.mark.parametrize("fn_api", [True, False])
def test_agents_with_recipient_tool(test_settings: Settings, fn_api: bool):
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
    processor_agent.enable_message(RecipientTool)
    processor_task = Task(
        processor_agent,
        name="Processor",
        default_human_response="",
        only_user_quits_root=False,
        system_message=f"""
        You are given this list of numbers:
        {INPUT_NUMBERS} 
        Your goal is to apply a transformation to each number.
        However you do not know how to do this transformation.
        You can take the help of two people to perform the 
        transformation.
        If the number is even, send it to EvenHandler,
        and if it is odd, send it to OddHandler.
        
        IMPORTANT: send the numbers ONE AT A TIME
        
        The handlers will transform the number and give you a new number.        
        If you send it to the wrong person, you will receive a negative value.
        Your aim is to never get a negative number, so you must 
        clearly specify who you are sending the number to, using the
        `recipient_message` tool/function-call, where the `content` field
        is the number you want to send, and the `recipient` field is the name
        of the intended recipient, either "EvenHandler" or "OddHandler".
        
        You are also forgetful when it comes to numbers bigger than 10.
        When a number is bigger than 10, just send your number without specifying
        a recipient, as a simple non-JSON string, and do this without explaining why 
        you are doing this, and do not correct yourself until I remind you.
        When I receive this message, I will remind you to correct this, and you can 
        then correct yourself. For example when trying to transform 25, since it is 
        bigger than 10, you will send me simply the  message "25", and say nothing else.
        
        Once all numbers in the given list have been transformed, 
        say DONE and show me the result.
        
        Start by requesting a transformation for the first number.
        """,
        llm_delegate=True,
        single_round=False,
    )
    even_agent = ChatAgent(config)
    even_task = Task(
        even_agent,
        name="EvenHandler",
        default_human_response="",
        system_message="""
        You will be given a number. 
        If it is even, divide by 2 and say the result, nothing else.
        If it is odd, say -10
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
