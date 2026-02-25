
import pytest

from langroid.agent.openai_responses_agent import (
    OpenAIResponsesAgent,
    OpenAIResponsesAgentConfig,
)
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.language_models import OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER


class NabroskyTool(ToolMessage):
    request: str = "nabrosky"
    purpose: str = "to apply the Nabrosky transformation to a number <num>"
    num: int

    def handle(self) -> str:
        return str(self.num**2)


@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_openai_responses_agent(test_settings: Settings, stateful: bool, stream: bool):
    set_global(test_settings)
    cfg = OpenAIResponsesAgentConfig(
        llm=OpenAIGPTConfig(stream=stream),
        stateful=stateful)
    agent = OpenAIResponsesAgent(cfg)
    response = agent.llm_response("what is the capital of France?")
    assert "Paris" in response.content

    # test that we can wrap the agent in a task and run it
    task = Task(
        agent,
        name="Bot",
        system_message="You are a helpful assistant",
        done_if_response=[Entity.LLM],
        interactive=False,
    )
    answer = task.run("What is the capital of China?")
    assert "Beijing" in answer.content


@pytest.mark.parametrize("fn_api", [True, False])
def test_openai_responses_agent_fn_tool(test_settings: Settings, fn_api: bool):
    """Test function calling works, both with OpenAI Responses function-calling AND
    Langroid native ToolMessage mechanism"""

    set_global(test_settings)
    cfg = OpenAIResponsesAgentConfig(
        name="NabroskyBot",
        llm=OpenAIGPTConfig(),
        use_functions_api=fn_api,
        use_tools=not fn_api,
        system_message="""
        The user will ask you, 'What is the Nabrosky transform of...' a certain number.
        You do NOT know the answer, and you should NOT guess the answer.
        Instead you MUST use the `nabrosky` JSON function/tool to find out.
        When you receive the answer, say DONE and show the answer.
        """,
    )
    agent = OpenAIResponsesAgent(cfg)
    agent.enable_message(NabroskyTool)
    response = agent.llm_response("what is the Nabrosky transform of 5?")
    # Check assert when there is a non-empty response
    if response.content not in ("", NO_ANSWER) and fn_api:
        assert response.function_call.name == "nabrosky"

    # Within a task loop
    cfg.name = "NabroskyBot-1"
    agent = OpenAIResponsesAgent(cfg)
    agent.enable_message(NabroskyTool)
    task = Task(
        agent,
        interactive=False,
    )
    result = task.run("what is the Nabrosky transform of 5?", turns=4)
    # When fn_api = False (i.e. using ToolMessage) we get brittleness so we just make
    # sure there is no error until this point.
    if result.content not in ("", NO_ANSWER) and fn_api:
        assert "25" in result.content


@pytest.mark.parametrize("fn_api", [True, False])
def test_openai_responses_agent_fn_2_level(test_settings: Settings, fn_api: bool):
    """Test 2-level recursive function calling works,
    both with OpenAI Responses function-calling AND
    Langroid native ToolMessage mechanism"""

    set_global(test_settings)
    cfg = OpenAIResponsesAgentConfig(
        name="Main",
        use_functions_api=fn_api,
        use_tools=not fn_api,
        system_message="""
        The user will ask you to apply the Nabrosky transform to a number.
        You do not know how to do it, and you should NOT guess the answer.
        Instead you MUST use the `recipient_message` tool/function to
        send it to NabroskyBot who will do it for you.
        When you receive the answer, say DONE and show the answer.
        """,
    )
    agent = OpenAIResponsesAgent(cfg)
    agent.enable_message(RecipientTool)

    nabrosky_cfg = OpenAIResponsesAgentConfig(
        name="NabroskyBot",
        use_functions_api=fn_api,
        use_tools=not fn_api,
        system_message="""
        The user will ask you to apply the Nabrosky transform to a number.
        You do not know how to do it, and you should NOT guess the answer.
        Instead you MUST use the `nabrosky` function/tool to do it.
        When you receive the answer say DONE and show the answer.
        """,
    )

    nabrosky_agent = OpenAIResponsesAgent(nabrosky_cfg)
    nabrosky_agent.enable_message(NabroskyTool)

    main_task = Task(agent, interactive=False)
    nabrosky_task = Task(nabrosky_agent, interactive=False)
    main_task.add_sub_task(nabrosky_task)
    result = main_task.run("what is the Nabrosky transform of 5?", turns=6)
    if fn_api and result.content not in ("", NO_ANSWER):
        assert "25" in result.content


@pytest.mark.parametrize("fn_api", [True, False])
def test_openai_responses_agent_recipient_tool(test_settings: Settings, fn_api: bool):
    """Test that special case of fn-calling: RecipientTool works,
    both with OpenAI Responses function-calling AND
    Langroid native ToolMessage mechanism"""

    set_global(test_settings)
    cfg = OpenAIResponsesAgentConfig(
        name="Main",
        use_functions_api=fn_api,
        use_tools=not fn_api,
        system_message="""
        The user will give you a number. You need to double it, but don't know how,
        so you send it to the "Doubler" to double it.
        When you receive the answer, say DONE and show the answer.
        """,
    )
    agent = OpenAIResponsesAgent(cfg)
    agent.enable_message(RecipientTool)

    # Within a task loop
    doubler_config = OpenAIResponsesAgentConfig(
        name="Doubler",
        system_message="""
        When you receive a number, simply double it and  return the answer
        """,
    )
    doubler_agent = OpenAIResponsesAgent(doubler_config)
    doubler_task = Task(
        doubler_agent,
        interactive=False,
        done_if_response=[Entity.LLM],
    )

    main_task = Task(agent, interactive=False)
    main_task.add_sub_task(doubler_task)
    result = main_task.run("10", turns=4)
    if fn_api and result.content not in ("", NO_ANSWER):
        assert "20" in result.content


def test_openai_responses_agent_multi(test_settings: Settings):
    """
    Test task delegation with OpenAIResponsesAgent
    """
    set_global(test_settings)

    cfg = OpenAIResponsesAgentConfig(
        name="Teacher",
    )
    agent = OpenAIResponsesAgent(cfg)

    # wrap Agent in a Task to run interactive loop with user (or other agents)
    task = Task(
        agent,
        interactive=False,
        system_message="""
        Send a number. Your student will respond EVEN or ODD.
        You say RIGHT DONE or WRONG DONE.

        Start by sending a number.
        """,
    )

    cfg = OpenAIResponsesAgentConfig(
        name="Student",
    )
    student_agent = OpenAIResponsesAgent(cfg)
    student_task = Task(
        student_agent,
        interactive=False,
        done_if_response=[Entity.LLM],
        system_message="When you get a number, say EVEN if it is even, else say ODD",
    )
    task.add_sub_task(student_task)
    result = task.run()
    assert "RIGHT" in result.content
