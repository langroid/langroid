import pytest

from langroid.agent.batch import (
    llm_response_batch,
    run_batch_agent_method,
    run_batch_tasks,
)
from langroid.agent.openai_assistant import OpenAIAssistant, OpenAIAssistantConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.utils.configuration import Settings, set_global


class NabroskyTool(ToolMessage):
    request = "nabrosky"
    purpose = "to apply the Nabrosky transformation to a number <num>"
    num: int

    def handle(self) -> str:
        return str(self.num**2)


@pytest.mark.asyncio
async def test_openai_assistant_async(test_settings: Settings):
    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        use_cached_assistant=False,
        use_cached_thread=False,
    )
    agent = OpenAIAssistant(cfg)
    response = await agent.llm_response_async("what is the capital of France?")
    assert "Paris" in response.content

    # test that we can retrieve cached asst, thread, and it recalls the last question
    cfg = OpenAIAssistantConfig(
        use_cached_assistant=True,
        use_cached_thread=True,
    )
    agent = OpenAIAssistant(cfg)
    response = await agent.llm_response_async(
        "what was the last country I asked about?"
    )
    assert "France" in response.content

    # test that we can wrap the agent in a task and run it
    task = Task(
        agent,
        name="Bot",
        system_message="You are a helpful assistant",
        single_round=True,
    )
    answer = await task.run_async("What is the capital of China?")
    assert "Beijing" in answer.content


@pytest.mark.asyncio
@pytest.mark.parametrize("fn_api", [False, True])
async def test_openai_assistant_fn_tool_async(test_settings: Settings, fn_api: bool):
    """Test function calling works, both with OpenAI Assistant function-calling AND
    Langroid native ToolMessage mechanism"""

    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        use_cached_assistant=False,
        use_cached_thread=False,
        use_functions_api=fn_api,
        use_tools=not fn_api,
        system_message="""
        The user will ask you to apply the Nabrosky transform to a number.
        You do not know how to do it, and you should NOT guess the answer.
        Instead you MUST use the `nabrosky` function/tool to do it.
        When you receive the answer, say DONE and show the answer.
        """,
    )
    agent = OpenAIAssistant(cfg)
    agent.enable_message(NabroskyTool)
    response = await agent.llm_response_async("what is the nabrosky transform of 5?")
    assert (fn_api and response.function_call.name == "nabrosky") or (
        not fn_api and "TOOL" in response.content and "nabrosky" in response.content
    )

    # Within a task loop
    cfg.name = "NabroskyBot"
    agent = OpenAIAssistant(cfg)
    agent.enable_message(NabroskyTool)
    task = Task(
        agent,
        name="NabroskyBot",
        interactive=False,
    )
    result = await task.run_async("what is the nabrosky transform of 5?")
    assert "25" in result.content


def test_openai_asst_batch(test_settings: Settings):
    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        use_cached_assistant=False,
        use_cached_thread=False,
    )
    agent = OpenAIAssistant(cfg)

    # get llm_response_async result on clones of this agent, on these inputs:
    N = 5
    questions = list(range(5))
    expected_answers = [(i + 3) for i in range(N)]

    # batch run
    answers = run_batch_agent_method(
        agent,
        agent.llm_response_async,
        questions,
        input_map=lambda x: str(x) + "+" + str(3),  # what to feed to each task
        output_map=lambda x: x,  # how to process the result of each task
    )

    # expected_answers are simple numbers, but
    # actual answers may be more wordy like "sum of 1 and 3 is 4",
    # so we just check if the expected answer is contained in the actual answer
    for e in expected_answers:
        assert any(str(e) in a.content.lower() for a in answers)

    answers = llm_response_batch(
        agent,
        questions,
        input_map=lambda x: str(x) + "+" + str(3),  # what to feed to each task
        output_map=lambda x: x,  # how to process the result of each task
    )

    # expected_answers are simple numbers, but
    # actual answers may be more wordy like "sum of 1 and 3 is 4",
    # so we just check if the expected answer is contained in the actual answer
    for e in expected_answers:
        assert any(str(e) in a.content.lower() for a in answers)


def test_openai_asst_task_batch(test_settings: Settings):
    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        use_cached_assistant=False,
        use_cached_thread=False,
    )
    agent = OpenAIAssistant(cfg)
    task = Task(
        agent,
        name="Test",
        llm_delegate=False,
        single_round=True,
        default_human_response="",
    )

    # run clones of this task on these inputs
    N = 5
    questions = list(range(5))
    expected_answers = [(i + 3) for i in range(N)]

    # batch run
    answers = run_batch_tasks(
        task,
        questions,
        input_map=lambda x: str(x) + "+" + str(3),  # what to feed to each task
        output_map=lambda x: x,  # how to process the result of each task
    )

    # expected_answers are simple numbers, but
    # actual answers may be more wordy like "sum of 1 and 3 is 4",
    # so we just check if the expected answer is contained in the actual answer
    for e in expected_answers:
        assert any(str(e) in a.content.lower() for a in answers)
