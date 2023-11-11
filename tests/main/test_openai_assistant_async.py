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


class SquareTool(ToolMessage):
    request = "square"
    purpose = "to find the square of a number"
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
async def test_openai_assistant_fn_tool_async(test_settings: Settings):
    """Test function calling"""

    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        use_cached_assistant=False,
        use_cached_thread=False,
        use_functions_api=True,
        system_message="""
        The user will give you a number to square. 
        Use the `square` function to square it.
        When you receive the answer, say DONE.
        """,
    )
    agent = OpenAIAssistant(cfg)
    agent.enable_message(SquareTool)
    response = await agent.llm_response_async("what is the square of 5?")
    assert response.function_call.name == "square"

    # Within a task loop
    cfg.name = "SquaringBot"
    agent = OpenAIAssistant(cfg)
    agent.enable_message(SquareTool)
    task = Task(
        agent,
        name="SquaringBot",
        interactive=False,
    )
    result = await task.run_async("what is the square of 5?")
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
