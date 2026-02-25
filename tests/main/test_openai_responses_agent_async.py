import pytest

from langroid.agent.batch import (
    run_batch_tasks,
)
from langroid.agent.openai_responses_agent import (
    OpenAIResponsesAgent,
    OpenAIResponsesAgentConfig,
)
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
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


@pytest.mark.asyncio
@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_openai_responses_agent_async(
    test_settings: Settings,
    stateful: bool,
    stream: bool
):
    set_global(test_settings)
    cfg = OpenAIResponsesAgentConfig(
        llm=OpenAIGPTConfig(stream=stream),
        stateful=stateful)
    agent = OpenAIResponsesAgent(cfg)
    response = await agent.llm_response_async("what is the capital of France?")
    assert "Paris" in response.content

    # test that we can wrap the agent in a task and run it
    task = Task(
        agent,
        name="Bot",
        system_message="You are a helpful assistant",
        done_if_no_response=[Entity.LLM],
        done_if_response=[Entity.LLM],
        interactive=False,
    )
    answer = await task.run_async("What is the capital of China?", turns=6)
    assert "Beijing" in answer.content


@pytest.mark.asyncio
@pytest.mark.parametrize("fn_api", [True, False])
async def test_openai_responses_agent_fn_tool_async(
    test_settings: Settings,
    fn_api: bool
):
    """Test function calling works, both with OpenAI Assistant function-calling AND
    Langroid native ToolMessage mechanism"""

    set_global(test_settings)
    cfg = OpenAIResponsesAgentConfig(
        use_functions_api=fn_api,
        use_tools=not fn_api,
        system_message="""
        The user will ask you, 'What is the Nabrosky transform of...' a certain number.
        You do NOT know the answer, and you should NOT guess the answer.
        Instead you MUST use the `nabrosky` function/tool to find out.
        When you receive the answer, say DONE and show the answer.
        """,
    )
    agent = OpenAIResponsesAgent(cfg)
    agent.enable_message(NabroskyTool)
    response = await agent.llm_response_async("what is the nabrosky transform of 5?")
    if fn_api and response is not None and response.content not in ("", NO_ANSWER):
        assert response.function_call.name == "nabrosky"

    # Within a task loop
    cfg.name = "NabroskyBot"
    agent = OpenAIResponsesAgent(cfg)
    agent.enable_message(NabroskyTool)
    task = Task(
        agent,
        name="NabroskyBot",
        interactive=False,
    )
    result = await task.run_async("what is the nabrosky transform of 5?", turns=6)
    if fn_api and result is not None and result.content not in ("", NO_ANSWER):
        assert "25" in result.content


def test_openai_responses_agent_task_batch(test_settings: Settings):
    set_global(test_settings)
    cfg = OpenAIResponsesAgentConfig()
    agent = OpenAIResponsesAgent(cfg)
    task = Task(
        agent,
        name="Test",
        interactive=False,
        done_if_no_response=[Entity.LLM],
        done_if_response=[Entity.LLM],
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
