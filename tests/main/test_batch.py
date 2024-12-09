import asyncio
import time
from typing import Optional

import pytest

from langroid import ChatDocument
from langroid.agent.batch import (
    llm_response_batch,
    run_batch_agent_method,
    run_batch_function,
    run_batch_task_gen,
    run_batch_tasks,
)
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import DoneTool
from langroid.language_models.mock_lm import MockLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import DONE
from langroid.vector_store.base import VectorStoreConfig


def process_int(x: str) -> str:
    if int(x) == 0:
        return str(int(x) + 1)
    else:
        time.sleep(2)
        return str(int(x) + 1)


class _TestChatAgentConfig(ChatAgentConfig):
    vecdb: VectorStoreConfig = None
    llm = MockLMConfig(response_fn=lambda x: process_int(x))


@pytest.mark.parametrize("batch_size", [1, 2, 3, None])
@pytest.mark.parametrize("sequential", [True, False])
@pytest.mark.parametrize("stop_on_first", [True, False])
def test_task_batch(
    test_settings: Settings,
    sequential: bool,
    batch_size: Optional[int],
    stop_on_first: bool,
):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()

    agent = ChatAgent(cfg)
    task = Task(
        agent,
        name="Test",
        interactive=False,
        done_if_response=[Entity.LLM],
        done_if_no_response=[Entity.LLM],
    )

    # run clones of this task on these inputs
    N = 3
    questions = list(range(N))
    expected_answers = [(i + 1) for i in range(N)]

    # batch run
    answers = run_batch_tasks(
        task,
        questions,
        input_map=lambda x: str(x),  # what to feed to each task
        output_map=lambda x: x,  # how to process the result of each task
        sequential=sequential,
        batch_size=batch_size,
        stop_on_first_result=stop_on_first,
    )

    if stop_on_first:
        # only the task with input 0 succeeds since it's fastest
        non_null_answer = [a for a in answers if a is not None][0]
        assert non_null_answer is not None
        assert non_null_answer.content == str(expected_answers[0])
    else:
        for e in expected_answers:
            assert any(str(e) in a.content.lower() for a in answers)


@pytest.mark.parametrize("batch_size", [1, 2, 3, None])
@pytest.mark.parametrize("sequential", [True, False])
@pytest.mark.parametrize("use_done_tool", [True, False])
def test_task_batch_turns(
    test_settings: Settings,
    sequential: bool,
    batch_size: Optional[int],
    use_done_tool: bool,
):
    """Test if `turns`, `max_cost`, `max_tokens` params work as expected.
    The latter two are not really tested (since we need to turn off caching etc)
    we just make sure they don't break anything.
    """
    set_global(test_settings)
    cfg = _TestChatAgentConfig()

    class _TestChatAgent(ChatAgent):
        def handle_message_fallback(
            self, msg: str | ChatDocument
        ) -> str | DoneTool | None:

            if isinstance(msg, ChatDocument) and msg.metadata.sender == Entity.LLM:
                return (
                    DoneTool(content=str(msg.content))
                    if use_done_tool
                    else DONE + " " + str(msg.content)
                )

    agent = _TestChatAgent(cfg)
    agent.llm.reset_usage_cost()
    task = Task(
        agent,
        name="Test",
        interactive=False,
    )

    # run clones of this task on these inputs
    N = 3
    questions = list(range(N))
    expected_answers = [(i + 1) for i in range(N)]

    # batch run
    answers = run_batch_tasks(
        task,
        questions,
        input_map=lambda x: str(x),  # what to feed to each task
        output_map=lambda x: x,  # how to process the result of each task
        sequential=sequential,
        batch_size=batch_size,
        turns=2,
        max_cost=0.005,
        max_tokens=100,
    )

    # expected_answers are simple numbers, but
    # actual answers may be more wordy like "sum of 1 and 3 is 4",
    # so we just check if the expected answer is contained in the actual answer
    for e in expected_answers:
        assert any(str(e) in a.content.lower() for a in answers)


@pytest.mark.parametrize("sequential", [True, False])
@pytest.mark.parametrize("stop_on_first", [True, False])
def test_agent_llm_response_batch(
    test_settings: Settings,
    sequential: bool,
    stop_on_first: bool,
):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()

    agent = ChatAgent(cfg)

    # get llm_response_async result on clones of this agent, on these inputs:
    N = 3
    questions = list(range(N))
    expected_answers = [(i + 1) for i in range(N)]

    # batch run
    answers = run_batch_agent_method(
        agent,
        agent.llm_response_async,
        questions,
        input_map=lambda x: str(x),  # what to feed to each task
        output_map=lambda x: x,  # how to process the result of each task
        sequential=sequential,
        stop_on_first_result=stop_on_first,
    )

    if stop_on_first:
        # only the task with input 0 succeeds since it's fastest
        non_null_answer = [a for a in answers if a is not None][0]
        assert non_null_answer is not None
        assert non_null_answer.content == str(expected_answers[0])
    else:
        for e in expected_answers:
            assert any(str(e) in a.content.lower() for a in answers)

    answers = llm_response_batch(
        agent,
        questions,
        input_map=lambda x: str(x),  # what to feed to each task
        output_map=lambda x: x,  # how to process the result of each task
        sequential=sequential,
        stop_on_first_result=stop_on_first,
    )

    if stop_on_first:
        # only the task with input 0 succeeds since it's fastest
        non_null_answer = [a for a in answers if a is not None][0]
        assert non_null_answer is not None
        assert non_null_answer.content == str(expected_answers[0])
    else:
        for e in expected_answers:
            assert any(str(e) in a.content.lower() for a in answers)


@pytest.mark.parametrize("stop_on_first", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2, 3, None])
@pytest.mark.parametrize("sequential", [True, False])
def test_task_gen_batch(
    test_settings: Settings,
    sequential: bool,
    stop_on_first: bool,
    batch_size: Optional[int],
):
    set_global(test_settings)

    def task_gen(i: int) -> Task:
        async def response_fn_async(x):
            match i:
                case 0:
                    await asyncio.sleep(0.1)
                    return str(x)
                case 1:
                    return "hmm"
                case _:
                    await asyncio.sleep(0.2)
                    return str(2 * int(x))

        class _TestChatAgentConfig(ChatAgentConfig):
            vecdb: VectorStoreConfig = None
            llm = MockLMConfig(response_fn_async=response_fn_async)

        cfg = _TestChatAgentConfig()
        return Task(
            ChatAgent(cfg),
            name=f"Test-{i}",
            single_round=True,
        )

    # run the generated tasks on these inputs
    questions = list(range(3))
    expected_answers = ["0", "hmm", "4"]

    # batch run
    answers = run_batch_task_gen(
        task_gen,
        questions,
        sequential=sequential,
        stop_on_first_result=stop_on_first,
        batch_size=batch_size,
    )

    if stop_on_first:
        non_null_answer = [a for a in answers if a is not None][0].content

        # Unless the first task is scheduled alone,
        # the second task should always finish first
        if batch_size == 1:
            assert "0" in non_null_answer
        else:
            assert "hmm" in non_null_answer
    else:
        for answer, expected in zip(answers, expected_answers):
            assert answer is not None
            assert expected in answer.content.lower()


@pytest.mark.parametrize("batch_size", [None, 1, 2, 3])
@pytest.mark.parametrize("handle_exceptions", [False, True])
@pytest.mark.parametrize("sequential", [True, False])
@pytest.mark.parametrize("fn_api", [False, True])
@pytest.mark.parametrize("tools_api", [False, True])
@pytest.mark.parametrize("use_done_tool", [True, False])
def test_task_gen_batch_exceptions(
    test_settings: Settings,
    fn_api: bool,
    tools_api: bool,
    use_done_tool: bool,
    sequential: bool,
    handle_exceptions: bool,
    batch_size: Optional[int],
):
    set_global(test_settings)

    class ComputeTool(ToolMessage):
        request: str = "compute"
        purpose: str = "To compute an unknown function of the input"
        input: int

    system_message = """
    You will make a call with the `compute` tool/function with
    `input` the value I provide. 
    """

    def task_gen(i: int) -> Task:
        cfg = ChatAgentConfig(
            vecdb=None,
            llm=OpenAIGPTConfig(),
            use_functions_api=fn_api,
            use_tools=not fn_api,
            use_tools_api=tools_api,
        )
        agent = ChatAgent(cfg)
        agent.enable_message(ComputeTool)
        if use_done_tool:
            agent.enable_message(DoneTool)
        task = Task(
            agent,
            name=f"Test-{i}",
            system_message=system_message,
            interactive=False,
        )

        def handle(m: ComputeTool) -> str | DoneTool:
            if i != 1:
                return (
                    DoneTool(content="success") if use_done_tool else f"{DONE} success"
                )
            else:
                raise RuntimeError("disaster")

        setattr(agent, "compute", handle)
        return task

    # run the generated tasks on these inputs
    questions = list(range(3))

    # batch run
    try:
        answers = run_batch_task_gen(
            task_gen,
            questions,
            sequential=sequential,
            handle_exceptions=handle_exceptions,
            batch_size=batch_size,
        )
        error_encountered = False

        for i in [0, 2]:
            a = answers[i]
            assert a is not None
            assert "success" in a.content.lower()

        assert answers[1] is None
    except RuntimeError as e:
        error_encountered = True
        assert "disaster" in str(e)

    assert error_encountered != handle_exceptions


@pytest.mark.parametrize(
    "func, input_list, batch_size, expected",
    [
        (lambda x: x * 2, [1, 2, 3], None, [2, 4, 6]),
        (lambda x: x + 1, [1, 2, 3, 4], 2, [2, 3, 4, 5]),
        (lambda x: x * x, [], None, []),
        (lambda x: x * 3, [1, 2], 1, [3, 6]),
    ],
)
def test_run_batch_function(func, input_list, batch_size, expected):
    result = run_batch_function(func, input_list, batch_size=batch_size)
    assert result == expected
