import asyncio
import time
from typing import Optional

import pytest

from langroid import ChatDocument
from langroid.agent.batch import (
    ExceptionHandling,
    _convert_exception_handling,
    _process_batch_async,
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
from langroid.utils.configuration import Settings, set_global, settings
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
@pytest.mark.parametrize("return_type", [True, False])
def test_task_batch(
    test_settings: Settings,
    sequential: bool,
    batch_size: Optional[int],
    stop_on_first: bool,
    return_type: bool,
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

    if return_type:
        # specialized to return str
        task = task[str]

    # run clones of this task on these inputs
    N = 3
    questions = list(range(N))
    expected_answers = [(i + 1) for i in range(N)]

    orig_quiet = settings.quiet
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
    assert settings.quiet == orig_quiet

    if stop_on_first:
        # only the task with input 0 succeeds since it's fastest
        non_null_answer = [a for a in answers if a is not None][0]
        assert non_null_answer is not None
        answer = non_null_answer if return_type else non_null_answer.content
        assert answer == str(expected_answers[0])
    else:
        for e in expected_answers:
            if return_type:
                assert any(str(e) in a.lower() for a in answers)
            else:
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


@pytest.mark.parametrize("batch_size", [1, 2, 3, None])
@pytest.mark.parametrize("sequential", [True, False])
@pytest.mark.parametrize("stop_on_first", [True, False])
def test_agent_llm_response_batch(
    test_settings: Settings,
    sequential: bool,
    stop_on_first: bool,
    batch_size: Optional[int],
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
        batch_size=batch_size,
    )

    if stop_on_first:
        # only the task with input 0 succeeds since it's fastest
        non_null_answer = [a for a in answers if a is not None][0]
        assert non_null_answer is not None
        assert non_null_answer.content == str(expected_answers[0])
    else:
        for e in expected_answers:
            assert any(str(e) in a.content.lower() for a in answers)

    # Test the helper function as well
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
@pytest.mark.parametrize(
    "handle_exceptions", [ExceptionHandling.RETURN_EXCEPTION, True, False]
)
@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("fn_api", [False, True])
@pytest.mark.parametrize("use_done_tool", [True, False])
def test_task_gen_batch_exceptions(
    test_settings: Settings,
    fn_api: bool,
    use_done_tool: bool,
    sequential: bool,
    handle_exceptions: bool | ExceptionHandling,
    batch_size: Optional[int],
):
    set_global(test_settings)
    kill_called = []  # Track Task.kill() calls

    class ComputeTool(ToolMessage):
        request: str = "compute"
        purpose: str = "To compute an unknown function of the input"
        input: int

    system_message = """
    You will make a call with the `compute` tool/function with
    `input` the value I provide. 
    """

    class MockTask(Task):
        """Mock Task that raises exceptions for testing"""

        def kill(self):
            kill_called.append(self.name)
            super().kill()

    def task_gen(i: int) -> Task:
        cfg = ChatAgentConfig(
            vecdb=None,
            llm=OpenAIGPTConfig(async_stream_quiet=False),
            use_functions_api=fn_api,
            use_tools=not fn_api,
            use_tools_api=True,
        )
        agent = ChatAgent(cfg)
        agent.enable_message(ComputeTool)
        if use_done_tool:
            agent.enable_message(DoneTool)
        task = MockTask(
            agent,
            name=f"Test-{i}",
            system_message=system_message,
            interactive=False,
        )

        def handle(m: ComputeTool) -> str | DoneTool:
            if i == 1:
                raise RuntimeError("disaster")
            elif i == 2:
                raise asyncio.CancelledError()
            return DoneTool(content="success") if use_done_tool else f"{DONE} success"

        setattr(agent, "compute", handle)
        return task

    questions = list(range(3))

    try:
        answers = run_batch_task_gen(
            task_gen,
            questions,
            sequential=sequential,
            handle_exceptions=handle_exceptions,
            batch_size=batch_size,
        )
        error_encountered = False

        # Test successful case
        assert answers[0] is not None
        assert "success" in answers[0].content.lower()

        # the task that raised CancelledError
        assert kill_called == ["Test-2"]

        # Test RuntimeError case
        if (
            _convert_exception_handling(handle_exceptions)
            == ExceptionHandling.RETURN_EXCEPTION
        ):
            assert isinstance(answers[1], RuntimeError)
            assert "disaster" in str(answers[1])

            assert isinstance(answers[2], asyncio.CancelledError)
        elif (
            _convert_exception_handling(handle_exceptions)
            == ExceptionHandling.RETURN_NONE
        ):
            assert answers[1] is None
            assert answers[2] is None
        else:
            assert False, "Invalid handle_exceptions value"
    except RuntimeError as e:
        error_encountered = True
        assert "disaster" in str(e)
    except asyncio.CancelledError:
        error_encountered = True

    assert error_encountered == (
        _convert_exception_handling(handle_exceptions) == ExceptionHandling.RAISE
    )


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


def test_batch_size_processing(test_settings: Settings):
    """Test that batch_size parameter correctly processes items in batches"""
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)

    N = 5
    questions = list(range(N))
    batch_size = 2

    answers = run_batch_agent_method(
        agent,
        agent.llm_response_async,
        questions,
        input_map=lambda x: str(x),
        output_map=lambda x: x,
        sequential=True,
        batch_size=batch_size,
    )

    # Verify we got all expected answers
    assert len(answers) == N
    for i, answer in enumerate(answers):
        assert answer is not None
        assert str(i + 1) in answer.content


@pytest.mark.parametrize("sequential", [True, False])
@pytest.mark.parametrize(
    "handle_exceptions", [True, False, ExceptionHandling.RETURN_EXCEPTION]
)
def test_process_batch_async_basic(sequential, handle_exceptions):
    """Test the core async batch processing function"""

    async def mock_task(input: str, i: int) -> str:
        if i == 1:  # Make second task fail
            raise ValueError("Task failed")
        await asyncio.sleep(0.1)
        return f"Processed {input}"

    inputs = ["a", "b", "c"]
    coroutine = _process_batch_async(
        inputs,
        mock_task,
        sequential=sequential,
        handle_exceptions=handle_exceptions,
        output_map=lambda x: x,
    )
    # If handle_exceptions is True, the function should return
    # the results of the successful tasks
    orig_quiet = settings.quiet
    if _convert_exception_handling(handle_exceptions) == ExceptionHandling.RETURN_NONE:
        results = asyncio.run(coroutine)
        assert results[1] is None
        assert "Processed" in results[0]
        assert "Processed" in results[2]
        assert settings.quiet == orig_quiet
    # If handle_exceptions is False, the function should raise an error
    elif _convert_exception_handling(handle_exceptions) == ExceptionHandling.RAISE:
        with pytest.raises(ValueError):
            results = asyncio.run(coroutine)
    # If handle_exceptions is RETURN_EXCEPTION, the function should return
    # the results of the successful tasks and the exception of the failed task
    else:
        assert (
            _convert_exception_handling(handle_exceptions)
            == ExceptionHandling.RETURN_EXCEPTION
        )
        results = asyncio.run(coroutine)
        assert settings.quiet == orig_quiet
        assert "Processed" in results[0]
        assert "Processed" in results[2]
        assert isinstance(results[1], ValueError)


@pytest.mark.parametrize("stop_on_first_result", [True, False])
def test_process_batch_async_stop_on_first(stop_on_first_result):
    """Test stop_on_first_result behavior"""

    async def mock_task(input: str, i: int) -> str:
        await asyncio.sleep(0.1 * i)  # Make later tasks slower
        return f"Processed {input}"

    inputs = ["a", "b", "c"]
    results = asyncio.run(
        _process_batch_async(
            inputs,
            mock_task,
            stop_on_first_result=stop_on_first_result,
            sequential=False,
            handle_exceptions=ExceptionHandling.RAISE,
            output_map=lambda x: x,
        )
    )

    # When stop_on_first_result is True, only the first task should complete
    if stop_on_first_result:
        assert any(r is not None for r in results)
        assert any(r is None for r in results)
        # First task should complete first due to sleep timing
        assert results[0] is not None
        assert "Processed a" in results[0]
    # When stop_on_first_result is False, all tasks should complete
    else:
        assert all(r is not None for r in results)
        assert all("Processed" in r for r in results)
