import asyncio
import json
import time
from typing import Optional

import pytest

from langroid.agent.batch import run_batch_task_gen
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.special.doc_chat_agent import apply_nest_asyncio
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import DoneTool
from langroid.language_models.mock_lm import MockLMConfig
from langroid.utils.constants import DONE

apply_nest_asyncio()


def echo_response(x: str) -> str:
    return x


async def echo_response_async(x: str) -> str:
    return x


class _TestAsyncToolHandlerConfig(ChatAgentConfig):
    llm: MockLMConfig = MockLMConfig(
        response_dict={
            "sleep 1": 'TOOL sleep: {"seconds": "0"}',
            "sleep 2": 'TOOL sleep: {"seconds": "1"}',
            "sleep 3": 'TOOL sleep: {"seconds": "2"}',
            "sleep 4": 'TOOL sleep: {"seconds": "3"}',
            "sleep 5": 'TOOL sleep: {"seconds": "4"}',
        },
    )


async def scheduler(events: dict[int, asyncio.Event], done_event: asyncio.Event):
    """
    Implicitly forces sequential scheduling (in the order of the keys
    of `events`) via asyncio Events. Each scheduled task must first wait
    on its corresponding Event and signal completion by setting `done_event`
    on completion.
    """
    turns = list(sorted(list(events.items()), key=lambda item: item[0]))

    for _, wait in turns:
        wait.set()
        await done_event.wait()
        # Allow the task which signaled completion to exit
        await asyncio.sleep(0.01)
        done_event.clear()


@pytest.mark.parametrize("stop_on_first", [False, True])
@pytest.mark.asyncio
async def test_async_tool_handler(
    stop_on_first: bool,
):
    """
    Test that async tool handlers are working.

    Define an agent with a "sleep" tool that sleeps for specified number
    of seconds. Implement both sync and async handler for this tool.
    Create a batch of 5 tasks that run the "sleep" tool with decreasing
    sleep times: 4, 3, 2, 1, 0 seconds. Sleep is simulated by scheduling
    the tasks from shortest to longest sleep times.
    Run these tasks in parallel and ensure that:
     * async handler is called for all tasks
     * tasks actually sleep
     * tasks finish in the expected order (reverse from the start order)
    """

    class SleepTool(ToolMessage):
        request: str = "sleep"
        purpose: str = "To sleep for specified number of seconds"
        seconds: int

    done_event = asyncio.Event()
    wait_events = {i: asyncio.Event() for i in [0, 1, 2, 3, 4]}

    def task_gen(i: int) -> Task:
        # create a mock agent that calls "sleep" tool
        cfg = _TestAsyncToolHandlerConfig()
        agent = ChatAgent(cfg)
        agent.enable_message(SleepTool)
        agent.enable_message(DoneTool)

        # sync tool handler
        def handle(m: SleepTool) -> str | DoneTool:
            response = {
                "handler": "sync",
                "seconds": m.seconds,
            }
            if m.seconds > 0:
                time.sleep(m.seconds)
            response["end"] = time.perf_counter()
            return DoneTool(content=json.dumps(response))

        setattr(agent, "sleep", handle)

        # async tool handler
        async def handle_async(m: SleepTool) -> str | DoneTool:
            response = {
                "handler": "async",
                "seconds": m.seconds,
            }
            await wait_events[m.seconds].wait()

            response["end"] = time.perf_counter()

            done_event.set()
            return DoneTool(content=json.dumps(response))

        setattr(agent, "sleep_async", handle_async)

        # create a task that runs this agent
        task = Task(agent, name=f"Test-{i}", interactive=False)
        return task

    # run clones of this task on these inputs
    N = 5
    questions = [f"sleep {str(N - x)}" for x in range(N)]

    # Start executing the scheduler
    scheduler_task = asyncio.create_task(scheduler(wait_events, done_event))

    # batch run
    answers = run_batch_task_gen(
        task_gen,
        questions,
        sequential=False,
        stop_on_first_result=stop_on_first,
    )
    scheduler_task.cancel()

    for a in answers:
        if a is not None:
            d = json.loads(a.content)
            # ensure that async handler was called
            assert d["handler"] == "async"

    if stop_on_first:
        # only the last task (which doesn't sleep) should succeed
        non_null_answers = [a for a in answers if a is not None]
        assert len(non_null_answers) == 1
        d = json.loads(non_null_answers[0].content)
        assert d["seconds"] == 0
    else:
        # tasks should end in reverse order
        assert all(a is not None for a in answers)
        ends = [json.loads(a.content)["end"] for a in answers]
        assert ends == sorted(ends, reverse=True)
        seconds = [json.loads(a.content)["seconds"] for a in answers]
        assert seconds == sorted(seconds, reverse=True)


class _TestAsyncUserResponseConfig(ChatAgentConfig):
    llm: MockLMConfig = MockLMConfig(
        response_fn=echo_response, response_fn_async=echo_response_async
    )


async def get_user_response_async(prompt: str) -> str:
    return "async response"


def get_user_response(prompt: str) -> str:
    return "sync response"


@pytest.mark.asyncio
async def test_async_user_response():
    """
    Test that async human response callbacks are called by `user_response_asnyc`
    when available, falling back to sync callbacks.
    """
    cfg = _TestAsyncUserResponseConfig()

    agent = ChatAgent(cfg)
    agent.callbacks.get_user_response = get_user_response

    # `user_response_async()` should call the sync callback
    # if it is the only one available
    response = await agent.user_response_async()
    assert response is not None
    assert response.content == "sync response"

    agent.callbacks.get_user_response_async = get_user_response_async

    # `user_response()` should always call the sync callback
    response = agent.user_response()
    assert response is not None
    assert response.content == "sync response"

    # `user_response_async()` should call the sync callback if available
    response = await agent.user_response_async()
    assert response is not None
    assert response.content == "async response"


@pytest.mark.skip(reason="Flaky test, needs adjustment?")
@pytest.mark.parametrize("stop_on_first", [True, False])
@pytest.mark.asyncio
async def test_async_user_response_batch(
    stop_on_first: bool,
):
    """
    Test that there is no blocking in async human response callbacks.
    Similar to test_async_tool_handler.
    """
    # Number of tasks
    N = 5

    done_event = asyncio.Event()
    wait_events = {i: asyncio.Event() for i in [0, 1, 2, 3, 4]}

    def task_gen(i: int) -> Task:
        # reverse order
        wait = N - i - 1
        cfg = _TestAsyncUserResponseConfig()
        agent = ChatAgent(cfg)

        async def get_user_response_async(prompt: str) -> str:
            await wait_events[wait].wait()
            end_time = time.time()
            done_event.set()
            return f"{DONE} async response {end_time} {i}"

        agent.callbacks.get_user_response = get_user_response
        agent.callbacks.get_user_response_async = get_user_response_async

        # create a task that runs this agent
        task = Task(
            agent,
            name=f"Test-{i}",
        )
        return task

    # run clones of this task on these inputs
    questions = [str(i) for i in range(N)]

    # Start executing the scheduler
    scheduler_task = asyncio.create_task(scheduler(wait_events, done_event))

    # batch run
    answers = run_batch_task_gen(
        task_gen,
        questions,
        sequential=False,
        stop_on_first_result=stop_on_first,
    )
    scheduler_task.cancel()

    for a in answers:
        if a is not None:
            # ensure that async handler was called
            assert "async" in a.content

    if stop_on_first:
        # only the last task (which doesn't sleep) should succeed
        non_null_answers = [a for a in answers if a is not None]
        assert len(non_null_answers) == 1
        assert "0" in non_null_answers[0].content
    else:
        # tasks should end in reverse order
        def get_task_result(answer: Optional[ChatDocument]) -> tuple[int, float]:
            assert answer is not None
            end_time, id = answer.content.split()[-2:]
            id = int(id)
            end_time = float(end_time)

            return id, end_time

        order = [
            result[0]
            for result in sorted(
                [get_task_result(a) for a in answers],
                key=lambda result: result[1],
            )
        ]
        assert order == list(reversed(range(N)))
