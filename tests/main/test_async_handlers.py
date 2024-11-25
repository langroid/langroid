import asyncio
import json
import time

import pytest

from langroid.agent.batch import run_batch_task_gen
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import DoneTool
from langroid.language_models.mock_lm import MockLMConfig
from langroid.utils.configuration import Settings, set_global


def echo_response(x: str) -> str:
    return x


async def echo_response_async(x: str) -> str:
    await asyncio.sleep(.1)
    return x


class _TestAsyncToolHandlerConfig(ChatAgentConfig):
    llm = MockLMConfig(
        response_dict={
            "sleep 1": 'TOOL: sleep {"seconds": 1}',
            "sleep 2": 'TOOL: sleep {"seconds": 2}',
            "sleep 3": 'TOOL: sleep {"seconds": 3}',
            "sleep 4": 'TOOL: sleep {"seconds": 4}',
            "sleep 5": 'TOOL: sleep {"seconds": 5}'
        },
        response_fn=echo_response,
        response_fn_async=echo_response_async
    )


@pytest.mark.parametrize("stop_on_first", [True, False])
def test_async_tool_handler(
    test_settings: Settings,
    stop_on_first: bool,
):
    """
    Test that async tool handlers are working.

    Define an agent with a "sleep" tool that sleeps for specified number
    of seconds. Implement both sync and async handler for this tool.
    Create a batch of 5 tasks that run the "sleep" tool with decreasing
    sleep times: 5, 4, 3, 2, 1 seconds.
    Run these tasks in parallel and ensure that:
     * async handler is called for all tasks
     * tasks actually sleep
     * tasks finish in the expected order (reverse from the start order)
    """
    set_global(test_settings)

    class SleepTool(ToolMessage):
        request: str = "sleep"
        purpose: str = "To sleep for specified number of seconds"
        seconds: int

    def task_gen(i: int) -> Task:
        # create a mock agent that calls "sleep" tool
        cfg = _TestAsyncToolHandlerConfig()
        agent = ChatAgent(cfg)
        agent.enable_message(SleepTool)
        agent.enable_message(DoneTool)

        # sync tool handler
        def handle(m: SleepTool) -> str | DoneTool:
            response = {
                'handler': 'sync',
                'seconds': m.seconds,
                'start': time.perf_counter()
            }
            if m.seconds > 0:
                time.sleep(m.seconds)
            response['end'] = time.perf_counter()
            return DoneTool(content=json.dumps(response))

        setattr(agent, "sleep", handle)

        # async tool handler
        async def handle_async(m: SleepTool) -> str | DoneTool:
            response = {
                'handler': 'async',
                'seconds': m.seconds,
                'start': time.perf_counter()
            }
            if m.seconds > 0:
                await asyncio.sleep(m.seconds)
            response['end'] = time.perf_counter()
            return DoneTool(content=json.dumps(response))

        setattr(agent, "sleep_async", handle_async)

        # create a task that runs this agent
        task = Task(
            agent,
            name=f"Test-{i}",
            interactive=False
        )
        return task

    # run clones of this task on these inputs
    N = 5
    questions = [f"sleep {str(N - x)}" for x in range(N)]

    # batch run
    answers = run_batch_task_gen(
        task_gen,
        questions,
        sequential=False,
        stop_on_first_result=stop_on_first,
    )

    for a in answers:
        if a is not None:
            d = json.loads(a.content)
            # ensure that async handler was called
            assert(d["handler"] == "async")
            # ensure tasks slept for some time
            assert(d["end"] > d["start"])

    if stop_on_first:
        # only the last task (that slept for 1 sec) should succeed
        non_null_answers = [a for a in answers if a is not None]
        assert len(non_null_answers) == 1
        d = json.loads(non_null_answers[0].content)
        assert d["seconds"] == 1
    else:
        # tasks should end in reverse order
        for i, a in enumerate(answers):
            assert a is not None
            if i > 0:
                d = json.loads(a.content)
                d_prev = json.loads(answers[i - 1].content)
                assert d_prev["end"] > d["end"]


class _TestAsyncUserResponseConfig(ChatAgentConfig):
    llm = MockLMConfig(
        response_fn=echo_response,
        response_fn_async=echo_response_async
    )


async def get_user_response_async(prompt: str) -> str:
    await asyncio.sleep(.1)
    return "async response"


def get_user_response(prompt: str) -> str:
    return "sync response"


def test_async_user_response(
    test_settings: Settings,
):
    """
    Test that async human response callbacks are working.
    """
    set_global(test_settings)
    cfg = _TestAsyncUserResponseConfig()

    agent = ChatAgent(cfg)
    agent.callbacks.get_user_response = get_user_response

    task = Task(
        agent,
        name="Test"
    )

    # task.run_async() should call sync callback if it is the only one available
    response = asyncio.run(task.run_async("hi", turns=2))
    assert response.content == "sync response"

    agent.callbacks.get_user_response_async = get_user_response_async

    # task.run() should always call sync callback
    response = task.run("hi", turns=2)
    assert response.content == "sync response"

    # task.run_async() should always call async callback if it is available
    response = asyncio.run(task.run_async("hi", turns=2))
    assert response.content == "async response"
