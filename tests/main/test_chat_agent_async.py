import asyncio
from typing import Optional

import pytest

from langroid.agent.base import NO_ANSWER
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.utils.configuration import Settings, set_global
from langroid.vector_store.base import VectorStoreConfig


class _TestChatAgentConfig(ChatAgentConfig):
    vecdb: Optional[VectorStoreConfig] = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        cache_config=RedisCacheConfig(fake=False),
        use_chat_for_completion=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_quiet", [True, False])
async def test_chat_agent_async(test_settings: Settings, stream_quiet: bool):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    cfg.llm.async_stream_quiet = stream_quiet
    # just testing that these don't fail
    agent = ChatAgent(cfg)
    response = await agent.llm_response_async("what is the capital of France?")
    assert "Paris" in response.content


@pytest.mark.asyncio
async def test_responses_async(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)

    # direct LLM response to query
    response = await agent.llm_response_async("what is the capital of France?")
    assert "Paris" in response.content

    # human is prompted for input, and we specify the default response
    agent.default_human_response = "What about England?"
    response = await agent.user_response_async()
    assert "England" in response.content

    response = await agent.llm_response_async("what about England?")
    assert "London" in response.content

    # agent attempts to handle the query, but has no response since
    # the message is not a structured msg that matches an enabled ToolMessage.
    response = await agent.agent_response_async("What is the capital of France?")
    assert response is None


@pytest.mark.asyncio
async def test_task_step_async(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)
    task = Task(
        agent,
        name="Test",
    )
    msg = "What is the capital of France?"
    task.init(msg)
    assert task.pending_message.content == msg

    # LLM answers
    await task.step_async()
    assert "Paris" in task.pending_message.content
    assert task.pending_message.metadata.sender == Entity.LLM

    agent.default_human_response = "What about England?"
    # User asks about England
    await task.step_async()
    assert "England" in task.pending_message.content
    assert task.pending_message.metadata.sender == Entity.USER

    # LLM answers
    await task.step_async()
    assert "London" in task.pending_message.content
    assert task.pending_message.metadata.sender == Entity.LLM

    # It's Human's turn; they say nothing,
    # and this is reflected in `self.pending_message` as NO_ANSWER
    agent.default_human_response = ""
    # Human says '', which is an invalid response, so pending msg stays same
    await task.step_async()
    assert "London" in task.pending_message.content
    assert task.pending_message.metadata.sender == Entity.LLM

    # LLM cannot respond to itself, so pending msg still does not change
    await task.step_async()
    assert "London" in task.pending_message.content
    assert task.pending_message.metadata.sender == Entity.LLM

    # reset task
    question = "What is my name?"
    task = Task(
        agent,
        name="Test",
        system_message=f""" Your job is to always say "{NO_ANSWER}" """,
        restart=True,
    )
    # LLM responds with NO_ANSWER, which is an invalid msg,
    # which is normally an invalid message, but it is the ONLY explicit message
    # in the step, so is processed as a valid step result, and the pending msg is
    # updated to this message.
    task.init(question)
    await task.step_async()
    assert NO_ANSWER in task.pending_message.content
    assert task.pending_message.metadata.sender == Entity.LLM


@pytest.mark.asyncio
async def test_task(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)
    task = Task(
        agent,
        name="Test",
    )
    question = "What is the capital of France?"
    agent.default_human_response = question

    # run task with null initial message
    await task.run_async(turns=3)

    # 3 Turns:
    # 1. LLM initiates convo saying thanks how can I help (since do_task msg empty)
    # 2. User asks the `default_human_response`: What is the capital of France?
    # 3. LLM responds

    assert task.pending_message.metadata.sender == Entity.LLM
    assert "Paris" in task.pending_message.content

    agent.default_human_response = "What about England?"

    # run task with initial question
    await task.run_async(msg=question, turns=3)

    # 3 Turns:
    # 1. LLM answers question, since task is run with the question
    # 2. User asks the `default_human_response`: What about England?
    # 3. LLM responds

    assert task.pending_message.metadata.sender == Entity.LLM
    assert "London" in task.pending_message.content


@pytest.mark.asyncio
async def test_chat_agent_async_concurrent(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()

    async def _run_task(msg: str):
        # each invocation needs to create its own ChatAgent
        agent = ChatAgent(cfg)
        return await agent.llm_response_async(msg)

    N = 3
    questions = ["1+" + str(i) for i in range(N)]
    expected_answers = [str(i + 1) for i in range(N)]
    answers = await asyncio.gather(*(_run_task(msg=question) for question in questions))
    assert len(answers) == len(questions)
    for e in expected_answers:
        assert any(e in a.content for a in answers)


@pytest.mark.asyncio
async def test_task_concurrent(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()

    async def _run_task(msg: str):
        # each invocation needs to create its own ChatAgent,
        # else the states gets mangled by concurrent calls!
        agent = ChatAgent(cfg)
        task = Task(
            agent,
            name="Test",
            interactive=False,
            done_if_response=[Entity.LLM],
            default_human_response="",
        )
        return await task.run_async(msg=msg)

    N = 5
    questions = ["1+" + str(i) for i in range(N)]
    expected_answers = [str(i + 1) for i in range(N)]

    # concurrent async calls to all tasks
    answers = await asyncio.gather(*(_run_task(msg=question) for question in questions))

    assert len(answers) == len(questions)

    for e in expected_answers:
        assert any(e.lower() in a.content.lower() for a in answers)
