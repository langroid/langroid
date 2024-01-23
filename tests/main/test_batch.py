import pytest

from langroid.agent.batch import (
    llm_response_batch,
    run_batch_agent_method,
    run_batch_tasks,
)
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.utils.configuration import Settings, set_global
from langroid.vector_store.base import VectorStoreConfig


class _TestChatAgentConfig(ChatAgentConfig):
    vecdb: VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        cache_config=RedisCacheConfig(fake=False),
        chat_model=OpenAIChatModel.GPT4,
        use_chat_for_completion=True,
    )


@pytest.mark.parametrize("sequential", [True, False])
def test_task_batch(test_settings: Settings, sequential: bool):
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
    expected_answers = [(i + 3) for i in range(N)]

    # batch run
    answers = run_batch_tasks(
        task,
        questions,
        input_map=lambda x: str(x) + "+" + str(3),  # what to feed to each task
        output_map=lambda x: x,  # how to process the result of each task
        sequential=sequential,
    )

    # expected_answers are simple numbers, but
    # actual answers may be more wordy like "sum of 1 and 3 is 4",
    # so we just check if the expected answer is contained in the actual answer
    for e in expected_answers:
        assert any(str(e) in a.content.lower() for a in answers)


@pytest.mark.parametrize("sequential", [True, False])
def test_agent_llm_response_batch(test_settings: Settings, sequential: bool):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()

    agent = ChatAgent(cfg)

    # get llm_response_async result on clones of this agent, on these inputs:
    N = 3
    questions = list(range(N))
    expected_answers = [(i + 3) for i in range(N)]

    # batch run
    answers = run_batch_agent_method(
        agent,
        agent.llm_response_async,
        questions,
        input_map=lambda x: str(x) + "+" + str(3),  # what to feed to each task
        output_map=lambda x: x,  # how to process the result of each task
        sequential=sequential,
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
        sequential=sequential,
    )

    # expected_answers are simple numbers, but
    # actual answers may be more wordy like "sum of 1 and 3 is 4",
    # so we just check if the expected answer is contained in the actual answer
    for e in expected_answers:
        assert any(str(e) in a.content.lower() for a in answers)
