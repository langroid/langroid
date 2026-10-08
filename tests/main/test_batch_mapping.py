"""Regression tests for per-item batch output mapping."""

from typing import Any

import pytest

from langroid.agent.batch import ExceptionHandling, run_batch_agent_method
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity


class MappingAgent(ChatAgent):
    async def produce(self, message: str | ChatDocument | None) -> ChatDocument | None:
        assert isinstance(message, str)
        if message == "task_error":
            raise ValueError("task failed")
        if message == "empty":
            return None
        return ChatDocument(
            content=message, metadata=ChatDocMetaData(sender=Entity.AGENT)
        )


@pytest.mark.parametrize("sequential", [True, False])
@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("policy", list(ExceptionHandling))
@pytest.mark.parametrize("failure", ["bad", "task_error"])
def test_batch_mapping_exception_policy(
    sequential: bool,
    batch_size: int | None,
    policy: ExceptionHandling,
    failure: str,
) -> None:
    agent = MappingAgent(ChatAgentConfig(llm=MockLMConfig(), vecdb=None, parsing=None))
    mapped: list[Any] = []

    def output_map(result: ChatDocument | None) -> int:
        mapped.append(result)
        assert isinstance(result, ChatDocument)
        return int(result.content)

    def run() -> list[Any]:
        return run_batch_agent_method(
            agent,
            agent.produce,
            ["1", failure, "3"],
            sequential=sequential,
            batch_size=batch_size,
            handle_exceptions=policy,
            output_map=output_map,
        )

    if policy == ExceptionHandling.RAISE:
        with pytest.raises(ValueError):
            run()
        return

    results = run()
    assert results[0] == 1
    assert results[2] == 3
    if policy == ExceptionHandling.RETURN_NONE:
        assert results[1] is None
    else:
        assert isinstance(results[1], ValueError)
    assert all(isinstance(item, ChatDocument) for item in mapped)


@pytest.mark.parametrize("sequential", [True, False])
@pytest.mark.parametrize("policy", list(ExceptionHandling))
def test_batch_mapping_preserves_successful_none(
    sequential: bool, policy: ExceptionHandling
) -> None:
    agent = MappingAgent(ChatAgentConfig(llm=MockLMConfig(), vecdb=None, parsing=None))
    results = run_batch_agent_method(
        agent,
        agent.produce,
        ["1", "empty", "3"],
        sequential=sequential,
        handle_exceptions=policy,
        output_map=lambda result: -1 if result is None else int(result.content),
    )
    assert results == [1, -1, 3]
