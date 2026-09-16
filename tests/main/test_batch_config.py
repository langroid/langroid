"""Regression tests for isolated configurations of batch agent copies."""

import asyncio
import json

import pytest

from langroid.agent.base import Agent, AgentConfig
from langroid.agent.batch import run_batch_agent_method
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity


class RecordingConfig(AgentConfig):
    seen_inputs: list[str] = []


class RecordingAgent(Agent):
    async def record(self, message: str | ChatDocument | None) -> ChatDocument:
        assert isinstance(message, str)
        assert isinstance(self.config, RecordingConfig)
        self.config.seen_inputs.append(message)
        await asyncio.sleep(0)
        return ChatDocument(
            content=json.dumps(self.config.seen_inputs),
            metadata=ChatDocMetaData(sender=Entity.AGENT, sender_name=self.config.name),
        )


@pytest.mark.parametrize("sequential", [True, False])
@pytest.mark.parametrize("batch_size", [None, 1, 2, 3])
def test_batch_agent_config_isolation(sequential: bool, batch_size: int | None) -> None:
    config = RecordingConfig(
        name="Worker", llm=MockLMConfig(), vecdb=None, parsing=None
    )
    agent = RecordingAgent(config)
    results = run_batch_agent_method(
        agent,
        agent.record,
        ["a", "b", "c"],
        sequential=sequential,
        batch_size=batch_size,
    )
    assert [r.metadata.sender_name for r in results] == [
        "Worker-0",
        "Worker-1",
        "Worker-2",
    ]
    assert [json.loads(r.content) for r in results] == [["a"], ["b"], ["c"]]
    assert config.name == "Worker"
    assert config.seen_inputs == []
    assert config.llm is not None and config.llm.stream is True
