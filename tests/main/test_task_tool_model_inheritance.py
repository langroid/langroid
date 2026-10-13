import threading

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.orchestration import DoneTool
from langroid.agent.tools.task_tool import TaskTool
from langroid.language_models.mock_lm import MockLM, MockLMConfig
from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global


@pytest.fixture
def parent_agent(test_settings: Settings) -> ChatAgent:
    test_settings.chat_model = ""
    test_settings.cache = False
    test_settings.quiet = True
    set_global(test_settings)
    return ChatAgent(
        ChatAgentConfig(
            llm=MockLMConfig(
                default_response=DoneTool(content="delegated result").model_dump_json()
            )
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_task_tool_inherits_parent_model(
    parent_agent: ChatAgent, is_async: bool
) -> None:
    tool = TaskTool(
        system_message="Complete the task", prompt="Delegate", tools=["NONE"]
    )
    sub_task = tool._set_up_task(parent_agent)
    assert isinstance(sub_task.agent.llm, MockLM)

    result = (
        await tool.handle_async(parent_agent) if is_async else tool.handle(parent_agent)
    )
    assert result is not None
    assert result.content == "delegated result"


def test_task_tool_copies_parent_provider_config(parent_agent: ChatAgent) -> None:
    provider_parent = ChatAgent(
        ChatAgentConfig(
            llm=OpenAIGPTConfig(
                chat_model="custom-model",
                api_base="http://localhost:8000/v1",
                api_key="test-key",
                temperature=0.7,
            )
        )
    )
    tool = TaskTool(
        system_message="Complete the task", prompt="Delegate", tools=["NONE"]
    )
    child = tool._set_up_task(provider_parent).agent
    assert isinstance(child.llm, OpenAIGPT)
    assert isinstance(provider_parent.llm, OpenAIGPT)
    assert child.llm.config.chat_model == provider_parent.llm.config.chat_model
    assert child.llm.config.api_base == provider_parent.llm.config.api_base
    assert child.llm.config.api_key == provider_parent.llm.config.api_key
    assert child.llm.config.temperature == 0.7
    child.llm.config.temperature = 0.1
    assert provider_parent.llm.config.temperature == 0.7


class _LockedKeyProvider:
    """A rotating-credential provider of the shape the docs invite.

    `docs/notes/rotating-api-keys.md` tells users the provider is responsible
    for caching and refreshing its own token, which in practice means a
    mutex-protected object. A `threading.Lock` cannot be deep-copied, so this
    is the smallest faithful stand-in for a real one.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def __call__(self) -> str:
        with self.lock:
            return "rotated-key"


def test_task_tool_inherits_uncopyable_api_key_provider() -> None:
    """A parent whose key provider holds a lock must not break delegation.

    `OpenAIGPT.__init__` deliberately keeps `api_key_provider` out of its deep
    copy, because deep-copying it both raises on non-copyable state and
    defeats the client cache, which is keyed on the callable's identity. The
    inheritance path has to do the same, or a parent that constructs fine
    cannot delegate at all.
    """
    provider = _LockedKeyProvider()
    parent = ChatAgent(
        ChatAgentConfig(
            llm=OpenAIGPTConfig(
                chat_model="custom-model",
                api_base="http://localhost:8000/v1",
                api_key_provider=provider,
            )
        )
    )
    tool = TaskTool(
        system_message="Complete the task", prompt="Delegate", tools=["NONE"]
    )

    child = tool._set_up_task(parent).agent

    assert isinstance(child.llm, OpenAIGPT)
    assert isinstance(parent.llm, OpenAIGPT)
    # Identity, not equality: a copied provider is a different cache key, so
    # every sub-agent would build its own client.
    assert child.llm.config.api_key_provider is provider
    assert parent.llm.config.api_key_provider is provider
    assert child.llm.config.api_base == "http://localhost:8000/v1"


def test_task_tool_explicit_model_overrides_parent(parent_agent: ChatAgent) -> None:
    tool = TaskTool(
        system_message="Complete the task",
        prompt="Delegate",
        tools=["NONE"],
        model="gpt-4.1-mini",
    )
    child = tool._set_up_task(parent_agent).agent
    assert isinstance(child.llm, OpenAIGPT)
    assert child.llm.config.chat_model == "gpt-4.1-mini"
