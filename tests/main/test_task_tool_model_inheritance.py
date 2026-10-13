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
