"""
NOTE: running this test requires setting the GOOGLE_API_KEY and GOOGLE_CSE_ID
environment variables in your `.env` file, as explained in the
[README](https://github.com/langroid/langroid#gear-installation-and-setup).

"""

import pytest

import langroid as lr
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.duckduckgo_search_tool import DuckduckgoSearchTool
from langroid.agent.tools.exa_search_tool import ExaSearchTool
from langroid.agent.tools.google_search_tool import GoogleSearchTool
from langroid.agent.tools.tavily_search_tool import TavilySearchTool
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global

cfg = ChatAgentConfig(
    name="test-langroid",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        cache_config=RedisCacheConfig(fake=False),
    ),
    parsing=ParsingConfig(),
    prompts=PromptsConfig(),
    use_functions_api=False,
    use_tools=True,
)
agent = ChatAgent(cfg)


@pytest.mark.parametrize(
    "search_tool_cls",
    [ExaSearchTool, TavilySearchTool, GoogleSearchTool, DuckduckgoSearchTool],
)
@pytest.mark.parametrize("use_functions_api", [True, False])
@pytest.mark.parametrize("use_tools_api", [True, False])
def test_agent_web_search_tool(
    test_settings: Settings,
    search_tool_cls: lr.ToolMessage,
    use_functions_api: bool,
    use_tools_api: bool,
):
    """
    Test whether LLM is able to GENERATE message (tool) in required format, AND the
    agent handles the message correctly.
    Args:
        test_settings: test settings from conftest.py
        use_functions_api: whether to use LLM's functions api or not
            (i.e. use the langroid ToolMessage tools instead).
    """
    set_global(test_settings)
    agent = ChatAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api
    agent.config.use_tools_api = use_tools_api
    agent.enable_message(search_tool_cls)

    llm_msg = agent.llm_response_forget(
        "Find 3 results on the internet about the LK-99 superconducting material."
    )
    assert isinstance(agent.get_tool_messages(llm_msg)[0], search_tool_cls)

    try:
        agent_result = agent.handle_message(llm_msg).content
    except Exception as e:
        pytest.skip(f"Skipping test: {e}")
    assert len(agent_result.split("\n\n")) == 3
    assert all(
        "lk-99" in x or "supercond" in x for x in agent_result.lower().split("\n\n")
    )
