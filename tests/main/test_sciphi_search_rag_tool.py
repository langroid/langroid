"""
NOTE: running this example requires setting the SCIPHI_API_KEY environment
variable in your `.env` file, as explained in the documentation for the SciPhi API.

"""
import itertools

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.sciphi_search_rag_tool import SciPhiSearchRAGTool
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import (
    OpenAIChatModel,
    OpenAIGPTConfig,
)
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global

cfg = ChatAgentConfig(
    name="test-langroid",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        cache_config=RedisCacheConfig(fake=False),
    ),
    parsing=ParsingConfig(),
    prompts=PromptsConfig(),
    use_functions_api=False,
    use_tools=True,
)
agent = ChatAgent(cfg)

# Define the range of values each variable can have
use_vals = [True, False]
handle_vals = [True, False]
force_vals = [True, False]
message_classes = [None, SciPhiSearchRAGTool]

# Get the cartesian product
cartesian_product = list(
    itertools.product(message_classes, use_vals, handle_vals, force_vals)
)

agent.enable_message(SciPhiSearchRAGTool)


# @pytest.mark.parametrize("use_functions_api", [True, False])
@pytest.mark.parametrize("use_functions_api", [True])
def test_llm_tool_message(
    test_settings: Settings,
    use_functions_api: bool,
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
    agent.enable_message(SciPhiSearchRAGTool)

    llm_msg = agent.llm_response_forget(
        "Find 3 results on the internet about the LK-99 superconducting material."
    )

    tool_name = SciPhiSearchRAGTool.default_value("request")

    if use_functions_api:
        assert llm_msg.function_call.name == tool_name

    else:
        tools = agent.get_tool_messages(llm_msg)
        assert len(tools) == 1
        assert isinstance(tools[0], SciPhiSearchRAGTool)
    agent_result = agent.handle_message(llm_msg)
    # check there are at least 3 results
    assert len(agent_result.split("\n\n")) >= 3
    # check that the some key terms appear in at least 3 paragraphs
    assert (
        sum(
            "lk-99" in x or "supercond" in x for x in agent_result.lower().split("\n\n")
        )
        >= 3
    )
