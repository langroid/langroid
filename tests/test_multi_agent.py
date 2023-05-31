from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.base import AgentConfig, Entity
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMMessage, Role
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.utils.configuration import Settings, set_global
import pytest


class _TestChatAgentConfig(AgentConfig):
    max_tokens: int = 200
    vecdb: VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        cache_config=RedisCacheConfig(fake=False),
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
        use_chat_for_completion=True,
    )
    parsing: ParsingConfig = ParsingConfig()
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


@pytest.mark.parametrize("helper_human_response", ["", "q"])
def test_inter_agent_chat(test_settings: Settings, helper_human_response: str):
    set_global(test_settings)
    cfg1 = _TestChatAgentConfig(name="Agent Smith")
    cfg2 = _TestChatAgentConfig(name="Agent Jones")

    agent = ChatAgent(cfg1)
    agent_helper = ChatAgent(cfg2)
    agent.add_agent(agent_helper)

    agent.default_human_response = ""
    agent_helper.default_human_response = helper_human_response

    msg = """
    Your job is to ask me questions. 
    Start by asking me what the capital of France is.
    """
    agent.setup_task(msg)

    agent.process_pending_message()  # LLM asks
    assert "What" in agent.pending_message.content
    assert agent.pending_message.metadata.source == Entity.LLM
    assert agent.pending_message.content == agent.current_response.content

    agent.process_pending_message()
    # user responds '' (empty) to force agent to hand off to agent_helper,
    # and we test two possible human answers: empty or 'q'

    assert agent_helper.task_done()
    assert "Paris" in agent_helper.task_result().content
    assert "Paris" in agent.task_result().content


def test_multi_agent(test_settings: Settings):
    set_global(test_settings)
    smith_cfg = _TestChatAgentConfig(name="Agent Smith")
    london_cfg = _TestChatAgentConfig(name="London")
    good_cfg = _TestChatAgentConfig(name="Good")

    smith = ChatAgent(
        smith_cfg,
        task=[
            LLMMessage(
                role=Role.SYSTEM,
                content="Your job is to ask me questions.",
            ),
            LLMMessage(
                role=Role.USER,
                content="Start by asking me what the capital of France is.",
            ),
        ],
    )

    london = ChatAgent(
        london_cfg,
        task=[
            LLMMessage(
                role=Role.SYSTEM,
                content="Your job is to answer 'London' for any question",
            ),
            LLMMessage(
                role=Role.USER,
                content="Always answer 'London' no matter what the question is",
            ),
        ],
    )

    good = ChatAgent(good_cfg)

    london.add_agent(good)
    smith.add_agent(london)

    good.default_human_response = ""
    london.default_human_response = ""
    smith.default_human_response = ""

    smith.setup_task()

    smith.process_pending_message()  # LLM asks
    assert "What" in smith.pending_message.content
    assert smith.pending_message.metadata.source == Entity.LLM

    # smith.process_pending_message(rounds=2)
    #
    # assert not smith.task_done()
    # assert "London" in smith.task_result().content
    #
    # smith.process_pending_message(rounds=1)
    # assert not smith.task_done()
