from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.base import AgentConfig, Entity
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.utils.configuration import Settings, set_global


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


def test_chat_agent(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    # just testing that these don't fail
    agent = ChatAgent(cfg)
    response = agent.llm_response("what is the capital of France?")
    assert "Paris" in response.content


def test_process_messages(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)
    msg = "What is the capital of France?"
    agent.setup_task(msg)

    agent.process_pending_message()
    assert agent.sender == Entity.LLM and "Paris" in agent.current_response.content

    agent.default_human_response = "What about England?"
    agent.process_pending_message()
    assert agent.sender == Entity.USER and "England" in agent.current_response.content

    agent.process_pending_message()
    assert agent.sender == Entity.LLM and "London" in agent.current_response.content

    agent.default_human_response = ""
    agent.process_pending_message()
    assert agent.current_response is None


def test_task(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)
    agent.default_human_response = "What is the capital of France?"

    agent.do_task(rounds=3)

    # Rounds:
    # 1. LLM initiates convo.
    # 2. User asks
    # 3. LLM responds

    assert agent.sender == Entity.LLM and "Paris" in agent.current_response.content
    assert "France" in agent.pending_message.content
