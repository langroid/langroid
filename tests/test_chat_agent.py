from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.base import AgentConfig, Entity, LLM_NO_ANSWER
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


def test_responses(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)

    # direct LLM response to query
    response = agent.llm_response("what is the capital of France?")
    assert "Paris" in response.content

    # human is prompted for input, and we specify the default response
    agent.default_human_response = "What about England?"
    response = agent.user_response()
    assert "England" in response.content

    response = agent.llm_response("what about England?")
    assert "London" in response.content

    # agent attempts to handle the query, but has no response since
    # the message is not a structured msg that matches an enabled AgentMessage.
    response = agent.agent_response("What is the capital of France?")
    assert response is None


def test_process_messages(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)
    msg = "What is the capital of France?"
    agent.setup_task(msg)
    assert agent.pending_message.content == msg

    # LLM answers
    agent.process_pending_message()
    assert "Paris" in agent.current_response.content
    assert "Paris" in agent.pending_message.content
    assert agent.current_response.metadata.sender == Entity.LLM

    agent.default_human_response = "What about England?"
    # User asks about England
    agent.process_pending_message()
    assert "England" in agent.current_response.content
    assert "England" in agent.pending_message.content
    assert agent.current_response.metadata.sender == Entity.USER

    # LLM answers
    agent.process_pending_message()
    assert "London" in agent.current_response.content
    assert "London" in agent.pending_message.content
    assert agent.current_response.metadata.sender == Entity.LLM

    # It's Human's turn; they say nothing,
    # and this is reflected in `self.current_response` as None,
    # but `self.pending_message` is still set to the last message.
    agent.default_human_response = ""
    # Human says ''
    agent.process_pending_message()
    assert agent.current_response is None
    assert "London" in agent.pending_message.content
    assert agent.pending_message.metadata.sender == Entity.LLM

    # no more responders are allowed.
    agent.process_pending_message()
    assert agent.current_response is None
    assert "London" in agent.pending_message.content
    assert agent.pending_message.metadata.sender == Entity.LLM

    # reset task
    question = "What is my name?"
    agent.setup_task(
        msg=question,
        system_message=f""" Your job is to always say "{LLM_NO_ANSWER}" """,
    )
    # LLM responds with LLN_NO_ANSWER
    agent.process_pending_message()
    assert agent.current_response is None
    assert agent.pending_message.content == question
    assert agent.pending_message.metadata.sender == Entity.USER


def test_task(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)
    question = "What is the capital of France?"
    agent.default_human_response = question

    # set up task with null initial message
    agent.do_task(rounds=3)

    # Rounds:
    # 1. LLM initiates convo saying thanks how can I help (since do_task msg empty)
    # 2. User asks
    # 3. LLM responds

    assert agent.current_response.metadata.sender == Entity.LLM
    assert "Paris" in agent.current_response.content
    assert "Paris" in agent.pending_message.content

    agent.default_human_response = "What about England?"

    # set up task with initial question
    agent.do_task(msg=question, rounds=3)

    # Rounds:
    # 1. LLM answers question, since do_task has the question already
    # 2. User asks What about England?
    # 3. LLM responds

    assert agent.current_response.metadata.sender == Entity.LLM
    assert "London" in agent.current_response.content
    assert "London" in agent.pending_message.content
