from llmagent.agent.base import NO_ANSWER, Entity
from llmagent.agent.chat_agent import ChatAgent, ChatAgentConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.utils.configuration import Settings, set_global
from llmagent.vector_store.base import VectorStoreConfig


class _TestChatAgentConfig(ChatAgentConfig):
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
    agent.init_chat(user_message=msg)
    assert agent.pending_message.content == msg

    # LLM answers
    agent.process_pending_message()
    assert "Paris" in agent.pending_message.content
    assert agent.pending_message.metadata.sender == Entity.LLM

    agent.default_human_response = "What about England?"
    # User asks about England
    agent.process_pending_message()
    assert "England" in agent.pending_message.content
    assert agent.pending_message.metadata.sender == Entity.USER

    # LLM answers
    agent.process_pending_message()
    assert "London" in agent.pending_message.content
    assert agent.pending_message.metadata.sender == Entity.LLM

    # It's Human's turn; they say nothing,
    # and this is reflected in `self.pending_message` as NO_ANSWER
    agent.default_human_response = ""
    # Human says ''
    agent.process_pending_message()
    assert NO_ANSWER in agent.pending_message.content
    assert agent.pending_message.metadata.sender == Entity.USER

    # Since chat was user-initiated, LLM can still respond to NO_ANSWER
    # with something like "How can I help?"
    agent.process_pending_message()
    assert NO_ANSWER not in agent.pending_message.content
    assert agent.pending_message.metadata.sender == Entity.LLM

    # reset task
    question = "What is my name?"
    agent.init_chat(
        system_message=f""" Your job is to always say "{NO_ANSWER}" """,
        user_message=question,
        restart=True,
    )
    # LLM responds with NO_ANSWER
    agent.process_pending_message()
    assert NO_ANSWER in agent.pending_message.content
    assert agent.pending_message.metadata.sender == Entity.LLM


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
    # 2. User asks the `default_human_response`: What is the capital of France?
    # 3. LLM responds

    assert agent.pending_message.metadata.sender == Entity.LLM
    assert "Paris" in agent.pending_message.content

    agent.default_human_response = "What about England?"

    # set up task with initial question
    agent.do_task(msg=question, rounds=3)

    # Rounds:
    # 1. LLM answers question, since do_task has the question already
    # 2. User asks the `default_human_response`: What about England?
    # 3. LLM responds

    assert agent.pending_message.metadata.sender == Entity.LLM
    assert "London" in agent.pending_message.content
