from langroid.agent.base import NO_ANSWER
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.vector_store.base import VectorStoreConfig


class _TestChatAgentConfig(ChatAgentConfig):
    max_tokens: int = 200
    vecdb: VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        cache_config=RedisCacheConfig(fake=False),
        chat_model=OpenAIChatModel.GPT4,
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
    # the message is not a structured msg that matches an enabled ToolMessage.
    response = agent.agent_response("What is the capital of France?")
    assert response is None


def test_process_messages(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)
    task = Task(
        agent,
        name="Test",
        llm_delegate=False,
        single_round=False,
        only_user_quits_root=False,
    )
    msg = "What is the capital of France?"
    task.init(msg)
    assert task.pending_message.content == msg

    # LLM answers
    task.step()
    assert "Paris" in task.pending_message.content
    assert task.pending_message.metadata.sender == Entity.LLM

    agent.default_human_response = "What about England?"
    # User asks about England
    task.step()
    assert "England" in task.pending_message.content
    assert task.pending_message.metadata.sender == Entity.USER

    # LLM answers
    task.step()
    assert "London" in task.pending_message.content
    assert task.pending_message.metadata.sender == Entity.LLM

    # It's Human's turn; they say nothing,
    # and this does NOT update the self.pending_message
    agent.default_human_response = ""
    # Human says ''
    task.step()
    assert "London" in task.pending_message.content
    assert task.pending_message.metadata.sender == Entity.LLM

    # Since chat was user-initiated, LLM can still respond to NO_ANSWER
    # with something like "How can I help?"
    task.step()
    assert task.pending_message.metadata.sender == Entity.LLM

    # reset task
    question = "What is my name?"
    task = Task(
        agent,
        name="Test",
        system_message=f""" Your job is to always say "{NO_ANSWER}" """,
        user_message=question,
        llm_delegate=False,
        single_round=False,
        restart=True,
        only_user_quits_root=False,
    )
    # LLM responds with NO_ANSWER
    task.init()
    task.step()
    assert NO_ANSWER in task.pending_message.content
    assert task.pending_message.metadata.sender == Entity.LLM


def test_task(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)
    task = Task(agent, name="Test", llm_delegate=False, single_round=False)
    question = "What is the capital of France?"
    agent.default_human_response = question

    # run task with null initial message
    task.run(turns=3)

    # 3 Turns:
    # 1. LLM initiates convo saying thanks how can I help (since task msg empty)
    # 2. User asks the `default_human_response`: What is the capital of France?
    # 3. LLM responds

    assert task.pending_message.metadata.sender == Entity.LLM
    assert "Paris" in task.pending_message.content

    agent.default_human_response = "What about England?"

    # run task with initial question
    task.run(msg=question, turns=3)

    # 3 Turns:
    # 1. LLM answers question, since task is run with the question
    # 2. User asks the `default_human_response`: What about England?
    # 3. LLM responds

    assert task.pending_message.metadata.sender == Entity.LLM
    assert "London" in task.pending_message.content


def test_simple_task(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)
    task = Task(
        agent,
        interactive=False,
        single_round=True,
        system_message="""
        User will give you a number, respond with the square of the number.
        """,
    )

    response = task.run(msg="5")
    assert "25" in response.content

    # create new task with SAME agent, and restart=True,
    # verify that this works fine, i.e. does not use previous state of agent

    task = Task(
        agent,
        interactive=False,
        single_round=True,
        restart=True,
        system_message="""
        User will give you a number, respond with the square of the number.
        """,
    )

    response = task.run(msg="7")
    assert "49" in response.content
