import tempfile

import pytest

from langroid.agent.openai_assistant import (
    AssistantTool,
    OpenAIAssistant,
    OpenAIAssistantConfig,
    ToolType,
)
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.language_models import OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER


class NabroskyTool(ToolMessage):
    request: str = "nabrosky"
    purpose: str = "to apply the Nabrosky transformation to a number <num>"
    num: int

    def handle(self) -> str:
        return str(self.num**2)


def test_openai_assistant(test_settings: Settings):
    set_global(test_settings)
    cfg = OpenAIAssistantConfig()
    agent = OpenAIAssistant(cfg)
    response = agent.llm_response("what is the capital of France?")
    assert "Paris" in response.content

    # test that we can retrieve cached asst, thread, and it recalls the last question
    cfg = OpenAIAssistantConfig(
        use_cached_assistant=True,
        use_cached_thread=True,
    )
    agent1 = OpenAIAssistant(cfg)
    response = agent1.llm_response("what was the last country I asked about?")
    if (
        agent1.thread.id == agent.thread.id
        and agent1.assistant.id == agent.assistant.id
    ):
        assert "France" in response.content

    # test that we can wrap the agent in a task and run it
    task = Task(
        agent,
        name="Bot",
        system_message="You are a helpful assistant",
        done_if_response=[Entity.LLM],
        interactive=False,
    )
    answer = task.run("What is the capital of China?")
    assert "Beijing" in answer.content


def test_openai_assistant_cache(test_settings: Settings):
    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        cache_responses=True,
    )
    agent = OpenAIAssistant(cfg)
    question = "Who wrote the novel War and Peace?"
    agent.llm.cache.delete_keys_pattern(f"*{question}*")
    response = agent.llm_response(question)
    assert "Tolstoy" in response.content

    # create fresh agent, and use a NEW thread
    cfg = OpenAIAssistantConfig(
        name="New",
        cache_responses=True,
        use_cached_assistant=False,
        use_cached_thread=False,
    )
    agent = OpenAIAssistant(cfg)
    # now this answer should be found in cache
    response = agent.llm_response(question)
    assert "Tolstoy" in response.content
    assert response.metadata.cached
    # check that we were able to insert assistant response and continue conv.
    response = agent.llm_response("When was he born?")
    assert "1828" in response.content

    # create fresh agent, and use a NEW thread, check BOTH answers should be cached.
    cfg = OpenAIAssistantConfig(
        name="New2",
        cache_responses=True,
        use_cached_assistant=False,
        use_cached_thread=False,
    )
    agent = OpenAIAssistant(cfg)
    # now this answer should be found in cache
    response = agent.llm_response("Who wrote the novel War and Peace?")
    assert "Tolstoy" in response.content
    assert response.metadata.cached
    # check that we were able to insert assistant response and continue conv.
    response = agent.llm_response("When was he born?")
    assert "1828" in response.content
    assert response.metadata.cached


@pytest.mark.xfail(
    reason="Flaky due to non-deterministic LLM tool-use behavior",
    run=True,
    strict=False,
)
@pytest.mark.parametrize("fn_api", [True, False])
def test_openai_assistant_fn_tool(test_settings: Settings, fn_api: bool):
    """Test function calling works, both with OpenAI Assistant function-calling AND
    Langroid native ToolMessage mechanism"""

    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        name="NabroskyBot",
        llm=OpenAIGPTConfig(),
        use_functions_api=fn_api,
        use_tools=not fn_api,
        system_message="""
        The user will ask you, 'What is the Nabrosky transform of...' a certain number.
        You do NOT know the answer, and you should NOT guess the answer.
        Instead you MUST use the `nabrosky` JSON function/tool to find out.
        When you receive the answer, say DONE and show the answer.
        """,
    )
    agent = OpenAIAssistant(cfg)
    agent.enable_message(NabroskyTool)
    response = agent.llm_response("what is the Nabrosky transform of 5?")
    # When fn_api is used, the LLM should produce a function_call (not text
    # content). Assert unconditionally so that a regression surfaces as an
    # xfail rather than silently passing.
    if fn_api:
        assert (
            response.function_call is not None
        ), "Expected function_call but LLM responded with text"
        assert response.function_call.name == "nabrosky"

    # Within a task loop
    cfg.name = "NabroskyBot-1"
    agent = OpenAIAssistant(cfg)
    agent.enable_message(NabroskyTool)
    task = Task(
        agent,
        interactive=False,
    )
    result = task.run("what is the Nabrosky transform of 5?", turns=4)
    # When fn_api = False (i.e. using ToolMessage) we get brittleness so we just make
    # sure there is no error until this point.
    if result.content not in ("", NO_ANSWER) and fn_api:
        assert "25" in result.content


@pytest.mark.xfail(
    reason="Flaky/Soon-To-be-deprecated API, may fail",
    run=True,
    strict=False,
)
@pytest.mark.parametrize("fn_api", [True, False])
def test_openai_assistant_fn_2_level(test_settings: Settings, fn_api: bool):
    """Test 2-level recursive function calling works,
    both with OpenAI Assistant function-calling AND
    Langroid native ToolMessage mechanism"""

    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        name="Main",
        llm=OpenAIGPTConfig(),
        use_functions_api=fn_api,
        use_tools=not fn_api,
        system_message="""
        The user will ask you to apply the Nabrosky transform to a number.
        You do not know how to do it, and you should NOT guess the answer.
        Instead you MUST use the `recipient_message` tool/function to 
        send it to NabroskyBot who will do it for you.
        When you receive the answer, say DONE and show the answer.
        """,
    )
    agent = OpenAIAssistant(cfg)
    agent.enable_message(RecipientTool)

    nabrosky_cfg = OpenAIAssistantConfig(
        name="NabroskyBot",
        llm=OpenAIGPTConfig(),
        use_functions_api=fn_api,
        use_tools=not fn_api,
        system_message="""
        The user will ask you to apply the Nabrosky transform to a number.
        You do not know how to do it, and you should NOT guess the answer.
        Instead you MUST use the `nabrosky` function/tool to do it.
        When you receive the answer say DONE and show the answer.
        """,
    )

    nabrosky_agent = OpenAIAssistant(nabrosky_cfg)
    nabrosky_agent.enable_message(NabroskyTool)

    main_task = Task(agent, interactive=False)
    nabrosky_task = Task(nabrosky_agent, interactive=False)
    main_task.add_sub_task(nabrosky_task)
    result = main_task.run("what is the Nabrosky transform of 5?", turns=6)
    if fn_api and result.content not in ("", NO_ANSWER):
        assert "25" in result.content


@pytest.mark.parametrize("fn_api", [True, False])
def test_openai_assistant_recipient_tool(test_settings: Settings, fn_api: bool):
    """Test that special case of fn-calling: RecipientTool works,
    both with OpenAI Assistant function-calling AND
    Langroid native ToolMessage mechanism"""

    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        name="Main",
        use_functions_api=fn_api,
        use_tools=not fn_api,
        system_message="""
        The user will give you a number. You need to double it, but don't know how,
        so you send it to the "Doubler" to double it. 
        When you receive the answer, say DONE and show the answer.
        """,
    )
    agent = OpenAIAssistant(cfg)
    agent.enable_message(RecipientTool)

    # Within a task loop
    doubler_config = OpenAIAssistantConfig(
        name="Doubler",
        system_message=""" 
        When you receive a number, simply double it and  return the answer
        """,
    )
    doubler_agent = OpenAIAssistant(doubler_config)
    doubler_task = Task(
        doubler_agent,
        interactive=False,
        done_if_response=[Entity.LLM],
    )

    main_task = Task(agent, interactive=False)
    main_task.add_sub_task(doubler_task)
    result = main_task.run("10", turns=4)
    if fn_api and result.content not in ("", NO_ANSWER):
        assert "20" in result.content


@pytest.mark.skip(
    """
This no longer works since the OpenAI Assistants API for file_search
has changed, and requires explicit vector-store creation:
https://platform.openai.com/docs/assistants/tools/file-search
We will update langroid to catch up with this at some point.
"""
)
def test_openai_assistant_retrieval(test_settings: Settings):
    """
    Test that Assistant can answer question
    based on retrieval from file.
    """
    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        llm=OpenAIGPTConfig(),
        system_message="""
        Answer questions based on the provided file, using the `file_search` tool
        """,
    )
    agent = OpenAIAssistant(cfg)

    # create temp file with in-code text content
    text = """
    Vladislav Nabrosky was born in China. He then emigrated to the United States,
    where he wrote the novel Lomita. He was a professor at Purnell University.
    """
    # open a temp file and write text to it
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(text)
        f.close()
        # get the filename
        filename = f.name

    # must enable retrieval first, then add file
    agent.add_assistant_tools([AssistantTool(type=ToolType.RETRIEVAL)])
    agent.add_assistant_files([filename])

    response = agent.llm_response("where was Vladislav Nabrosky born?")
    assert "China" in response.content

    response = agent.llm_response("what novel did he write?")
    assert "Lomita" in response.content


@pytest.mark.xfail(
    reason="May fail due to unknown flakiness",
    run=True,
    strict=False,
)
def test_openai_asst_code_interpreter(test_settings: Settings):
    """
    Test that Assistant can answer questions using code.
    """
    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        llm=OpenAIGPTConfig(),
        system_message="Answer questions by running code if needed",
    )
    agent = OpenAIAssistant(cfg)

    # create temp file with in-code text content
    text = """
    Vlad Nabrosky was born in Russia. He then emigrated to the United States,
    where he wrote the novel Lomita. He was a professor at Purnell University.
    """

    # open a temp file and write text to it
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(text)
        f.close()
        # get the filename
        filename = f.name

    # must enable retrieval first, then add file
    agent.add_assistant_tools([AssistantTool(type="code_interpreter")])
    agent.add_assistant_files([filename])

    response = agent.llm_response(
        "what is the 10th fibonacci number, when you start with 1 and 2?"
    )
    assert "89" in response.content

    response = agent.llm_response("how many words are in the file?")
    assert str(len(text.split())) in response.content


def test_openai_assistant_multi(test_settings: Settings):
    """
    Test task delegation with OpenAIAssistant
    """
    set_global(test_settings)

    cfg = OpenAIAssistantConfig(
        use_cached_assistant=False,
        use_cached_thread=False,
        name="Teacher",
        llm=OpenAIGPTConfig(),
    )
    agent = OpenAIAssistant(cfg)

    # wrap Agent in a Task to run interactive loop with user (or other agents)
    task = Task(
        agent,
        interactive=False,
        system_message="""
        Send a number. Your student will respond EVEN or ODD. 
        You say RIGHT DONE or WRONG DONE.
        
        Start by sending a number.
        """,
    )

    cfg = OpenAIAssistantConfig(
        use_cached_assistant=False,
        use_cached_thread=False,
        name="Student",
    )
    student_agent = OpenAIAssistant(cfg)
    student_task = Task(
        student_agent,
        interactive=False,
        done_if_response=[Entity.LLM],
        system_message="When you get a number, say EVEN if it is even, else say ODD",
    )
    task.add_sub_task(student_task)
    result = task.run()
    assert "RIGHT" in result.content
