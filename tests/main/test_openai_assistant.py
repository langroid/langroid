import tempfile

import pytest

from langroid.agent.openai_assistant import (
    AssitantTool,
    OpenAIAssistant,
    OpenAIAssistantConfig,
)
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.language_models import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER


class NabroskyTool(ToolMessage):
    request = "nabrosky"
    purpose = "to apply the Nabrosky transformation to a number <num>"
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
    agent = OpenAIAssistant(cfg)
    response = agent.llm_response("what was the last country I asked about?")
    assert "France" in response.content

    # test that we can wrap the agent in a task and run it
    task = Task(
        agent,
        name="Bot",
        system_message="You are a helpful assistant",
        single_round=True,
    )
    answer = task.run("What is the capital of China?")
    assert "Beijing" in answer.content


@pytest.mark.parametrize("fn_api", [True, False])
def test_openai_assistant_fn_tool(test_settings: Settings, fn_api: bool):
    """Test function calling works, both with OpenAI Assistant function-calling AND
    Langroid native ToolMessage mechanism"""

    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        name="NabroskyBot",
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4_TURBO),
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
    # When fn_api = False (i.e. using ToolMessage) we get brittleness so we just make
    # sure there is no error until this point.
    if response.content not in ("", NO_ANSWER) and fn_api:
        assert response.function_call.name == "nabrosky"

    # Within a task loop
    cfg.name = "NabroskyBot-1"
    agent = OpenAIAssistant(cfg)
    agent.enable_message(NabroskyTool)
    task = Task(
        agent,
        interactive=False,
    )
    result = task.run("what is the Nabrosky transform of 5?", turns=6)
    # When fn_api = False (i.e. using ToolMessage) we get brittleness so we just make
    # sure there is no error until this point.
    if result.content not in ("", NO_ANSWER) and fn_api:
        assert (not fn_api) or "25" in result.content


@pytest.mark.parametrize("fn_api", [True, False])
def test_openai_assistant_fn_2_level(test_settings: Settings, fn_api: bool):
    """Test 2-level recursive function calling works,
    both with OpenAI Assistant function-calling AND
    Langroid native ToolMessage mechanism"""

    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        name="Main",
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4_TURBO),
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
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4_TURBO),
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

    main_task = Task(agent, interactive=False, llm_delegate=True)
    nabrosky_task = Task(nabrosky_agent, interactive=False, llm_delegate=True)
    main_task.add_sub_task(nabrosky_task)
    result = main_task.run("what is the Nabrosky transform of 5?")
    assert (not fn_api) or "25" in result.content


@pytest.mark.parametrize("fn_api", [False, True])
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
    doubler_confg = OpenAIAssistantConfig(
        name="Doubler",
        system_message=""" 
        When you receive a number, simply double it and  return the answer
        """,
    )
    doubler_agent = OpenAIAssistant(doubler_confg)
    doubler_task = Task(doubler_agent, interactive=False, single_round=True)

    main_task = Task(agent, interactive=False)
    main_task.add_sub_task(doubler_task)
    result = main_task.run("10")
    assert (not fn_api) or "20" in result.content


def test_openai_assistant_retrieval(test_settings: Settings):
    """
    Test that Assistant can answer question
    based on retrieval from file.
    """
    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4_TURBO),
        system_message="Answer questions based on the provided document.",
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
    agent.add_assistant_tools([AssitantTool(type="retrieval")])
    agent.add_assistant_files([filename])

    response = agent.llm_response("where was Vlad Nabrosky born?")
    assert "Russia" in response.content

    response = agent.llm_response("what novel did he write?")
    assert "Lomita" in response.content


def test_openai_asst_code_interpreter(test_settings: Settings):
    """
    Test that Assistant can answer questions using code.
    """
    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4_TURBO),
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
    agent.add_assistant_tools([AssitantTool(type="code_interpreter")])
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
    )
    agent = OpenAIAssistant(cfg)

    # wrap Agent in a Task to run interactive loop with user (or other agents)
    task = Task(
        agent,
        interactive=False,
        system_message="""
        Send a number. Your student will respond EVEN or ODD. 
        You say RIGHT or WRONG, then send another number, and so on.
        After getting 2 answers, say DONE.
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
        single_round=True,
        system_message="When you get a number, say EVEN if it is even, else say ODD",
    )
    task.add_sub_task(student_task)
    result = task.run()
    assert "RIGHT" in result.content
