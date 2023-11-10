import tempfile

from langroid.agent.openai_assistant import (
    AssitantTool,
    OpenAIAssistant,
    OpenAIAssistantConfig,
)
from langroid.agent.task import Task
from langroid.language_models import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global


def test_openai_assistant(test_settings: Settings):
    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        use_cached_assistant=False,
        use_cached_thread=False,
    )
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


def test_openai_assistant_retrieval(test_settings: Settings):
    """
    Test that Assistant can answer question
    based on retrieval from file.
    """
    set_global(test_settings)
    cfg = OpenAIAssistantConfig(
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4_TURBO),
        use_cached_assistant=False,
        use_cached_thread=False,
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
        default_human_response="",
        only_user_quits_root=False,
        system_message="""
        Send a number. Your student will responde EVEN or ODD. 
        You say RIGHT or WRONG, then send another number, and so on.
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
        default_human_response="",
        single_round=True,
        system_message="When you get a number, say EVEN if it is even, else say ODD",
    )
    task.add_sub_task(student_task)
    task.run(turns=5)
