"""
A simple example of two agents collaborating on a task.

The Student Agent is tasked with writing 3 key points on Language Models,
  and it is told that it knows nothing about the topic, and
  can consult an Expert Agent for help.

Run as follows:

python3 examples/quick-start/two-agent-chat.py

"""

import typer

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()


def chat() -> None:
    config = ChatAgentConfig(
        llm=OpenAIGPTConfig(
            chat_model=OpenAIChatModel.GPT4o,
        ),
        vecdb=None,
    )
    student_agent = ChatAgent(config)
    student_task = Task(
        student_agent,
        name="Student",
        system_message="""
        Your task is to write 3 short bullet points about 
        Language Models in the context of Machine Learning. 
        However you are a novice to this field, and know nothing about this topic. 
        To collect your bullet points, you can ask me questions,
        one at a time, which I will answer.
        Once you have what you need, say DONE, and show me the 3 bullet points. 
        """,
    )
    expert_agent = ChatAgent(config)
    expert_task = Task(
        expert_agent,
        name="Expert",
        system_message="""
        You are an expert on Language Models in Machine Learning. 
        You will receive questions on this topic, and you must answer these
        very concisely, in one or two sentences, in a way that is easy for a novice to 
        understand.
        """,
        single_round=True,  # task done after 1 step() with valid response
    )
    student_task.add_sub_task(expert_task)
    student_task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat()


if __name__ == "__main__":
    app()
