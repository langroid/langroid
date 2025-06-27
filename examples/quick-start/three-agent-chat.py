"""
Use Langroid to set up a collaboration among three agents:

- Student: needs to write 4 key points about Language Model Training and
Evaluation, and knows nothing about these topics. It can consult two expert Agents:
- TrainingExpert: an expert on Language Model Training
- EvaluationExpert: an expert on Language Model Evaluation

To ensure that the Student's message is handled by the correct expert, it
is instructed to specify the intended recipient in the message using
"TO[<recipient>]" syntax.


Run as follows:

python3 examples/quick-start/three-agent-chat.py

"""

import typer

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tools.recipient_tool import RecipientTool
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
    student_agent.enable_message(RecipientTool)
    student_task = Task(
        student_agent,
        name="Student",
        llm_delegate=True,
        single_round=False,
        system_message="""
        Your task is to write 4 short bullet points about 
        Language Models in the context of Machine Learning (ML),
        especially about training, and evaluating them. 
        However you are a novice to this field, and know nothing about this topic. 
        To collect your bullet points, you will consult 2 people:
        TrainingExpert and EvaluationExpert.
        You will ask ONE question at a time, to ONE of these experts. 
        To clarify who your question is for, you must use 
        the `recipient_message` tool/function-call, setting 
        the `content` field to the question you want to ask, and the
        `recipient` field to either TrainingExpert or EvaluationExpert.

        Once you have collected the points you need,
        say DONE, and show me the 4 bullet points. 
        """,
    )
    training_expert_agent = ChatAgent(config)
    training_expert_task = Task(
        training_expert_agent,
        name="TrainingExpert",
        system_message="""
        You are an expert on Training Language Models in Machine Learning. 
        You will receive questions on this topic, and you must answer these
        very concisely, in one or two sentences, in a way that is easy for a novice to 
        understand.
        """,
        single_round=True,  # task done after 1 step() with valid response
    )

    evaluation_expert_agent = ChatAgent(config)
    evaluation_expert_task = Task(
        evaluation_expert_agent,
        name="EvaluationExpert",
        system_message="""
        You are an expert on Evaluating Language Models in Machine Learning. 
        You will receive questions on this topic, and you must answer these
        very concisely, in one or two sentences, in a way that is easy for a novice to 
        understand.
        """,
        single_round=True,  # task done after 1 step() with valid response
    )

    student_task.add_sub_task([training_expert_task, evaluation_expert_task])
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
