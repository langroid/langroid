"""
A toy numerical example showing how two agents can collaborate on a task.

The Student Agent is tasked with calculating the sum of a list of numbers,
and is told that it knows nothing about addition, and can ask for help
from an Adder Agent who can add pairs of numbers.

Run as follows (omit -m to default to GTP4o):

python3 examples/quick-start/two-agent-chat-num.py -m ollama/qwen2.5:latest

For more explanation see the
[Getting Started guide](https://langroid.github.io/langroid/quick-start/two-agent-chat-num/)
"""

import typer
from rich.prompt import Prompt

import langroid as lr

app = typer.Typer()

lr.utils.logging.setup_colored_logging()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
) -> None:
    lr.utils.configuration.set_global(
        lr.utils.configuration.Settings(
            debug=debug,
            cache=not nocache,
        )
    )

    llm_config = lr.language_models.OpenAIGPTConfig(
        chat_model=model or lr.language_models.OpenAIChatModel.GPT4o,
    )

    student_config = lr.ChatAgentConfig(
        name="Student",
        llm=llm_config,
        vecdb=None,
        system_message="""
            You will receive a list of numbers from me (the User),
            and your goal is to calculate their sum.
            However you do not know how to add numbers.
            I can help you add numbers, two at a time, since
            I only know how to add pairs of numbers.
            Send me a pair of numbers to add, one at a time, 
            and I will tell you their sum.
            For each question, simply ask me the sum in math notation, 
            e.g., simply say "1 + 2", etc, and say nothing else.
            Once you have added all the numbers in the list, 
            say DONE and give me the final sum. 
        """,
    )
    student_agent = lr.ChatAgent(student_config)
    student_task = lr.Task(
        student_agent,
        name="Student",
        interactive=False,
        single_round=False,
        llm_delegate=True,
    )

    adder_config = lr.ChatAgentConfig(
        name="Adder",
        llm=llm_config,
        vecdb=None,
        system_message="""
            You are an expert on addition of numbers. 
            When given numbers to add, simply return their sum, say nothing else
            """,
    )
    adder_agent = lr.ChatAgent(adder_config)
    adder_task = lr.Task(
        adder_agent,
        interactive=False,
        single_round=True,
    )

    student_task.add_sub_task(adder_task)

    nums = Prompt.ask(
        """
        Enter the list of numbers whose sum you want to calculate
        """,
        default="3 1 5 2",
    )

    student_task.run(nums)


if __name__ == "__main__":
    app()
