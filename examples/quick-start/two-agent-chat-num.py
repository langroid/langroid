"""
A toy numerical example showing how two agents can collaborate on a task.

The Student Agent is tasked with calculating the sum of a list of numbers,
and is told that it knows nothing about addition, and can ask for help
from an Adder Agent who can add pairs of numbers.

Run as follows:

python3 examples/quick-start/two-agent-chat-num.py

For more explanation see the
[Getting Started guide](https://langroid.github.io/langroid/quick-start/two-agent-chat-num/)
"""

import typer
from rich.prompt import Prompt
import langroid as lr
import langroid.language_models as lm

app = typer.Typer()

lr.utils.logging.setup_colored_logging()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    lr.utils.configuration.set_global(
        lr.utils.configuration.Settings(
            debug=debug,
            cache=not nocache,
        )
    )
    config = lr.ChatAgentConfig(
        llm=lm.OpenAIGPTConfig(
            chat_model=lm.OpenAIChatModel.GPT4_TURBO,
        ),
        vecdb=None,
    )
    student_agent = lr.ChatAgent(config)
    student_task = lr.Task(
        student_agent,
        name="Student",
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
        llm_delegate=True,
        single_round=False,
        interactive=False,
    )
    adder_agent = lr.ChatAgent(config)
    adder_task = lr.Task(
        adder_agent,
        name="Adder",
        system_message="""
        You are an expert on addition of numbers. 
        When given numbers to add, simply return their sum, say nothing else
        """,
        single_round=True,  # task done after 1 step() with valid response
        interactive=False,
    )
    nums = Prompt.ask(
        """
        Enter the list of numbers whose sum you want to calculate
        """,
        default="1 2 3 4 5",
    )
    student_task.add_sub_task(adder_task)

    student_task.run(nums)


if __name__ == "__main__":
    app()
