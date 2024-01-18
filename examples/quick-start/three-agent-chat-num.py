"""
Use Langroid to set up a collaboration among three agents:

- Processor: needs to transform a list of numbers, does not know how to
apply the transformations, and sends out each number so that one of two
specialized agents apply the transformation.
- EvenHandler only transforms even numbers, otherwise says `DO-NOT-KNOW`
- OddHandler only transforms odd numbers, otherwise says `DO-NOT-KNOW`

Run as follows:

python3 examples/quick-start/three-agent-chat-num.py

For more explanation, see the
[Getting Started guide](https://langroid.github.io/langroid/quick-start/three-agent-chat-num/)
"""

import typer
import langroid as lr

app = typer.Typer()

lr.utils.logging.setup_colored_logging()

NO_ANSWER = lr.utils.constants.NO_ANSWER


def chat() -> None:
    config = lr.ChatAgentConfig(
        llm=lr.language_models.OpenAIGPTConfig(
            chat_model=lr.language_models.OpenAIChatModel.GPT4,
        ),
        vecdb=None,
    )
    processor_agent = lr.ChatAgent(config)
    processor_task = lr.Task(
        processor_agent,
        name="Processor",
        system_message="""
        You will receive a list of numbers from the user.
        Your goal is to apply a transformation to each number.
        However you do not know how to do this transformation, 
        so the user will help you. 
        You can simply send the user each number FROM THE GIVEN LIST
        and the user will return the result 
        with the appropriate transformation applied.
        IMPORTANT: only send one number at a time, concisely, say nothing else.
        Once you have accomplished your goal, say DONE and show the result.
        Start by asking the user for the list of numbers.
        """,
        single_round=False,
    )
    even_agent = lr.ChatAgent(config)
    even_task = lr.Task(
        even_agent,
        name="EvenHandler",
        system_message=f"""
        You will be given a number. 
        If it is even, divide by 2 and say the result, nothing else.
        If it is odd, say {NO_ANSWER}
        """,
        single_round=True,  # task done after 1 step() with valid response
    )

    odd_agent = lr.ChatAgent(config)
    odd_task = lr.Task(
        odd_agent,
        name="OddHandler",
        system_message=f"""
        You will be given a number n. 
        If it is odd, return (n*3+1), say nothing else. 
        If it is even, say {NO_ANSWER}
        """,
        single_round=True,  # task done after 1 step() with valid response
    )

    processor_task.add_sub_task([even_task, odd_task])
    processor_task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    lr.utils.configuration.set_global(
        lr.utils.configuration.Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat()


if __name__ == "__main__":
    app()
