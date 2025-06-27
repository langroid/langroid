"""
Use Langroid to set up a collaboration among three agents:

- Processor: needs to transform a number, does not know how to
apply the transformation, and sends out the number so that one of two
specialized agents apply the transformation.
- EvenHandler only transforms even numbers, otherwise says `DO-NOT-KNOW`
- OddHandler only transforms odd numbers, otherwise says `DO-NOT-KNOW`

Run as follows (omit -m <model> to default to GPT4o):

python3 examples/quick-start/three-agent-chat-num.py -m gemini/gemini-2.0-flash-exp

For more explanation, see the
[Getting Started guide](https://langroid.github.io/langroid/quick-start/three-agent-chat-num/)
"""

import typer
from rich.prompt import Prompt

import langroid as lr

app = typer.Typer()

lr.utils.logging.setup_colored_logging()

NO_ANSWER = lr.utils.constants.NO_ANSWER


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    lr.utils.configuration.set_global(
        lr.utils.configuration.Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    llm_config = lr.language_models.OpenAIGPTConfig(
        chat_model=model or lr.language_models.OpenAIChatModel.GPT4o,
        # or, e.g., "ollama/qwen2.5-coder:latest", or "gemini/gemini-2.0-flash-exp"
    )

    processor_config = lr.ChatAgentConfig(
        name="Processor",
        llm=llm_config,
        system_message="""
        You will receive a number from the user.
        Simply repeat that number, DO NOT SAY ANYTHING else,
        and wait for a TRANSFORMATION of the number 
        to be returned to you.
        
        Once you have received the RESULT, simply say "DONE",
        do not say anything else.
        """,
        vecdb=None,
    )

    processor_agent = lr.ChatAgent(processor_config)
    processor_task = lr.Task(
        processor_agent,
        interactive=False,
        single_round=False,
    )

    even_config = lr.ChatAgentConfig(
        name="EvenHandler",
        llm=llm_config,
        system_message=f"""
        You will be given a number N. Respond as follows:
        
        - If N is even, divide N by 2 and show the result, 
          in the format: 
            RESULT = <result>
          and say NOTHING ELSE.
        - If N is odd, say {NO_ANSWER}
        """,
    )
    even_agent = lr.ChatAgent(even_config)
    even_task = lr.Task(
        even_agent,
        single_round=True,  # task done after 1 step() with valid response
    )

    odd_config = lr.ChatAgentConfig(
        name="OddHandler",
        llm=llm_config,
        system_message=f"""
        You will be given a number N. Respond as follows:
        
        - if N is odd, return the result (N*3+1), in the format:
            RESULT = <result> 
            and say NOTHING ELSE.
        
        - If N is even, say {NO_ANSWER}
        """,
    )
    odd_agent = lr.ChatAgent(odd_config)
    odd_task = lr.Task(
        odd_agent,
        single_round=True,  # task done after 1 step() with valid response
    )

    processor_task.add_sub_task([even_task, odd_task])
    number = Prompt.ask(
        "[blue]What number do you want to transform? ",
        default="11",
    )

    processor_task.run(number)


if __name__ == "__main__":
    app()
