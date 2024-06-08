"""
A simple example of a Langroid Agent equipped with a Tool/function-calling.

The Agent has a "secret" list of numbers in "mind", and the LLM's task is to
find the smallest number in the list. The LLM can make use of the ProbeTool
which takes a number as argument. The agent's `probe` method handles this tool,
and returns the number of numbers in the list that are less than or equal to the
number in the ProbeTool message.

Run as follows:

python3 examples/quick-start/chat-agent-tool.py

For more explanation see
[the Getting Started guide](https://langroid.github.io/langroid/quick-start/chat-agent-tool/).
"""

import typer
from rich import print
from langroid.pydantic_v1 import BaseSettings

import langroid as lr

app = typer.Typer()

lr.utils.logging.setup_colored_logging()


class ProbeTool(lr.agent.ToolMessage):
    request: str = "probe"
    purpose: str = """
        To find how many numbers in my list are less than or equal to  
        the <number> you specify.
        """
    number: int


class SpyGameAgent(lr.ChatAgent):
    def __init__(self, config: lr.ChatAgentConfig):
        super().__init__(config)
        self.numbers = [3, 4, 8, 11, 15]

    def probe(self, msg: ProbeTool) -> str:
        # return how many numbers in self.numbers are less or equal to msg.number
        return str(len([n for n in self.numbers if n <= msg.number]))


class CLIOptions(BaseSettings):
    fn_api: bool = False  # whether to use OpenAI's function-calling


def chat(opts: CLIOptions) -> None:
    print(
        """
        [blue]Welcome to the number guessing game!
        Enter x or q to quit
        """
    )
    spy_game_agent = SpyGameAgent(
        lr.ChatAgentConfig(
            name="Spy",
            llm=lr.language_models.OpenAIGPTConfig(
                chat_model=lr.language_models.OpenAIChatModel.GPT4,
            ),
            vecdb=None,
            use_tools=not opts.fn_api,
            use_functions_api=opts.fn_api,
        )
    )

    spy_game_agent.enable_message(ProbeTool)
    task = lr.Task(
        spy_game_agent,
        system_message="""
            I have a list of numbers between 1 and 20.
            Your job is to find the smallest of them.
            To help with this, you can give me a number and I will
            tell you how many of my numbers are equal or less than your number.
            Once you have found the smallest number,
            you can say DONE and report your answer.
        """,
    )
    task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    fn_api: bool = typer.Option(False, "--fn_api", "-f", help="use functions api"),
) -> None:
    lr.utils.configuration.set_global(
        lr.utils.configuration.Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat(CLIOptions(fn_api=fn_api))


if __name__ == "__main__":
    app()
