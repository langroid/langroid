"""
This example shows how you can use Langroid to define a basic Agent
encapsulating a chat LLM, and use it to set up an interactive chat session.

Run as follows:

python3 examples/quick-start/chat-agent.py

More details in the
[Getting Started guide](https://langroid.github.io/langroid/quick-start/chat-agent/)
"""

import typer
from rich import print

import langroid as lr

app = typer.Typer()

lr.utils.logging.setup_colored_logging()


def chat() -> None:
    print(
        """
        [blue]Welcome to the basic chatbot!
        Enter x or q to quit
        """
    )
    config = lr.ChatAgentConfig(
        llm=lr.language_models.OpenAIGPTConfig(
            chat_model=lr.language_models.OpenAIChatModel.GPT4o,
        ),
        vecdb=None,
    )
    agent = lr.ChatAgent(config)
    task = lr.Task(agent, name="Bot")
    task.run()


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
