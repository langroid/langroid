# script
# requires-python = ">=3.11"
# dependencies = [
#     "langroid",
# ]
# ///
"""
Basic chat example using MiniMax LLMs.

MiniMax provides high-performance language models with up to 1M context length.

Setup:
- Set the MINIMAX_API_KEY environment variable (or add to .env file).
  Get your API key from https://www.minimax.io/

Run like this:

python3 examples/basic/chat-minimax.py

Use optional arguments to change the settings:

-m <model_name>   # e.g. MiniMax-M2.7-highspeed, MiniMax-M2.5
-ns               # no streaming
-d                # debug mode
-nc               # no cache

For details on MiniMax models, see:
https://langroid.github.io/langroid/tutorials/local-llm-setup/#minimax-llms
"""

import typer
from dotenv import load_dotenv
from rich import print

import langroid as lr
import langroid.language_models as lm
from langroid.utils.configuration import Settings, set_global

app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option(
        "MiniMax-M2.7",
        "--model",
        "-m",
        help="MiniMax model name (e.g. MiniMax-M2.7, MiniMax-M2.7-highspeed)",
    ),
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
    print(
        """
        [blue]Welcome to the MiniMax chatbot!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    llm_config = lm.OpenAIGPTConfig(
        chat_model=f"minimax/{model}",
        max_output_tokens=1000,
        timeout=45,
    )

    agent_config = lr.ChatAgentConfig(
        llm=llm_config,
        system_message="You are a helpful assistant. Be concise in your answers.",
    )

    agent = lr.ChatAgent(agent_config)
    task = lr.Task(agent)
    task.run("hello")


if __name__ == "__main__":
    app()
