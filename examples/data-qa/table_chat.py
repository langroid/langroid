"""
Example showing how to chat with a tabular dataset:
csv, tsv, or any other pandas-readable.

Run like this

python3 examples/data-qa/table_chat.py

Optional args:
* -d or --debug to enable debug mode
* -ns or --nostream to disable streaming
* -nc or --nocache to disable caching
* -m or --model to specify a model name

To run with a local model via ollama, do this:
```
ollama run dolphin-mixtral # best model for this script

python3 examples/data-qa/table_chat.py -m ollama/dolphin-mixtral:latest
```

For more info on running Langroid with local LLM, see here:
https://langroid.github.io/langroid/tutorials/local-llm-setup/
"""

import typer
from rich import print
from rich.prompt import Prompt

from langroid.agent.special.table_chat_agent import TableChatAgent, TableChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global

app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )

    print("[blue]Welcome to the tabular-data chatbot!\n")
    path = Prompt.ask(
        "[blue]Enter a local path or URL to a tabular dataset (hit enter to use default)\n",
        default="https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv",
    )

    agent = TableChatAgent(
        config=TableChatAgentConfig(
            data=path,
            llm=OpenAIGPTConfig(
                chat_model=model or OpenAIChatModel.GPT4o,
                chat_context_length=16_000,  # adjust based on model
                timeout=45,
                temperature=0.2,
            ),
        )
    )
    task = Task(agent, interactive=True)
    task.run("Can you help me with some questions about a tabular dataset?")


if __name__ == "__main__":
    app()
