"""
Use OpenAI Assistant with Retrieval tool + file to answer questions.

Run like this:
python3 examples/docqa/oai-retrieval-assistant.py

"""
import typer
from rich import print
from rich.prompt import Prompt
import os
from pydantic import BaseSettings
import tempfile

from langroid.agent.openai_assistant import (
    OpenAIAssistantConfig,
    OpenAIAssistant,
    AssitantTool,
)
from langroid.parsing.url_loader import URLLoader
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.agent.task import Task
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CLIOptions(BaseSettings):
    debug: bool = False
    cache: bool = True


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    cli_opts = CLIOptions(
        debug=debug,
        cache=not nocache,
    )

    chat(cli_opts)


def chat(opts: CLIOptions) -> None:
    set_global(Settings(debug=opts.debug, cache=opts.cache))

    cfg = OpenAIAssistantConfig(
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4_TURBO),
        use_cached_assistant=False,
        use_cached_thread=False,
        system_message="Answer questions based on the provided document.",
    )
    agent = OpenAIAssistant(cfg)

    print("[blue]Welcome to the retrieval chatbot!")
    path = Prompt.ask("Enter a URL or file path")
    # if path is a url, use UrlLoader to get text as a document
    if path.startswith("http"):
        text = URLLoader([path]).load()[0].content
        # save text to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            f.close()
            # get the filename
            path = f.name
    agent.add_assistant_tools([AssitantTool(type="retrieval")])
    agent.add_assistant_files([path])

    print("[cyan]Enter x or q to quit")

    task = Task(
        agent,
        llm_delegate=False,
        single_round=False,
    )
    task.run("Please help me with questions about the document I provided")


if __name__ == "__main__":
    app()
