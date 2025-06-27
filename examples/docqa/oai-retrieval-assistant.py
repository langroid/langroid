"""
Use OpenAI Assistant with Retrieval tool + file to answer questions.

Run like this:
python3 examples/docqa/oai-retrieval-assistant.py

"""

import os
import tempfile

import typer
from rich import print
from rich.prompt import Prompt

from langroid.agent.openai_assistant import (
    AssistantTool,
    OpenAIAssistant,
    OpenAIAssistantConfig,
)
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.url_loader import URLLoader
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@app.command()
def chat() -> None:
    reuse = (
        Prompt.ask(
            "Reuse existing assistant, threads if available? (y/n)",
            default="y",
        )
        == "y"
    )

    cfg = OpenAIAssistantConfig(
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4o),
        use_cached_assistant=reuse,
        use_cached_thread=reuse,
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
    agent.add_assistant_tools([AssistantTool(type="retrieval")])

    if path:  # may be empty if continuing from previous session
        agent.add_assistant_files([path])

    print("[cyan]Enter x or q to quit")

    task = Task(agent)

    task.run("Please help me with questions about the document I provided")


if __name__ == "__main__":
    app()
