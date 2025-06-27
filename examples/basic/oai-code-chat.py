"""
The most basic use of code-interpreter, using an OpenAssistant agent,
powered by the OpenAI Assistant API's code-interpreter tool.

Run like this:

python3 examples/basic/oai-code-chat.py
"""

import tempfile

import typer
from dotenv import load_dotenv
from rich import print
from rich.prompt import Prompt

from langroid.agent.openai_assistant import (
    AssistantTool,
    OpenAIAssistant,
    OpenAIAssistantConfig,
    ToolType,
)
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.url_loader import URLLoader
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()


@app.command()
def chat() -> None:
    print(
        """
        [blue]Welcome to the basic chatbot!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    default_sys_msg = "You are a helpful assistant. Be concise in your answers."

    sys_msg = Prompt.ask(
        "[blue]Tell me who I am. Hit Enter for default, or type your own\n",
        default=default_sys_msg,
    )

    path = Prompt.ask("Enter a URL or file path, or hit enter if no files")
    if path:
        # if path is a url, use UrlLoader to get text as a document
        if path.startswith("http"):
            text = URLLoader([path]).load()[0].content
            # save text to a temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(text)
                f.close()
                # get the filename
                path = f.name

    config = OpenAIAssistantConfig(
        system_message=sys_msg,
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4o),
    )
    agent = OpenAIAssistant(config)
    agent.add_assistant_tools([AssistantTool(type=ToolType.CODE_INTERPRETER)])
    if path:
        agent.add_assistant_files([path])

    task = Task(agent)

    task.run(
        """
        Help me with some questions, 
        using the CODE INTERPRETER tool, and any uploaded files as needed.
        """
    )


if __name__ == "__main__":
    app()
