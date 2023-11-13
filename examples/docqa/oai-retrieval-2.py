"""
Use TWO OpenAI Assistants in Langroid's Multi-Agent mode to answer questions:
 - Planner Agent: takes user question, plans, decides how to ask the Retrieval Agent
 - Retrieval Agent: takes the question from the Master Agent, answers based on docs

Run like this:
python3 examples/docqa/oai-retrieval-2.py

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
from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.agent.task import Task
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CLIOptions(BaseSettings):
    debug: bool = False
    cache: bool = True
    tool: bool = False


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    tool: bool = typer.Option(
        False, "--tool", "-t", help="use Langroid ToolMessage instead of fn-call"
    ),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    cli_opts = CLIOptions(
        tool=tool,
        debug=debug,
        cache=not nocache,
    )

    chat(cli_opts)


def chat(opts: CLIOptions) -> None:
    set_global(Settings(debug=opts.debug, cache=opts.cache))

    planner_cfg = OpenAIAssistantConfig(
        name="Planner",
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4_TURBO),
        use_cached_assistant=False,
        use_cached_thread=False,
        use_functions_api=not opts.tool,
        use_tools=opts.tool,
        system_message="""
        You will receive questions from the user about some docs, 
        but you don't have access to them, but you have a Retriever to help you, since
        they have access to the docs. For each question I send you, decide how you want 
        to ask the Retriever: you can rephrase, decompose or simplify the question and 
        send it to the retriever. Once you think you have the info I need, then send 
        me (the User) a message with your answer.    
        
        Start by greeting the user and asking what they want to know.     
        """,
    )
    planner_agent = OpenAIAssistant(planner_cfg)
    planner_agent.enable_message(RecipientTool)

    retriever_cfg = OpenAIAssistantConfig(
        name="Retriever",
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4_TURBO),
        use_cached_assistant=False,
        use_cached_thread=False,
        system_message="Answer questions based on the documents provided.",
    )

    retriever_agent = OpenAIAssistant(retriever_cfg)

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
    retriever_agent.add_assistant_tools([AssitantTool(type="retrieval")])
    retriever_agent.add_assistant_files([path])

    print("[cyan]Enter x or q to quit")

    planner_task = Task(
        planner_agent,
        llm_delegate=True,
        single_round=False,
    )
    retriever_task = Task(
        retriever_agent,
        llm_delegate=False,
        single_round=True,
    )
    planner_task.add_sub_task(retriever_task)
    planner_task.run("")


if __name__ == "__main__":
    app()
