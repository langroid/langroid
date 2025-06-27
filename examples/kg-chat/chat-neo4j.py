"""
Single-agent to use to chat with an existing Neo4j knowledge-graph (KG) on cloud,
or locally.
If you have an existing Neo4j db on Aura (or possibly elsewhere, e.g. locally), you can
chat with it by specifying its URI, username, password, and database name in the dialog.

You can chose the defaults in the dialog, in which case it will use the
freely available Movies database.

Or,  you can populate
an empty Neo4j db with the cypher queries in the file `movies.cypher` in this folder.

See info on getting setup with Neo4j here:
 `https://github.com/langroid/langroid/blob/main/examples/kg-chat/README.md`

Run like this:
```
python3 examples/kg-chat/chat-neo4j.py
```
"""

import os

import typer
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.prompt import Prompt

import langroid.language_models as lm
from langroid import TaskConfig
from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
    Neo4jSettings,
)
from langroid.agent.task import Task
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import SEND_TO

console = Console()
app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=nocache,
            stream=not no_stream,
        )
    )
    print(
        """
        [blue]Welcome to Neo4j Knowledge Graph RAG chatbot!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    uri = Prompt.ask(
        "Neo4j URI ",
        default="neo4j+s://demo.neo4jlabs.com",
    )
    username = Prompt.ask(
        "No4j username ",
        default="movies",
    )
    db = Prompt.ask(
        "Neo4j database ",
        default="movies",
    )
    pw = Prompt.ask(
        "Neo4j password ",
        default="movies",
    )
    pw = pw or os.getenv("NEO4J_PASSWORD")
    neo4j_settings = Neo4jSettings(uri=uri, username=username, database=db, password=pw)

    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=128_000,
    )
    neo4j_config = Neo4jChatAgentConfig(
        neo4j_settings=neo4j_settings,
        llm=llm_config,
        chat_mode=True,
    )

    neo4j_agent = Neo4jChatAgent(neo4j_config)
    task_config = TaskConfig(addressing_prefix=SEND_TO)
    neo4j_task = Task(
        neo4j_agent,
        name="Neo4j",
        # user not awaited, UNLESS LLM explicitly addresses user via recipient_tool
        interactive=False,
        config=task_config,
    )

    neo4j_task.run()


if __name__ == "__main__":
    app()
