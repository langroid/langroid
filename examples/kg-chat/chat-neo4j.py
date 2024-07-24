"""
Single-agent to use to chat with an existing Neo4j knowledge-graph (KG) on cloud,
or locally.
If you have an existing Neo4j db on Aura (or possibly elsewhere, e.g. locally), you can
chat with it by specifying its URI, username, password, and database name in the dialog.

If you don't have an existing Neo4j db, and want to try this script, you can populate
an empty Neo4j db with the cypher queries in the file `movies.cypher` in this folder.

See info on getting setup with Neo4j here:
 `https://github.com/langroid/langroid/blob/main/examples/kg-chat/README.md`

Run like this:
```
python3 examples/kg-chat/chat-neo4j.py
```
"""

import typer
import os
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.prompt import Prompt

import langroid.language_models as lm
from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jSettings,
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
)
from langroid.utils.constants import SEND_TO
from langroid.agent.task import Task
from langroid.utils.configuration import Settings, set_global


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
        default="neo4j+s://8d105d28.databases.neo4j.io",
    )
    username = Prompt.ask(
        "No4j username ",
        default="neo4j",
    )
    db = Prompt.ask(
        "Neo4j database ",
        default="neo4j",
    )
    pw = Prompt.ask(
        "Neo4j password ",
        default="",
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
        addressing_prefix=SEND_TO,
    )

    neo4j_agent = Neo4jChatAgent(neo4j_config)

    neo4j_task = Task(
        neo4j_agent,
        name="Neo4j",
        # user not awaited, UNLESS LLM explicitly addresses user via recipient_tool
        interactive=False,
    )

    neo4j_task.run()


if __name__ == "__main__":
    app()
