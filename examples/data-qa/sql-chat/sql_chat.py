"""
Example showing how to chat with a SQL database.

Note if you are using this with a postgres db, you will need to:

(a) Install PostgreSQL dev libraries for your platform, e.g.
    - `sudo apt-get install libpq-dev` on Ubuntu,
    - `brew install postgresql` on Mac, etc.
(b) langroid with the postgres extra, e.g. `pip install langroid[postgres]`
    or `poetry add langroid[postgres]` or `poetry install -E postgres`
    or `uv pip install langroid[postgres]` or `uv add langroid[postgres]`.
    If this gives you an error, try `pip install psycopg2-binary` in your virtualenv.
"""

import json
import os
from typing import Any, Dict

import typer
from rich import print
from rich.prompt import Prompt

from langroid.exceptions import LangroidImportError

try:
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.engine import Engine
except ImportError as e:
    raise LangroidImportError(extra="sql", error=str(e))

from prettytable import PrettyTable

try:
    from .utils import fix_uri, get_database_uri
except ImportError:
    from utils import fix_uri, get_database_uri
import logging

from langroid.agent.special.sql.sql_chat_agent import (
    SQLChatAgent,
    SQLChatAgentConfig,
)
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import SEND_TO

logger = logging.getLogger(__name__)


app = typer.Typer()


def create_descriptions_file(filepath: str, engine: Engine) -> None:
    """
    Create an empty descriptions JSON file for SQLAlchemy tables.

    This function inspects the database, generates a template for table and
    column descriptions, and writes that template to a new JSON file.

    Args:
        filepath: The path to the file where the descriptions should be written.
        engine: The SQLAlchemy Engine connected to the database to describe.

    Raises:
        FileExistsError: If the file at `filepath` already exists.

    Returns:
        None
    """
    if os.path.exists(filepath):
        raise FileExistsError(f"File {filepath} already exists.")

    inspector = inspect(engine)
    descriptions: Dict[str, Dict[str, Any]] = {}

    for table_name in inspector.get_table_names():
        descriptions[table_name] = {
            "description": "",
            "columns": {col["name"]: "" for col in inspector.get_columns(table_name)},
        }

    with open(filepath, "w") as f:
        json.dump(descriptions, f, indent=4)


def load_context_descriptions(engine: Engine) -> dict:
    """
    Ask the user for a path to a JSON file and load context descriptions from it.

    Returns:
        dict: The context descriptions, or an empty dictionary if the user decides to skip this step.
    """

    while True:
        filepath = Prompt.ask(
            "[blue]Enter the path to your context descriptions file. \n"
            "('n' to create a NEW file, 's' to SKIP, or Hit enter to use DEFAULT) ",
            default="examples/data-qa/sql-chat/demo.json",
        )

        if filepath.strip() == "s":
            return {}

        if filepath.strip() == "n":
            filepath = Prompt.ask(
                "[blue]To create a new context description file, enter the path",
                default="examples/data-qa/sql-chat/description.json",
            )
            print(f"[blue]Creating new context description file at {filepath}...")
            create_descriptions_file(filepath, engine)
            print(
                f"[blue] Please fill in the descriptions in {filepath}, "
                f"then try again."
            )

        # Try to load the file
        if not os.path.exists(filepath):
            print(f"[red]The file '{filepath}' does not exist. Please try again.")
            continue

        try:
            with open(filepath, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(
                f"[red]The file '{filepath}' is not a valid JSON file. Please try again."
            )


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    tools: bool = typer.Option(
        False, "--tools", "-t", help="use langroid tools instead of function-calling"
    ),
    schema_tools: bool = typer.Option(
        False, "--schema_tools", "-st", help="use schema tools"
    ),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
            cache_type="redis",
        )
    )
    print("[blue]Welcome to the SQL database chatbot!\n")
    database_uri = Prompt.ask(
        """
        [blue]Enter the URI for your SQL database 
        (type 'i' for interactive, or hit enter for default)
        """,
        default="sqlite:///examples/data-qa/sql-chat/demo.db",
    )

    if database_uri == "i":
        database_uri = get_database_uri()

    database_uri = fix_uri(database_uri)
    logger.warning(f"Using database URI: {database_uri}")

    # Create engine and inspector
    engine = create_engine(database_uri)
    inspector = inspect(engine)

    context_descriptions = load_context_descriptions(engine)

    # Get table names
    table_names = inspector.get_table_names()

    for table_name in table_names:
        print(f"[blue]Table: {table_name}")

        # Create a new table for the columns
        table = PrettyTable()
        table.field_names = ["Column Name", "Type"]

        # Get the columns for the table
        columns = inspector.get_columns(table_name)
        for column in columns:
            table.add_row([column["name"], column["type"]])

        print(table)

    agent_config = SQLChatAgentConfig(
        name="sql",
        database_uri=database_uri,
        use_tools=tools,
        use_functions_api=not tools,
        show_stats=False,
        chat_mode=True,
        use_helper=True,
        context_descriptions=context_descriptions,  # Add context descriptions to the config
        use_schema_tools=schema_tools,
        addressing_prefix=SEND_TO,
        llm=OpenAIGPTConfig(
            chat_model=OpenAIChatModel.GPT4o,
        ),
    )
    agent = SQLChatAgent(agent_config)
    # Set interactive = False, but we user gets chance to respond
    # when explicitly addressed by LLM
    task = Task(agent, interactive=False)
    task.run()


if __name__ == "__main__":
    app()
