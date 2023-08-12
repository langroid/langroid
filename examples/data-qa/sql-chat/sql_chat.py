"""
Example showing how to chat with a SQL database
"""
import typer
from rich.prompt import Prompt
from rich import print

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from prettytable import PrettyTable

from langroid.agent.special.sql.sql_chat_agent import (
    SQLChatAgent,
    SQLChatAgentConfig,
)
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

from typing import Dict, Any
import json
import os

app = typer.Typer()

setup_colored_logging()


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


def chat() -> None:
    print("[blue]Welcome to the SQL database chatbot!\n")
    database_uri = Prompt.ask(
        "[blue]Enter the URI for your SQL database (hit enter for default)",
        default="sqlite:///examples/data-qa/sql-chat/demo.db",
    )

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

    agent = SQLChatAgent(
        config=SQLChatAgentConfig(
            database_uri=database_uri,
            use_tools=True,
            use_functions_api=False,
            context_descriptions=context_descriptions,  # Add context descriptions to the config
            llm=OpenAIGPTConfig(
                chat_model=OpenAIChatModel.GPT4,
            ),
        )
    )
    task = Task(agent)
    task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    cache_type: str = typer.Option(
        "redis", "--cachetype", "-ct", help="redis or momento"
    ),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
            cache_type=cache_type,
        )
    )
    chat()


if __name__ == "__main__":
    app()
