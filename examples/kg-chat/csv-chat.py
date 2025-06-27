"""
Example showing how to chat with a graph database generated from
csv, tsv, or any other pandas-readable.

This example will automatically generate all the required Cypher queries for Neo4j
to answer user's questions.

This example relies on neo4j. The easiest way to get access to neo4j is by
creating a cloud account at `https://neo4j.com/cloud/platform/aura-graph-database/`

Upon creating the account successfully, neo4j will create a text file that contains
account settings, please provide the following information (uri, username, password) as
described here
`https://github.com/langroid/langroid/tree/main/examples/kg-chat#requirements`

Run like this

python3 examples/kg-chat/csv-chat.py

Optional args:
* -d or --debug to enable debug mode
* -ns or --nostream to disable streaming
* -nc or --nocache to disable caching
* -m or --model to specify a model name

"""

import typer
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.prompt import Prompt

from langroid.agent.special.neo4j.csv_kg_chat import (
    CSVGraphAgent,
    CSVGraphAgentConfig,
)
from langroid.agent.special.neo4j.neo4j_chat_agent import Neo4jSettings
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global

console = Console()
app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    tools: bool = typer.Option(
        False, "--tools", "-t", help="use langroid tools instead of function-calling"
    ),
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
        [blue]Welcome to CSV Knowledge Graph RAG chatbot!
        Enter x or q to quit at any point.
        """
    )

    buid_kg = Prompt.ask(
        "Do you want to build the graph database from a CSV file? (y/n)",
        default="y",
    )
    if buid_kg == "y":
        csv_location = Prompt.ask(
            "Please provide the path/URL to the CSV",
            default="examples/docqa/data/imdb-drama.csv",
        )
    else:
        csv_location = None

    load_dotenv()

    neo4j_settings = Neo4jSettings()

    csv_kg_chat_agent = CSVGraphAgent(
        config=CSVGraphAgentConfig(
            data=csv_location,
            neo4j_settings=neo4j_settings,
            use_tools=tools,
            use_functions_api=not tools,
            llm=OpenAIGPTConfig(
                chat_model=model or OpenAIChatModel.GPT4o,
                chat_context_length=16_000,  # adjust based on model
                timeout=45,
                temperature=0.2,
            ),
        ),
    )

    if buid_kg == "y":
        num_rows = len(csv_kg_chat_agent.df)

        if num_rows > 1000:
            print(
                f"""
                [red]WARNING: The CSV file has {num_rows} rows. Loading this data and 
                generating the graph database will take long time.
                """
            )

            user_input_continue = Prompt.ask(
                "Do you want to continue with the whole dataset? (y/n)",
            )
            if user_input_continue == "n":
                sample_size = int(
                    Prompt.ask(
                        "Please enter the sample size",
                    )
                )
                print(
                    f"""
                    [green]The graph database will be generated for {sample_size} 
                    rows...
                    """
                )
                csv_kg_chat_agent.df = csv_kg_chat_agent.df.sample(n=sample_size)

            elif user_input_continue == "y":
                print(
                    """
                    [green]The graph database will be generated for the whole dataset...
                    """
                )

    csv_kg_chat_task = Task(
        csv_kg_chat_agent,
        name="CSVChatKG",
        interactive=True,
    )

    csv_kg_chat_task.run()


if __name__ == "__main__":
    app()
