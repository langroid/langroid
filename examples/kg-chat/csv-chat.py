import typer
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.prompt import Prompt

from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgentConfig,
    Neo4jSettings,
)
from langroid.agent.special.neo4j.csv_kg_chat import (
    CSVChatGraphAgent,
    PandasToKGTool,
    _load_csv_dataset,
    _preprocess_dataframe_for_neo4j,
)
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
    cache_type: str = typer.Option(
        "redis", "--cachetype", "-ct", help="redis or momento"
    ),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=nocache,
            stream=not no_stream,
            cache_type=cache_type,
        )
    )
    print(
        """
        [blue]Welcome to CSV Knowledge Graph RAG chatbot!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    neo4j_settings = Neo4jSettings()

    csv_kg_chat_agent = CSVChatGraphAgent(
        config=Neo4jChatAgentConfig(
            neo4j_settings=neo4j_settings,
            use_tools=tools,
            use_functions_api=not tools,
            llm=OpenAIGPTConfig(
                chat_model=model or OpenAIChatModel.GPT4_TURBO,
            ),
        ),
    )

    build_kg_instructions = ""

    buid_kg = Prompt.ask(
        "Do you want to build the graph database from a CSV file? (y/n)",
    )
    if buid_kg == "y":
        csv_location = Prompt.ask(
            "Please provide the path/URL to the CSV",
        )

        csv_dataframe = _load_csv_dataset(csv_location)
        headers = csv_dataframe.columns.tolist()

        # clean the CSV file before loading it into Neo4j
        csv_dataframe = _preprocess_dataframe_for_neo4j(csv_dataframe)

        num_rows = len(csv_dataframe)

        if num_rows > 1000:
            print(
                f"""
                [red]WARNING: The CSV file has {num_rows} rows. Loading this data and 
                generating the graph database will take long time.
                """
            )
            user_input = Prompt.ask(
                "Do you want to continue? (y/n)",
            )
            if user_input == "y":
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
                    csv_dataframe = csv_dataframe.sample(n=sample_size)

                csv_kg_chat_agent.csv_dataframe = csv_dataframe
                csv_kg_chat_agent.csv_location = csv_location
                build_kg_instructions = f"""
                    Your task is to build a knowledge graph based on a CSV file. 
                    
                    You need to generate the graph database based on these
                    headers: {headers} in the CSV file.
                    You can use the tool/function `pandas_to_kg` to display and confirm 
                    the nodes and relationships.
                """

            if user_input == "n":
                print("Quitting the chatbot...")
                return

    csv_kg_chat_agent.enable_message(PandasToKGTool)

    system_message = f"""
    You are an expert in Knowledge Graphs and analyzing them using Neo4j.
    You will be asked to answer questions based on the knowledge graph.
    {build_kg_instructions}.
    """

    csv_kg_chat_task = Task(
        csv_kg_chat_agent,
        name="CSVChatKG",
        system_message=system_message,
    )

    csv_kg_chat_task.run()


if __name__ == "__main__":
    app()
