import typer
from typing import List
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich import print
from pandas import DataFrame, read_csv

from langroid.agent.tool_message import ToolMessage
from langroid.agent.task import Task
from langroid.utils.configuration import set_global, Settings
from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
    Neo4jSettings,
)

console = Console()
app = typer.Typer()


def _load_csv_dataset(csv_location) -> DataFrame:
    """
    Load a CSV dataset from a given file path or URL.

    This function reads a CSV file from the specified path or URL into a pandas
        DataFrame. It handles exceptions that may occur during file reading.

    Returns:
    DataFrame: A DataFrame containing the data loaded from the CSV file.

    Raises:
    Exception: An error occurred while trying to read the CSV file.

    Note:
    The function assumes the CSV file has a header row.
    """
    try:
        df = read_csv(csv_location)
        return df
    except Exception as e:
        raise Exception(f"Error occurred while reading the CSV file: {e}")


def _preprocess_dataframe_for_neo4j(
    df: DataFrame, default_value: str = None, remove_null_rows: bool = True
) -> DataFrame:
    """
    Preprocess a DataFrame for Neo4j import by fixing mismatched quotes in string
        columns and handling null or missing values.

    Args:
        df (DataFrame): The DataFrame to be preprocessed.
        default_value (str, optional): The default value to replace null values.
        This is ignored if remove_null_rows is True. Defaults to None.
        remove_null_rows (bool, optional): If True, rows with any null values will
            be removed.
        If False, null values will be filled with default_value. Defaults to False.

    Returns:
        DataFrame: The preprocessed DataFrame ready for Neo4j import.
    """

    # Fix mismatched quotes in string columns
    for column in df.select_dtypes(include=["object"]):
        df[column] = df[column].apply(
            lambda x: x + '"' if (isinstance(x, str) and x.count('"') % 2 != 0) else x
        )

    # Handle null or missing values
    if remove_null_rows:
        df = df.dropna()
    else:
        if default_value is not None:
            df = df.fillna(default_value)

    return df


class CSVNodeGenerator(ToolMessage):
    request: str = "create_nodes_and_relationships_from_csv"
    purpose: str = """Use this tool to create ONLY nodes and their relashipnships based
    on the created model.
    Take into account that the Cypher query will be executed while iterating the rows in
    the CSV file (like this `index, row in df.iterrows()`). So NO NEED to load the CSV.
    Make sure you send me the cypher query in this format: 
    - placehoders in <cypherQuery> should be based on the CSV header. 
    - <args> an array wherein each element corresponds to every placeholder in the 
    <cypherQuery> and provided in the same order as the headers. 
    SO the <args> should be like this: `[row_dict[header] for header in headers]`
    """
    cypherQuery: str
    args: list[str]

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                cypherQuery="""MERGE (employee:Employee {name: $employeeName, id: $employeeId})\n
                MERGE (department:Department {name: $departmentName})\n
                MERGE (employee)-[:WORKS_IN]->(department)\n
                SET employee.email = $employeeEmail""",
                args=["employeeName", "employeeId", "departmentName", "employeeEmail"],
            ),
        ]


class CSVChatGraphAgent(Neo4jChatAgent):
    def __init__(self, config: Neo4jChatAgentConfig):
        super().__init__(config)
        self.config: Neo4jChatAgentConfig = config
        self.csv_location: None | str = None
        self.csv_dataframe: None | DataFrame = None

    def create_nodes_and_relationships_from_csv(self, msg: CSVNodeGenerator) -> str:
        """
        Creates nodes and relationships in the graph database based on the data in
        a CSV file.

        Args:
            msg (CSVNodeGenerator): An instance of the CSVNodeGenerator class containing
                the necessary information for generating nodes.

        Returns:
            str: A string indicating the success or failure of the operation.
        """
        response = ""
        with console.status("[cyan]Generating graph database..."):
            for index, row in self.csv_dataframe.iterrows():
                row_dict = row.to_dict()
                response = self.write_query(
                    msg.cypherQuery,
                    parameters={header: row_dict[header] for header in msg.args},
                )
                # there is a possibility the generated cypher query is not correct
                # so we need to check the response before continuing to the iteration
                if index == 0 and "successfully" not in response:
                    print(f"[red]{response}")
                    return response
        return "Graph database successfully generated"


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
                    You can use the tool/function
                    `create_nodes_and_relationships_from_csv` to display and confirm 
                    the nodes and relationships.
                """

            if user_input == "n":
                print("Quitting the chatbot...")
                return

    csv_kg_chat_agent.enable_message(CSVNodeGenerator)

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
