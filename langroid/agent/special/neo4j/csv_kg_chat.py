from typing import List, Optional

import typer
from pandas import DataFrame, read_csv
from rich import print
from rich.console import Console

from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
)
from langroid.agent.tool_message import ToolMessage

console = Console()
app = typer.Typer()


def _load_csv_dataset(csv_location: str) -> DataFrame:
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
    df: DataFrame, default_value: Optional[str] = None, remove_null_rows: bool = True
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


class PandasToKGTool(ToolMessage):
    request: str = "pandas_to_kg"
    purpose: str = """Use this tool to create ONLY nodes and their relationships based
    on the created model.
    Take into account that the Cypher query will be executed while iterating 
    over the rows in the CSV file (e.g. `index, row in df.iterrows()`),
    so there NO NEED to load the CSV.
    Make sure you send me the cypher query in this format: 
    - placehoders in <cypherQuery> should be based on the CSV header. 
    - <args> an array wherein each element corresponds to a placeholder in the 
    <cypherQuery> and provided in the same order as the headers. 
    SO the <args> should be the result of: `[row_dict[header] for header in headers]`
    """
    cypherQuery: str
    args: list[str]

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                cypherQuery="""MERGE (employee:Employee {name: $employeeName, 
                id: $employeeId})\n
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

    def pandas_to_kg(self, msg: PandasToKGTool) -> str:
        """
        Creates nodes and relationships in the graph database based on the data in
        a CSV file.

        Args:
            msg (PandasToKGTool): An instance of the PandasToKGTool class containing
                the necessary information for generating nodes.

        Returns:
            str: A string indicating the success or failure of the operation.
        """
        with console.status("[cyan]Generating graph database..."):
            if self.csv_dataframe is not None and hasattr(
                self.csv_dataframe, "iterrows"
            ):
                for index, row in self.csv_dataframe.iterrows():
                    row_dict = row.to_dict()
                    response = self.write_query(
                        msg.cypherQuery,
                        parameters={header: row_dict[header] for header in msg.args},
                    )
                    # there is a possibility the generated cypher query is not correct
                    # so we need to check the response before continuing to the
                    # iteration
                    if index == 0 and "successfully" not in response:
                        print(f"[red]{response}")
                        return response
            return "Graph database successfully generated"
