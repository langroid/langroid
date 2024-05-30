from typing import List, Optional, Tuple

import pandas as pd
import typer

from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
)
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.table_loader import read_tabular_data
from langroid.utils.output import status
from langroid.vector_store.base import VectorStoreConfig

app = typer.Typer()


BUILD_KG_INSTRUCTIONS = """
    Your task is to build a knowledge graph based on a CSV file. 
    
    You need to generate the graph database based on this
    header: 
    
    {header}
    
    and these sample rows: 
    
    {sample_rows}. 
    
    Leverage the above information to: 
    - Define node labels and their properties
    - Infer relationships
    - Infer constraints 
    ASK me if you need further information to figure out the schema.
    You can use the tool/function `pandas_to_kg` to display and confirm 
    the nodes and relationships.
"""

DEFAULT_CSV_KG_CHAT_SYSTEM_MESSAGE = """
    You are an expert in Knowledge Graphs and analyzing them using Neo4j.
    You will be asked to answer questions based on the knowledge graph.
"""


def _preprocess_dataframe_for_neo4j(
    df: pd.DataFrame, default_value: Optional[str] = None, remove_null_rows: bool = True
) -> pd.DataFrame:
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


class CSVGraphAgentConfig(Neo4jChatAgentConfig):
    system_message: str = DEFAULT_CSV_KG_CHAT_SYSTEM_MESSAGE
    data: str | pd.DataFrame | None  # data file, URL, or DataFrame
    separator: None | str = None  # separator for data file
    vecdb: None | VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        chat_model=OpenAIChatModel.GPT4_TURBO,
    )


class PandasToKGTool(ToolMessage):
    request: str = "pandas_to_kg"
    purpose: str = """Use this tool to create ONLY nodes and their relationships based
    on the created model.
    Take into account that the Cypher query will be executed while iterating 
    over the rows in the CSV file (e.g. `index, row in df.iterrows()`),
    so there NO NEED to load the CSV.
    Make sure you send me the cypher query in this format: 
    - placeholders in <cypherQuery> should be based on the CSV header. 
    - <args> an array wherein each element corresponds to a placeholder in the 
    <cypherQuery> and provided in the same order as the headers. 
    SO the <args> should be the result of: `[row_dict[header] for header in headers]`
    """
    cypherQuery: str
    args: list[str]

    @classmethod
    def examples(cls) -> List["ToolMessage" | Tuple[str, "ToolMessage"]]:
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


class CSVGraphAgent(Neo4jChatAgent):
    def __init__(self, config: CSVGraphAgentConfig):
        formatted_build_instr = ""
        if isinstance(config.data, pd.DataFrame):
            df = config.data
            self.df = df
        else:
            if config.data:
                df = read_tabular_data(config.data, config.separator)
                df_cleaned = _preprocess_dataframe_for_neo4j(df)

                df_cleaned.columns = df_cleaned.columns.str.strip().str.replace(
                    " +", "_", regex=True
                )

                self.df = df_cleaned

                formatted_build_instr = BUILD_KG_INSTRUCTIONS.format(
                    header=self.df.columns, sample_rows=self.df.head(3)
                )

        config.system_message = config.system_message + formatted_build_instr
        super().__init__(config)

        self.config: Neo4jChatAgentConfig = config

        self.enable_message(PandasToKGTool)

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
        with status("[cyan]Generating graph database..."):
            if self.df is not None and hasattr(self.df, "iterrows"):
                for counter, (index, row) in enumerate(self.df.iterrows()):
                    row_dict = row.to_dict()
                    response = self.write_query(
                        msg.cypherQuery,
                        parameters={header: row_dict[header] for header in msg.args},
                    )
                    # there is a possibility the generated cypher query is not correct
                    # so we need to check the response before continuing to the
                    # iteration
                    if counter == 0 and not response.success:
                        return str(response.data)
            return "Graph database successfully generated"
