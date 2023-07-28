"""
Agent that allows interaction with an SQL database using SQLAlchemy library. 
The agent can execute SQL queries in the database and return the result. 

Functionality includes:
- adding table and column context
- asking a question about a SQL schema
"""
import logging
from typing import Dict, Optional, Union

from prettytable import PrettyTable
from rich.console import Console
from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.vector_store.base import VectorStoreConfig

logger = logging.getLogger(__name__)

console = Console()

DEFAULT_TABLE_CHAT_SYSTEM_MESSAGE = """
You are a savvy data scientist, with expertise in analyzing SQL database,
using Python and the SQLAlchemy library for data manipulation.
You do not have access to the database 'db' directly, so you will need to use the 
`run_code` tool/function-call to answer the question.
The tables in the database are:
{tables}
Please note that ONLY the column names specified above should be used in the queries.
Avoid assuming any tables or columns other than those above.
"""


class SQLChatAgentConfig(ChatAgentConfig):
    system_message: str = DEFAULT_TABLE_CHAT_SYSTEM_MESSAGE
    user_message: None | str = None
    max_context_tokens: int = 1000
    cache: bool = True  # cache results
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    database_uri: str  # Database URI
    vecdb: None | VectorStoreConfig = None
    context_descriptions: Optional[
        Dict[str, Dict[str, Union[str, Dict[str, str]]]]
    ] = None

    """
    Optional, but strongly recommended, context descriptions for tables, columns, 
    and relationships. It should be a dictionary where each key is a table name 
    and its value is another dictionary. 

    In this inner dictionary:
    - The 'description' key corresponds to a string description of the table.
    - The 'columns' key corresponds to another dictionary where each key is a 
    column name and its value is a string description of that column.
    - The 'relationships' key corresponds to another dictionary where each key 
    is another table name and the value is a description of the relationship to 
    that table.

    For example:
    {
        'table1': {
            'description': 'description of table1',
            'relationships': {
                'table2': 'description of relationship to table2'
            },
            'columns': {
                'column1': 'description of column1 in table1',
                'column2': 'description of column2 in table1'
            }
        },
        'table2': {
            'description': 'description of table2',
            'columns': {
                'column3': 'description of column3 in table2',
                'column4': 'description of column4 in table2'
            }
        }
    }
    """

    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        completion_model=OpenAIChatModel.GPT4,
    )
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


class RunCodeTool(ToolMessage):
    request: str = "run_code"
    purpose: str = """
            To run <code> on the database 'db' and 
            return the results to answer a question.
            """
    code: str


class SQLChatAgent(ChatAgent):
    """
    Agent for chatting with a collection of documents.
    """

    def __init__(self, config: SQLChatAgentConfig):
        super().__init__(config)
        self.config: SQLChatAgentConfig = config
        self.engine = create_engine(config.database_uri)
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
        self.metadata.reflect(self.engine)

        # User-provided context descriptions
        context_descriptions = config.context_descriptions

        logger.info(
            f"""SQLChatAgent initialized with database: 
            {self.engine} and tables: 
            {self.metadata.tables}
            """
        )

        # Create inspector
        inspector = inspect(self.engine)

        # Gather table and column information
        tables_info = ""
        # Iterate over all tables present in the metadata
        for table_name in self.metadata.tables.keys():
            table_description: str = ""
            relationships: Dict[str, str] = {}

            # Check if context_descriptions are provided
            if context_descriptions is not None:
                # Fetch the table_info from context_descriptions
                table_info = context_descriptions.get(table_name, {})
                if isinstance(table_info, dict):
                    # Get description of the table
                    description = table_info.get("description", "")
                    assert isinstance(description, str)
                    table_description = description

                    # Get the relationships of the table with other tables
                    rels = table_info.get("relationships", {})
                    assert isinstance(rels, dict)
                    relationships = rels

            # Append table_name and table_description to tables_info
            if table_description:
                tables_info += (
                    f"\nTable: {table_name}\nDescription: {table_description}\n"
                )
            else:
                tables_info += f"\nTable: {table_name}\n"

            # If table has relationships, append them to tables_info
            if relationships:
                tables_info += "Relationships:\n"
                for related_table, description in relationships.items():
                    tables_info += f"- {related_table}: {description}\n"

            # Get column details for the table
            columns = inspector.get_columns(table_name)
            table = PrettyTable()
            table.field_names = ["Column Name", "Type", "Description"]

            # Loop over all columns to populate the table
            for column in columns:
                description = ""  # Initialize description as empty string
                if context_descriptions is not None:
                    table_info = context_descriptions.get(table_name, {})
                    if isinstance(table_info, dict):
                        columns_info = table_info.get("columns", {})
                        # Fetch column description from context_descriptions
                        if isinstance(columns_info, dict):
                            description = columns_info.get(column["name"], "")
                # Add a new row in the table for each column
                table.add_row([column["name"], column["type"], description])
            # Append the generated table to tables_info
            tables_info += f"{table}\n"

        # Update the system message with the table information
        self.config.system_message = self.config.system_message.format(
            tables=tables_info
        )

        # Enable the agent to use and handle the RunCodeTool
        self.enable_message(RunCodeTool)

    def run_code(self, msg: RunCodeTool) -> str:
        """
        Handle a RunCodeTool message by executing a SQL query and returning the result.

        Args:
            msg (RunCodeTool): The tool-message to handle.

        Returns:
            str: The result of executing the SQL query.
        """
        code = msg.code
        session = self.Session()
        result = ""

        try:
            # Execute the SQL query
            query_result = session.execute(text(code))
            # Commit the transaction
            session.commit()

            try:
                # Fetch all the rows from the result
                rows = query_result.fetchall()

                # Check if the list of rows is not empty
                if rows:
                    result = ", ".join(str(row) for row in rows)
                else:
                    result = "Query executed successfully."
            except Exception:
                result = "Query executed successfully but does not return any rows."
        except Exception as e:
            # In case of exception, rollback the transaction
            session.rollback()
            result = f"Error occurred: {str(e)}"
        finally:
            # Always close the session
            session.close()

        return result
