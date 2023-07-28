"""
Agent that allows interaction with an SQL database using SQLAlchemy library. 
The agent can execute SQL queries in the database and return the result. 

Functionality includes:
- adding table and column context
- asking a question about a SQL schema
"""
import logging
from typing import Any, Dict, Optional, Union

from prettytable import PrettyTable
from rich import print
from rich.console import Console
from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import Inspector
from sqlalchemy.orm import sessionmaker

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.prompts.prompts_config import PromptsConfig
from langroid.vector_store.base import VectorStoreConfig

logger = logging.getLogger(__name__)

console = Console()

DEFAULT_SQL_CHAT_SYSTEM_MESSAGE = """
You are a savvy data scientist, with expertise in analyzing SQL database,
using Python and the SQLAlchemy library for data manipulation.
You do not have access to the database 'db' directly, so you will need to use the 
`run_query` tool/function-call to answer the question.
Here is information about the tables, columns and their relationships:
{tables}
Please note that ONLY the column names specified above should be used in the queries.
Avoid assuming any tables or columns other than those above.
"""


class SQLChatAgentConfig(ChatAgentConfig):
    system_message: str = DEFAULT_SQL_CHAT_SYSTEM_MESSAGE
    user_message: None | str = None
    max_context_tokens: int = 1000
    cache: bool = True  # cache results
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    database_uri: str  # Database URI
    retry_query: int = 3  # Number of times to retry query
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


class RunQueryTool(ToolMessage):
    request: str = "run_query"
    purpose: str = """
            To run <query> on the database 'db' and 
            return the results to answer a question.
            """
    query: str


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
        self.tables_info = ""

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
        self.tables_info = self.gather_table_info(context_descriptions, inspector)

        # Update the system message with the table information
        self.config.system_message = self.config.system_message.format(
            tables=self.tables_info
        )

        # Enable the agent to use and handle the RunQueryTool
        self.enable_message(RunQueryTool)

    def agent_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        # Your override code here
        if msg is None:
            return None

        results = self.handle_message(msg)
        if results is None:
            return None

        output = results
        if "There was an error in your SQL Query" in output:
            output = "There was an error in the SQL Query. Press enter to retry."

        console.print(f"[red]{self.indent}", end="")
        print(f"[red]Agent: {output}")
        sender_name = self.config.name
        if isinstance(msg, ChatDocument) and msg.function_call is not None:
            sender_name = msg.function_call.name

        return ChatDocument(
            content=results,
            metadata=ChatDocMetaData(
                source=Entity.AGENT,
                sender=Entity.AGENT,
                sender_name=sender_name,
            ),
        )

    def gather_table_info(
        self,
        context_descriptions: Optional[
            Dict[str, Dict[str, Union[str, Dict[str, str]]]]
        ],
        inspector: Inspector,
    ) -> str:
        """
        Gather information about the tables in the database.

        Args:
            context_descriptions (Optional[Dict[str, Dict[str, str]]]):
                User-provided context descriptions.
            inspector (inspect): SQL alchemy inspector object.

        Returns:
            str: The formatted string containing the details of all tables.
        """
        tables_info = ""
        # Iterate over all tables present in the metadata
        for table_name in self.metadata.tables.keys():
            table_description: str = ""

            # Check if context_descriptions are provided
            if context_descriptions is not None:
                # Fetch the table_info from context_descriptions
                table_info = context_descriptions.get(table_name, {})
                if isinstance(table_info, dict):
                    # Get description of the table
                    description = table_info.get("description", "")
                    assert isinstance(description, str)
                    table_description = description

            # Append table_name and table_description to tables_info
            if table_description:
                tables_info += (
                    f"\nTable: {table_name}\nDescription: {table_description}\n"
                )
            else:
                tables_info += f"\nTable: {table_name}\n"

            # Get column details for the table
            columns = inspector.get_columns(table_name)
            table = PrettyTable()
            table.field_names = ["Column Name", "Type", "Description"]
            columns_info: Dict[str, Any] = {}
            # Loop over all columns to populate the table
            for column in columns:
                description = ""  # Initialize description as empty string
                if context_descriptions is not None:
                    table_info = context_descriptions.get(table_name, {})
                    if isinstance(table_info, dict):
                        # Check if 'columns' key is in the table_info dict
                        if "columns" in table_info and isinstance(
                            table_info["columns"], dict
                        ):
                            columns_info = table_info["columns"]
                            # Fetch column description from context_descriptions
                            description = columns_info.get(column["name"], "")
                # Add a new row in the table for each column
                table.add_row([column["name"], column["type"], description])

            # Append the generated table to tables_info
            tables_info += f"{table}\n"

        return tables_info

    def retry_query(self, e: Exception, query: str) -> str:
        result = f"""There was an error in your SQL Query: '{query}'
{str(e)} 
Run a new query, correcting the errors. 
Refer to the table description for information about the tables and columns.
{self.tables_info}"""

        return result

    def run_query(self, msg: RunQueryTool) -> str:
        """
        Handle a RunQueryTool message by executing a SQL query and returning the result.

        Args:
            msg (RunQueryTool): The tool-message to handle.

        Returns:
            str: The result of executing the SQL query.
        """
        query = msg.query
        session = self.Session()
        result = ""

        try:
            # Execute the SQL query
            query_result = session.execute(text(query))
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

            result = self.retry_query(e, query)
        finally:
            # Always close the session
            session.close()

        return result
