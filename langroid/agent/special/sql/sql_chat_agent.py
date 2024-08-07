"""
Agent that allows interaction with an SQL database using SQLAlchemy library. 
The agent can execute SQL queries in the database and return the result. 

Functionality includes:
- adding table and column context
- asking a question about a SQL schema
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from rich import print
from rich.console import Console

from langroid.exceptions import LangroidImportError
from langroid.utils.constants import DONE

try:
    from sqlalchemy import MetaData, Row, create_engine, inspect, text
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import ResourceClosedError, SQLAlchemyError
    from sqlalchemy.orm import Session, sessionmaker
except ImportError as e:
    raise LangroidImportError(extra="sql", error=str(e))

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.special.sql.utils.description_extractors import (
    extract_schema_descriptions,
)
from langroid.agent.special.sql.utils.populate_metadata import (
    populate_metadata,
    populate_metadata_with_schema_tools,
)
from langroid.agent.special.sql.utils.system_message import (
    DEFAULT_SYS_MSG,
    SCHEMA_TOOLS_SYS_MSG,
)
from langroid.agent.special.sql.utils.tools import (
    GetColumnDescriptionsTool,
    GetTableNamesTool,
    GetTableSchemaTool,
    RunQueryTool,
)
from langroid.mytypes import Entity
from langroid.vector_store.base import VectorStoreConfig

logger = logging.getLogger(__name__)

console = Console()

DEFAULT_SQL_CHAT_SYSTEM_MESSAGE = f"""
{{mode}}

You do not need to attempt answering a question with just one query. 
You could make a sequence of SQL queries to help you write the final query.
Also if you receive a null or other unexpected result,
(a) make sure you use the available TOOLs correctly, and 
(b) see if you have made an assumption in your SQL query, and try another way, 
   or use `run_query` to explore the database table contents before submitting your 
   final query. For example when searching for "males" you may have used "gender= 'M'",
in your query, because you did not know that the possible genders in the table
are "Male" and "Female". 

Start by asking what I would like to know about the data.

When you have FINISHED the given query or database update task, 
say {DONE} and show your answer.

"""

ADDRESSING_INSTRUCTION = f"""
IMPORTANT - Whenever you are NOT writing a SQL query, make sure you address the user
using {{prefix}}User. You MUST use the EXACT syntax {{prefix}} !!!

In other words, you ALWAYS write EITHER:
 - a SQL query using the `run_query` tool, 
 - OR address the user using {{prefix}}User, and include {DONE} to indicate your 
     task is FINISHED. 
"""


SQL_ERROR_MSG = "There was an error in your SQL Query"


class SQLChatAgentConfig(ChatAgentConfig):
    system_message: str = DEFAULT_SQL_CHAT_SYSTEM_MESSAGE
    user_message: None | str = None
    cache: bool = True  # cache results
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    database_uri: str = ""  # Database URI
    database_session: None | Session = None  # Database session
    vecdb: None | VectorStoreConfig = None
    context_descriptions: Dict[str, Dict[str, Union[str, Dict[str, str]]]] = {}
    use_schema_tools: bool = False
    multi_schema: bool = False
    addressing_prefix: str = ""

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

    If multi_schema support is enabled, the tables names in the description
    should be of the form 'schema_name.table_name'.

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


class SQLChatAgent(ChatAgent):
    """
    Agent for chatting with a SQL database
    """

    used_run_query: bool = False
    llm_responded: bool = False

    def __init__(self, config: "SQLChatAgentConfig") -> None:
        """Initialize the SQLChatAgent.

        Raises:
            ValueError: If database information is not provided in the config.
        """
        self._validate_config(config)
        self.config: SQLChatAgentConfig = config
        self._init_database()
        self._init_metadata()
        self._init_table_metadata()
        self._init_message_tools()

    def _validate_config(self, config: "SQLChatAgentConfig") -> None:
        """Validate the configuration to ensure all necessary fields are present."""
        if config.database_session is None and config.database_uri is None:
            raise ValueError("Database information must be provided")

    def _init_database(self) -> None:
        """Initialize the database engine and session."""
        if self.config.database_session:
            self.Session = self.config.database_session
            self.engine = self.Session.bind
        else:
            self.engine = create_engine(self.config.database_uri)
            self.Session = sessionmaker(bind=self.engine)()

    def _init_metadata(self) -> None:
        """Initialize the database metadata."""
        if self.engine is None:
            raise ValueError("Database engine is None")
        self.metadata: MetaData | List[MetaData] = []

        if self.config.multi_schema:
            logger.info(
                "Initializing SQLChatAgent with database: %s",
                self.engine,
            )

            self.metadata = []
            inspector = inspect(self.engine)

            for schema in inspector.get_schema_names():
                metadata = MetaData(schema=schema)
                metadata.reflect(self.engine)
                self.metadata.append(metadata)

                logger.info(
                    "Initializing SQLChatAgent with database: %s, schema: %s, "
                    "and tables: %s",
                    self.engine,
                    schema,
                    metadata.tables,
                )
        else:
            self.metadata = MetaData()
            self.metadata.reflect(self.engine)
            logger.info(
                "SQLChatAgent initialized with database: %s and tables: %s",
                self.engine,
                self.metadata.tables,
            )

    def _init_table_metadata(self) -> None:
        """Initialize metadata for the tables present in the database."""
        if not self.config.context_descriptions and isinstance(self.engine, Engine):
            self.config.context_descriptions = extract_schema_descriptions(
                self.engine, self.config.multi_schema
            )

        if self.config.use_schema_tools:
            self.table_metadata = populate_metadata_with_schema_tools(
                self.metadata, self.config.context_descriptions
            )
        else:
            self.table_metadata = populate_metadata(
                self.metadata, self.config.context_descriptions
            )

    def _init_message_tools(self) -> None:
        """Initialize message tools used for chatting."""
        message = self._format_message()
        self.config.system_message = self.config.system_message.format(mode=message)
        if self.config.addressing_prefix != "":
            self.config.system_message += ADDRESSING_INSTRUCTION.format(
                prefix=self.config.addressing_prefix
            )
        super().__init__(self.config)
        self.enable_message(RunQueryTool)
        if self.config.use_schema_tools:
            self._enable_schema_tools()

    def _format_message(self) -> str:
        if self.engine is None:
            raise ValueError("Database engine is None")

        """Format the system message based on the engine and table metadata."""
        return (
            SCHEMA_TOOLS_SYS_MSG.format(dialect=self.engine.dialect.name)
            if self.config.use_schema_tools
            else DEFAULT_SYS_MSG.format(
                dialect=self.engine.dialect.name, schema_dict=self.table_metadata
            )
        )

    def _enable_schema_tools(self) -> None:
        """Enable tools for schema-related functionalities."""
        self.enable_message(GetTableNamesTool)
        self.enable_message(GetTableSchemaTool)
        self.enable_message(GetColumnDescriptionsTool)

    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        self.llm_responded = True
        return super().llm_response(message)

    def user_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        self.llm_responded = False
        self.used_run_query = False
        return super().user_response(msg)

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:

        if not self.llm_responded:
            return None
        if self.used_run_query:
            prefix = (
                self.config.addressing_prefix + "User"
                if self.config.addressing_prefix
                else ""
            )
            return (
                DONE + prefix + (msg.content if isinstance(msg, ChatDocument) else msg)
            )

        else:
            reminder = """
            You may have forgotten to use the `run_query` tool to execute an SQL query
            for the user's question/request            
            """
            if self.config.addressing_prefix != "":
                reminder += f"""
                OR you may have forgotten to address the user using the prefix
                {self.config.addressing_prefix} 
                """
            return reminder

    def _agent_response(
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
        if SQL_ERROR_MSG in output:
            output = "There was an error in the SQL Query. Press enter to retry."

        console.print(f"[red]{self.indent}", end="")
        print(f"[red]Agent: {output}")
        sender_name = self.config.name
        if isinstance(msg, ChatDocument) and msg.function_call is not None:
            sender_name = msg.function_call.name

        content = results.content if isinstance(results, ChatDocument) else results

        return ChatDocument(
            content=content,
            metadata=ChatDocMetaData(
                source=Entity.AGENT,
                sender=Entity.AGENT,
                sender_name=sender_name,
            ),
        )

    def retry_query(self, e: Exception, query: str) -> str:
        """
        Generate an error message for a failed SQL query and return it.

        Parameters:
        e (Exception): The exception raised during the SQL query execution.
        query (str): The SQL query that failed.

        Returns:
        str: The error message.
        """
        logger.error(f"SQL Query failed: {query}\nException: {e}")

        # Optional part to be included based on `use_schema_tools`
        optional_schema_description = ""
        if not self.config.use_schema_tools:
            optional_schema_description = f"""\
            This JSON schema maps SQL database structure. It outlines tables, each 
            with a description and columns. Each table is identified by a key, and holds
            a description and a dictionary of columns, with column 
            names as keys and their descriptions as values.
            
            ```json
            {self.config.context_descriptions}
            ```"""

        # Construct the error message
        error_message_template = f"""\
        {SQL_ERROR_MSG}: '{query}'
        {str(e)}
        Run a new query, correcting the errors.
        {optional_schema_description}"""

        return error_message_template

    def run_query(self, msg: RunQueryTool) -> str:
        """
        Handle a RunQueryTool message by executing a SQL query and returning the result.

        Args:
            msg (RunQueryTool): The tool-message to handle.

        Returns:
            str: The result of executing the SQL query.
        """
        query = msg.query
        session = self.Session
        self.used_run_query = True
        try:
            logger.info(f"Executing SQL query: {query}")

            query_result = session.execute(text(query))
            session.commit()
            try:
                # attempt to fetch results: should work for normal SELECT queries
                rows = query_result.fetchall()
                response_message = self._format_rows(rows)
            except ResourceClosedError:
                # If we get here, it's a non-SELECT query (UPDATE, INSERT, DELETE)
                affected_rows = query_result.rowcount  # type: ignore
                response_message = f"""
                    Non-SELECT query executed successfully. 
                    Rows affected: {affected_rows}
                    """

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to execute query: {query}\n{e}")
            response_message = self.retry_query(e, query)
        finally:
            session.close()

        return response_message

    def _format_rows(self, rows: Sequence[Row[Any]]) -> str:
        """
        Format the rows fetched from the query result into a string.

        Args:
            rows (list): List of rows fetched from the query result.

        Returns:
            str: Formatted string representation of rows.
        """
        # TODO: UPDATE FORMATTING
        return (
            ",\n".join(str(row) for row in rows)
            if rows
            else "Query executed successfully."
        )

    def get_table_names(self, msg: GetTableNamesTool) -> str:
        """
        Handle a GetTableNamesTool message by returning the names of all tables in the
        database.

        Returns:
            str: The names of all tables in the database.
        """
        if isinstance(self.metadata, list):
            table_names = [", ".join(md.tables.keys()) for md in self.metadata]
            return ", ".join(table_names)

        return ", ".join(self.metadata.tables.keys())

    def get_table_schema(self, msg: GetTableSchemaTool) -> str:
        """
        Handle a GetTableSchemaTool message by returning the schema of all provided
        tables in the database.

        Returns:
            str: The schema of all provided tables in the database.
        """
        tables = msg.tables
        result = ""
        for table_name in tables:
            table = self.table_metadata.get(table_name)
            if table is not None:
                result += f"{table_name}: {table}\n"
            else:
                result += f"{table_name} is not a valid table name.\n"
        return result

    def get_column_descriptions(self, msg: GetColumnDescriptionsTool) -> str:
        """
        Handle a GetColumnDescriptionsTool message by returning the descriptions of all
        provided columns from the database.

        Returns:
            str: The descriptions of all provided columns from the database.
        """
        table = msg.table
        columns = msg.columns.split(", ")
        result = f"\nTABLE: {table}"
        descriptions = self.config.context_descriptions.get(table)

        for col in columns:
            result += f"\n{col} => {descriptions['columns'][col]}"  # type: ignore
        return result
