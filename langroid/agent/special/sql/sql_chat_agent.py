"""
Agent that allows interaction with an SQL database using SQLAlchemy library. 
The agent can execute SQL queries in the database and return the result. 

Functionality includes:
- adding table and column context
- asking a question about a SQL schema
"""
import logging
from typing import Any, Dict, Optional, Union

from rich import print
from rich.console import Console
from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

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
You are a savvy data scientist/database administrator, with expertise in 
answering questions by querying a {dialect} database.
You do not have access to the database 'db' directly, so you will need to use the 
`run_query` tool/function-call to answer questions.

The below JSON schema maps the SQL database structure. It outlines tables, each 
with a description and columns. Each table is identified by a key, 
and holds a description and a dictionary of columns, 
with column names as keys and their descriptions as values. 
{schema_dict}

ONLY the tables and column names and tables specified above should be used in
the generated queries. 
You must be smart about using the right tables and columns based on the 
english description. If you are thinking of using a table or column that 
does not exist, you are probably on the wrong track, so you should try
your best to answer based on an existing table or column.
DO NOT assume any tables or columns other than those above.

You do not need to attempt answering a question with just one query. 
You could make a sequence of SQL queries to help you write the final query.
Also if you receive a null or other unexpected result, 
see if you have made an assumption in your SQL query, and try another way, 
or use `run_query` to explore the database table contents before submitting your 
final query. For example when searching for "males" you may have used "gender= 'M'",
in your query, because you did not know that the possible genders in the table
are "Male" and "Female". 

Start by asking what I would like to know about the data.

"""

SQL_ERROR_MSG = "There was an error in your SQL Query"


class SQLChatAgentConfig(ChatAgentConfig):
    system_message: str = DEFAULT_SQL_CHAT_SYSTEM_MESSAGE
    user_message: None | str = None
    max_context_tokens: int = 1000
    cache: bool = True  # cache results
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    database_uri: str = ""  # Database URI
    database_session: None | Session = None  # Database session
    vecdb: None | VectorStoreConfig = None
    context_descriptions: Dict[str, Dict[str, Union[str, Dict[str, str]]]] = {}

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
        self.config: SQLChatAgentConfig = config
        if config.database_session is not None:
            self.Session = config.database_session
            self.engine = self.Session.bind
        else:
            self.engine = create_engine(config.database_uri)
            self.Session = sessionmaker(bind=self.engine)()
        self.metadata = MetaData()

        if self.engine is None:
            raise ValueError("Database engine is None")
        self.metadata.reflect(self.engine)

        logger.info(
            f"""SQLChatAgent initialized with database: 
            {self.engine} and tables: 
            {self.metadata.tables}
            """
        )

        if not config.context_descriptions and isinstance(self.engine, Engine):
            config.context_descriptions = extract_schema_descriptions(self.engine)

        # Combine database information with context descriptions
        schema_dict = combine_metadata(self.metadata, config.context_descriptions)

        # Update the system message with the table information
        self.config.system_message = self.config.system_message.format(
            schema_dict=schema_dict, dialect=self.engine.dialect.name
        )

        super().__init__(config)

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
        if SQL_ERROR_MSG in output:
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

    def retry_query(self, e: Exception, query: str) -> str:
        result = f"""{SQL_ERROR_MSG}: '{query}'
{str(e)} 
Run a new query, correcting the errors. 
This JSON schema maps SQL database structure. It outlines tables, each 
with a description and columns. Each table is identified by a key, 
and holds a description and a dictionary of columns, 
with column names as keys and their descriptions as values.

```json
{self.config.context_descriptions}
```"""

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
        session = self.Session
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


def combine_metadata(
    metadata: MetaData, info: Dict[str, Dict[str, Union[str, Dict[str, str]]]]
) -> Dict[str, Dict[str, Union[str, Dict[str, str]]]]:
    """
    Extracts information from an SQLAlchemy database's metadata and combines it
    with another dictionary with context descriptions.

    Args:
        metadata (MetaData): SQLAlchemy metadata object of the database.
        info (Dict[str, Dict[str, Any]]): A dictionary with table and column
                                             descriptions.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary with table and context information.
    """

    db_info: Dict[str, Dict[str, Union[str, Dict[str, str]]]] = {}

    # Create empty metadata dictionary with column datatypes
    for table_name, table in metadata.tables.items():
        # Populate tables with empty descriptions
        db_info[str(table_name)] = {"description": "", "columns": {}}

        for column in table.columns:
            # Populate columns with datatype
            db_info[str(table_name)]["columns"][str(column.name)] = (  # type: ignore
                str(column.type)
            )

    # Update the metadata dictionary with the context descriptions
    for table_name in db_info.keys():
        if table_name in info:
            # Populate table descriptions
            db_info[table_name]["description"] = info[table_name]["description"]

            for column_name in db_info[table_name]["columns"]:
                # Populate column descriptions
                if column_name in info[table_name]["columns"]:
                    db_info[table_name]["columns"][column_name] = (  # type: ignore
                        db_info[table_name]["columns"][column_name]  # type: ignore
                        + "; "
                        + info[table_name]["columns"][column_name]  # type: ignore
                    )

    return db_info


def extract_schema_descriptions(engine: Engine) -> Dict[str, Dict[str, Any]]:
    """
    Extracts the schema descriptions from the database connected to by the engine.

    Args:
        engine (Engine): SQLAlchemy engine instance.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary representation of table and column
        descriptions.
    """
    import langroid.agent.special.sql.utils.description_extractors as x

    extractors = {
        "postgresql": x.extract_postgresql_descriptions,
        "mysql": x.extract_mysql_descriptions,
    }
    return extractors.get(engine.dialect.name, x.extract_default_descriptions)(engine)
