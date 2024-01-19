import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseSettings
from rich import print
from rich.console import Console

if TYPE_CHECKING:
    import neo4j


from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.special.neo4j.utils.system_message import (
    DEFAULT_NEO4J_CHAT_SYSTEM_MESSAGE,
    DEFAULT_SYS_MSG,
    SCHEMA_TOOLS_SYS_MSG,
)
from langroid.agent.special.neo4j.utils.tools import (
    CypherQueryTool,
    GraphSchemaTool,
)
from langroid.mytypes import Entity

logger = logging.getLogger(__name__)

console = Console()

NEO4J_ERROR_MSG = "There was an error in your Cypher Query"

empty_nodes = "'nodes': []"
empty_relationships = "'relationships': []"
not_valid_query_response = [
    empty_nodes,
    empty_relationships,
    NEO4J_ERROR_MSG,
]


class Neo4jSettings(BaseSettings):
    uri: str = ""
    username: str = ""
    password: str = ""
    database: str = ""

    class Config:
        # This enables the use of environment variables to set the settings,
        # e.g. NEO4J_URI, NEO4J_USERNAME, etc.,
        # which can either be set in a .env file or in the shell via export cmds.
        env_prefix = "NEO4J_"


class Neo4jChatAgentConfig(ChatAgentConfig):
    neo4j_settings: Neo4jSettings = Neo4jSettings()
    system_message: str = DEFAULT_NEO4J_CHAT_SYSTEM_MESSAGE
    kg_schema: Optional[List[Dict[str, Any]]]
    database_created: bool = False
    use_schema_tools: bool = False
    use_functions_api: bool = True
    use_tools: bool = False


class Neo4jChatAgent(ChatAgent):
    def __init__(self, config: Neo4jChatAgentConfig):
        """Initialize the Neo4jChatAgent.

        Raises:
            ValueError: If database information is not provided in the config.
        """
        self.config = config
        self._validate_config()
        self._import_neo4j()
        self._initialize_connection()
        self._init_tool_messages()

    def _validate_config(self) -> None:
        """Validate the configuration to ensure all necessary fields are present."""
        assert isinstance(self.config, Neo4jChatAgentConfig)
        if (
            self.config.neo4j_settings.username is None
            and self.config.neo4j_settings.password is None
            and self.config.neo4j_settings.database
        ):
            raise ValueError("Neo4j env information must be provided")

    def _import_neo4j(self) -> None:
        """Dynamically imports the Neo4j module and sets it as a global variable."""
        global neo4j
        try:
            import neo4j
        except ImportError:
            raise ImportError(
                """
                neo4j not installed. Please install it via:
                pip install neo4j.
                Or when installing langroid, install it with the `neo4j` extra:
                pip install langroid[neo4j]
                """
            )

    def _initialize_connection(self) -> None:
        """
        Initializes a connection to the Neo4j database using the configuration settings.
        """
        try:
            assert isinstance(self.config, Neo4jChatAgentConfig)
            self.driver = neo4j.GraphDatabase.driver(
                self.config.neo4j_settings.uri,
                auth=(
                    self.config.neo4j_settings.username,
                    self.config.neo4j_settings.password,
                ),
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Neo4j connection: {e}")

    def close(self) -> None:
        """close the connection"""
        if self.driver:
            self.driver.close()

    def retry_query(self, e: Exception, query: str) -> str:
        """
        Generate an error message for a failed Cypher query and return it.

        Args:
            e (Exception): The exception raised during the Cypher query execution.
            query (str): The Cypher query that failed.

        Returns:
            str: The error message.
        """
        logger.error(f"Cypher Query failed: {query}\nException: {e}")

        # Construct the error message
        error_message_template = f"""\
        {NEO4J_ERROR_MSG}: '{query}'
        {str(e)}
        Run a new query, correcting the errors.
        """

        return error_message_template

    def read_query(
        self, query: str, parameters: Optional[Dict[Any, Any]] = None
    ) -> str:
        """
        Executes a given Cypher query with parameters on the Neo4j database.

        Args:
            query (str): The Cypher query string to be executed.
            parameters (Optional[Dict[Any, Any]]): A dictionary of parameters for the
            query. Defaults to None.

        Returns:
            str: The result of executing the Cypher query.
        """
        response_message = ""
        if not self.driver:
            raise ValueError("No database connection is established.")

        try:
            assert isinstance(self.config, Neo4jChatAgentConfig)
            with self.driver.session(
                database=self.config.neo4j_settings.database
            ) as session:
                result = session.run(query, parameters)
                # Check if there are records in the result
                if result.peek():
                    response_message = ", ".join(
                        [str(record.data()) for record in result]
                    )
                else:
                    response_message = "No records found."
        except Exception as e:
            logger.error(f"Failed to execute query: {query}\n{e}")
            response_message = self.retry_query(e, query)
        finally:
            self.close()

        return response_message

    def write_query(
        self, query: str, parameters: Optional[Dict[Any, Any]] = None
    ) -> bool:
        """
        Executes a write transaction using a given Cypher query on the Neo4j database.
        This method should be used for queries that modify the database.

        Args:
            query (str): The Cypher query string to be executed.
            parameters (dict, optional): A dict of parameters for the Cypher query

        Returns:
            bool: True if the query was executed successfully, False otherwise.
        """
        if not self.driver:
            raise ValueError("No database connection is established.")
        response = False
        try:
            assert isinstance(self.config, Neo4jChatAgentConfig)
            with self.driver.session(
                database=self.config.neo4j_settings.database
            ) as session:
                # Execute the query within a write transaction
                session.write_transaction(lambda tx: tx.run(query, parameters))
                response = True
        except Exception as e:
            logging.warning(
                f"An unexpected error occurred while executing the write query: {e}"
            )
        finally:
            self.close()
        return response

    # TODO: test under enterprise edition because community edition doesn't allow
    # database creation/deletion
    def remove_database(self) -> None:
        """Deletes all nodes and relationships from the current Neo4j database."""
        delete_query = """
                MATCH (n)
                DETACH DELETE n
            """
        if self.write_query(delete_query):
            print("[green]Database is deleted!")
        else:
            print("[red]Database is not deleted!")

    def make_query(self, msg: CypherQueryTool) -> str:
        """ "
        Handle a GenerateCypherQueries message by executing a Cypher query and
        returning the result.
        Args:
            msg (CypherQueryTool): The tool-message to handle.

        Returns:
            str: The result of executing the Cypherquery.
        """
        query = msg.cypherQuery

        logger.info(f"Executing Cypher query: {query}")
        return self.read_query(query)

    # TODO: There are various ways to get the schema. The current one uses the func
    # `read_query`, which requires post processing to identify whether the response upon
    # the schema query is valid. Another way is to isolate this func from `read_query`.
    # The current query works well. But we could use the queries here:
    # https://github.com/neo4j/NaLLM/blob/1af09cd117ba0777d81075c597a5081583568f9f/api/
    # src/driver/neo4j.py#L30
    def get_schema(self, msg: GraphSchemaTool | None) -> str:
        """
        Retrieves the schema of a Neo4j graph database.

        Args:
            msg (GraphSchemaTool): An instance of GraphDatabaseSchema, typically
            containing information or parameters needed for the database query.

        Returns:
            str: The visual representation of the database schema as a string, or a
            message stating that the database schema is empty or not valid.

        Raises:
            This function does not explicitly raise exceptions but depends on the
            behavior of 'self.read_query' method, which might raise exceptions related
             to database connectivity or query execution.
        """
        schema = self.read_query("CALL db.schema.visualization()")
        if not any(element in schema for element in not_valid_query_response):
            return schema
        else:
            return "The database schema does not have any nodes or relationships."

    def _init_tool_messages(self) -> None:
        """Initialize message tools used for chatting."""
        message = self._format_message()
        self.config.system_message = self.config.system_message.format(mode=message)
        super().__init__(self.config)
        self.enable_message(CypherQueryTool)
        self.enable_message(GraphSchemaTool)

    def _format_message(self) -> str:
        if self.driver is None:
            raise ValueError("Database driver None")
        assert isinstance(self.config, Neo4jChatAgentConfig)
        return (
            SCHEMA_TOOLS_SYS_MSG.format(schema=self.get_schema(None))
            if self.config.use_schema_tools
            else DEFAULT_SYS_MSG
        )

    def agent_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        if msg is None:
            return None

        results = self.handle_message(msg)
        if results is None:
            return None

        output = results
        if NEO4J_ERROR_MSG in output:
            output = "There was an error in the Cypher Query. Press enter to retry."

        console.print(f"[red]{self.indent}", end="")
        print(f"[red]Agent: {output}")
        sender_name = self.config.name
        if isinstance(msg, ChatDocument) and msg.function_call is not None:
            sender_name = msg.function_call.name

        content = results.content if isinstance(results, ChatDocument) else results
        recipient = (
            results.metadata.recipient if isinstance(results, ChatDocument) else ""
        )

        return ChatDocument(
            content=content,
            metadata=ChatDocMetaData(
                # source=Entity.AGENT,
                sender=Entity.LLM,
                sender_name=sender_name,
                recipient=recipient,
            ),
        )
