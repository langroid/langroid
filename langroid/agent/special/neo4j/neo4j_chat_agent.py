import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import BaseModel, BaseSettings
from rich import print
from rich.console import Console

from langroid.agent import ToolMessage

if TYPE_CHECKING:
    import neo4j


from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.special.neo4j.utils.system_message import (
    DEFAULT_NEO4J_CHAT_SYSTEM_MESSAGE,
    DEFAULT_SYS_MSG,
    SCHEMA_TOOLS_SYS_MSG,
)
from langroid.mytypes import Entity

logger = logging.getLogger(__name__)

console = Console()

NEO4J_ERROR_MSG = "There was an error in your Cypher Query"


# TOOLS to be used by the agent


class CypherRetrievalTool(ToolMessage):
    request: str = "retrieval_query"
    purpose: str = """Use this tool to send the Cypher query to retreive data from the 
    graph database based provided text description and schema."""
    cypher_query: str


class CypherCreationTool(ToolMessage):
    request: str = "create_query"
    purpose: str = """Use this tool to send the Cypher query to create 
    entities/relationships in the graph database."""
    cypher_query: str


class GraphSchemaTool(ToolMessage):
    request: str = "get_schema"
    purpose: str = """To get the schema of the graph database."""


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


class QueryResult(BaseModel):
    success: bool
    data: Optional[Union[str, List[Dict[Any, Any]]]] = None


class Neo4jChatAgentConfig(ChatAgentConfig):
    neo4j_settings: Neo4jSettings = Neo4jSettings()
    system_message: str = DEFAULT_NEO4J_CHAT_SYSTEM_MESSAGE
    kg_schema: Optional[List[Dict[str, Any]]]
    database_created: bool = False
    use_schema_tools: bool = True
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
    ) -> QueryResult:
        """
        Executes a given Cypher query with parameters on the Neo4j database.

        Args:
            query (str): The Cypher query string to be executed.
            parameters (Optional[Dict[Any, Any]]): A dictionary of parameters for
                                                    the query.

        Returns:
            QueryResult: An object representing the outcome of the query execution.
        """
        if not self.driver:
            return QueryResult(
                success=False, data="No database connection is established."
            )

        try:
            assert isinstance(self.config, Neo4jChatAgentConfig)
            with self.driver.session(
                database=self.config.neo4j_settings.database
            ) as session:
                result = session.run(query, parameters)
                if result.peek():
                    records = [record.data() for record in result]
                    return QueryResult(success=True, data=records)
                else:
                    return QueryResult(success=True, data=[])
        except Exception as e:
            logger.error(f"Failed to execute query: {query}\n{e}")
            error_message = self.retry_query(e, query)
            return QueryResult(success=False, data=error_message)
        finally:
            self.close()

    def write_query(
        self, query: str, parameters: Optional[Dict[Any, Any]] = None
    ) -> QueryResult:
        """
        Executes a write transaction using a given Cypher query on the Neo4j database.
        This method should be used for queries that modify the database.

        Args:
            query (str): The Cypher query string to be executed.
            parameters (dict, optional): A dict of parameters for the Cypher query.

        Returns:
            QueryResult: An object representing the outcome of the query execution.
                         It contains a success flag and an optional error message.
        """
        if not self.driver:
            return QueryResult(
                success=False, data="No database connection is established."
            )

        try:
            assert isinstance(self.config, Neo4jChatAgentConfig)
            with self.driver.session(
                database=self.config.neo4j_settings.database
            ) as session:
                session.write_transaction(lambda tx: tx.run(query, parameters))
                return QueryResult(success=True)
        except Exception as e:
            logging.warning(f"An error occurred: {e}")
            error_message = self.retry_query(e, query)
            return QueryResult(success=False, data=error_message)
        finally:
            self.close()

    # TODO: test under enterprise edition because community edition doesn't allow
    # database creation/deletion
    def remove_database(self) -> None:
        """Deletes all nodes and relationships from the current Neo4j database."""
        delete_query = """
                MATCH (n)
                DETACH DELETE n
            """
        response = self.write_query(delete_query)

        if response.success:
            print("[green]Database is deleted!")
        else:
            print("[red]Database is not deleted!")

    def retrieval_query(self, msg: CypherRetrievalTool) -> str:
        """ "
        Handle a CypherRetrievalTool message by executing a Cypher query and
        returning the result.
        Args:
            msg (CypherRetrievalTool): The tool-message to handle.

        Returns:
            str: The result of executing the cypher_query.
        """
        query = msg.cypher_query

        logger.info(f"Executing Cypher query: {query}")
        response = self.read_query(query)
        if response.success:
            return json.dumps(response.data)
        else:
            return str(response.data)

    def create_query(self, msg: CypherCreationTool) -> str:
        """ "
        Handle a CypherCreationTool message by executing a Cypher query and
        returning the result.
        Args:
            msg (CypherCreationTool): The tool-message to handle.

        Returns:
            str: The result of executing the cypher_query.
        """
        query = msg.cypher_query

        logger.info(f"Executing Cypher query: {query}")
        response = self.write_query(query)
        if response.success:
            return "Cypher query executed successfully"
        else:
            return str(response.data)

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
        schema_result = self.read_query("CALL db.schema.visualization()")
        if schema_result.success:
            # ther is a possibility that the schema is empty, which is a valid response
            # the schema.data will be: [{"nodes": [], "relationships": []}]
            return json.dumps(schema_result.data)
        else:
            return f"Failed to retrieve schema: {schema_result.data}"

    def _init_tool_messages(self) -> None:
        """Initialize message tools used for chatting."""
        message = self._format_message()
        self.config.system_message = self.config.system_message.format(mode=message)
        super().__init__(self.config)
        self.enable_message(CypherRetrievalTool)
        self.enable_message(CypherCreationTool)
        self.enable_message(GraphSchemaTool)

    def _format_message(self) -> str:
        if self.driver is None:
            raise ValueError("Database driver None")
        assert isinstance(self.config, Neo4jChatAgentConfig)
        return (
            SCHEMA_TOOLS_SYS_MSG
            if self.config.use_schema_tools
            else DEFAULT_SYS_MSG.format(schema=self.get_schema(None))
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
