import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from rich import print
from rich.console import Console

from langroid.pydantic_v1 import BaseModel, BaseSettings

if TYPE_CHECKING:
    import neo4j

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.special.neo4j.system_messages import (
    ADDRESSING_INSTRUCTION,
    DEFAULT_NEO4J_CHAT_SYSTEM_MESSAGE,
    DONE_INSTRUCTION,
    SCHEMA_PROVIDED_SYS_MSG,
    SCHEMA_TOOLS_SYS_MSG,
)
from langroid.agent.special.neo4j.tools import (
    CypherCreationTool,
    CypherRetrievalTool,
    GraphSchemaTool,
    cypher_creation_tool_name,
    cypher_retrieval_tool_name,
    graph_schema_tool_name,
)
from langroid.agent.tools.orchestration import DoneTool, ForwardTool
from langroid.exceptions import LangroidImportError
from langroid.mytypes import Entity
from langroid.utils.constants import SEND_TO

logger = logging.getLogger(__name__)

console = Console()

NEO4J_ERROR_MSG = "There was an error in your Cypher Query"


# TOOLS to be used by the agent


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
    data: List[Dict[Any, Any]] | str | None = None


class Neo4jChatAgentConfig(ChatAgentConfig):
    neo4j_settings: Neo4jSettings = Neo4jSettings()
    system_message: str = DEFAULT_NEO4J_CHAT_SYSTEM_MESSAGE
    kg_schema: Optional[List[Dict[str, Any]]] = None
    database_created: bool = False
    # whether agent MUST use schema_tools to get schema, i.e.
    # schema is NOT initially provided
    use_schema_tools: bool = True
    use_functions_api: bool = True
    use_tools: bool = False
    # whether the agent is used in a continuous chat with user,
    # as opposed to returning a result from the task.run()
    chat_mode: bool = False
    addressing_prefix: str = ""


class Neo4jChatAgent(ChatAgent):
    def __init__(self, config: Neo4jChatAgentConfig):
        """Initialize the Neo4jChatAgent.

        Raises:
            ValueError: If database information is not provided in the config.
        """
        self.config: Neo4jChatAgentConfig = config
        self._validate_config()
        self._import_neo4j()
        self._initialize_db()
        self._init_tools_sys_message()
        self.init_state()

    def init_state(self) -> None:
        super().init_state()
        self.current_retrieval_cypher_query: str = ""
        self.tried_schema: bool = False

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ForwardTool | None:
        """
        When LLM sends a no-tool msg, assume user is the intended recipient,
        and if in interactive mode, forward the msg to the user.
        """

        done_tool_name = DoneTool.default_value("request")
        forward_tool_name = ForwardTool.default_value("request")
        if isinstance(msg, ChatDocument) and msg.metadata.sender == Entity.LLM:
            if self.interactive:
                return ForwardTool(agent="User")
            else:
                if self.config.chat_mode:
                    return f"""
                    Since you did not explicitly address the User, it is not clear
                    whether:
                    - you intend this to be the final response to the 
                      user's query/request, in which case you must use the 
                      `{forward_tool_name}` to indicate this.
                    - OR, you FORGOT to use an Appropriate TOOL,
                      in which case you should use the available tools to
                      make progress on the user's query/request.
                    """
                return f"""
                The intent of your response is not clear:
                - if you intended this to be the final answer to the user's query,
                    then use the `{done_tool_name}` to indicate so,
                    with the `content` set to the answer or result.
                - otherwise, use one of the available tools to make progress 
                    to arrive at the final answer.
                """
        return None

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
            raise LangroidImportError("neo4j", "neo4j")

    def _initialize_db(self) -> None:
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
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]  # type: ignore
                self.config.database_created = count > 0

            # If database has data, get schema
            if self.config.database_created:
                # this updates self.config.kg_schema
                self.graph_schema_tool(None)

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
        # Check if query contains database/collection creation patterns
        query_upper = query.upper()
        is_creation_query = any(
            [
                "CREATE" in query_upper,
                "MERGE" in query_upper,
                "CREATE CONSTRAINT" in query_upper,
                "CREATE INDEX" in query_upper,
            ]
        )

        if is_creation_query:
            self.config.database_created = True
            logger.info("Detected database/collection creation query")

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

    def cypher_retrieval_tool(self, msg: CypherRetrievalTool) -> str:
        """ "
        Handle a CypherRetrievalTool message by executing a Cypher query and
        returning the result.
        Args:
            msg (CypherRetrievalTool): The tool-message to handle.

        Returns:
            str: The result of executing the cypher_query.
        """
        if not self.tried_schema:
            return f"""
            You did not yet use the `{graph_schema_tool_name}` tool to get the schema 
            of the neo4j knowledge-graph db. Use that tool first before using 
            the `{cypher_retrieval_tool_name}` tool, to ensure you know all the correct
            node labels, relationship types, and property keys available in
            the database.
            """
        elif not self.config.database_created:
            return f"""
            You have not yet created the Neo4j database. 
            Use the `{cypher_creation_tool_name}`
            tool to create the database first before using the 
            `{cypher_retrieval_tool_name}` tool.
            """
        query = msg.cypher_query
        self.current_retrieval_cypher_query = query
        logger.info(f"Executing Cypher query: {query}")
        response = self.read_query(query)
        if isinstance(response.data, list) and len(response.data) == 0:
            return """
            No results found; check if your query used the right label names -- 
            remember these are case sensitive, so you have to use the exact label
            names you found in the schema. 
            Or retry using one of the  RETRY-SUGGESTIONS in your instructions. 
            """
        return str(response.data)

    def cypher_creation_tool(self, msg: CypherCreationTool) -> str:
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
            self.config.database_created = True
            return "Cypher query executed successfully"
        else:
            return str(response.data)

    # TODO: There are various ways to get the schema. The current one uses the func
    # `read_query`, which requires post processing to identify whether the response upon
    # the schema query is valid. Another way is to isolate this func from `read_query`.
    # The current query works well. But we could use the queries here:
    # https://github.com/neo4j/NaLLM/blob/1af09cd117ba0777d81075c597a5081583568f9f/api/
    # src/driver/neo4j.py#L30
    def graph_schema_tool(
        self, msg: GraphSchemaTool | None
    ) -> str | Optional[Union[str, List[Dict[Any, Any]]]]:
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
        self.tried_schema = True
        if self.config.kg_schema is not None and len(self.config.kg_schema) > 0:
            return self.config.kg_schema
        schema_result = self.read_query("CALL db.schema.visualization()")
        if schema_result.success:
            # there is a possibility that the schema is empty, which is a valid response
            # the schema.data will be: [{"nodes": [], "relationships": []}]
            self.config.kg_schema = schema_result.data  # type: ignore
            return schema_result.data
        else:
            return f"Failed to retrieve schema: {schema_result.data}"

    def _init_tools_sys_message(self) -> None:
        """Initialize message tools used for chatting."""
        self.tried_schema = False
        message = self._format_message()
        self.config.system_message = self.config.system_message.format(mode=message)
        if self.config.chat_mode:
            self.config.addressing_prefix = self.config.addressing_prefix or SEND_TO
            self.config.system_message += ADDRESSING_INSTRUCTION.format(
                prefix=self.config.addressing_prefix
            )
        else:
            self.config.system_message += DONE_INSTRUCTION
        super().__init__(self.config)
        # Note we are enabling GraphSchemaTool regardless of whether
        # self.config.use_schema_tools is True or False, because
        # even when schema provided, the agent may later want to get the schema,
        # e.g. if the db evolves, or if it needs to bring in the schema
        self.enable_message(
            [
                GraphSchemaTool,
                CypherRetrievalTool,
                CypherCreationTool,
                DoneTool,
            ]
        )

    def _format_message(self) -> str:
        if self.driver is None:
            raise ValueError("Database driver None")
        assert isinstance(self.config, Neo4jChatAgentConfig)
        return (
            SCHEMA_TOOLS_SYS_MSG
            if self.config.use_schema_tools
            else SCHEMA_PROVIDED_SYS_MSG.format(schema=self.graph_schema_tool(None))
        )
