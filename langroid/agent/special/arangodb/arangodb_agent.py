import datetime
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from arango.client import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import ArangoError, ServerConnectionError
from numpy import ceil
from rich import print
from rich.console import Console

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.special.arangodb.system_messages import (
    ADDRESSING_INSTRUCTION,
    DEFAULT_ARANGO_CHAT_SYSTEM_MESSAGE,
    DONE_INSTRUCTION,
    SCHEMA_PROVIDED_SYS_MSG,
    SCHEMA_TOOLS_SYS_MSG,
)
from langroid.agent.special.arangodb.tools import (
    AQLCreationTool,
    AQLRetrievalTool,
    ArangoSchemaTool,
    aql_retrieval_tool_name,
    arango_schema_tool_name,
)
from langroid.agent.special.arangodb.utils import count_fields, trim_schema
from langroid.agent.tools.orchestration import DoneTool, ForwardTool
from langroid.exceptions import LangroidImportError
from langroid.mytypes import Entity
from langroid.pydantic_v1 import BaseModel, BaseSettings
from langroid.utils.constants import SEND_TO

logger = logging.getLogger(__name__)
console = Console()

ARANGO_ERROR_MSG = "There was an error in your AQL Query"
T = TypeVar("T")


class ArangoSettings(BaseSettings):
    client: ArangoClient | None = None
    db: StandardDatabase | None = None
    url: str = ""
    username: str = ""
    password: str = ""
    database: str = ""

    class Config:
        env_prefix = "ARANGO_"


class QueryResult(BaseModel):
    success: bool
    data: Optional[
        Union[
            str,
            int,
            float,
            bool,
            None,
            List[Any],
            Dict[str, Any],
            List[Dict[str, Any]],
        ]
    ] = None

    class Config:
        # Allow arbitrary types for flexibility
        arbitrary_types_allowed = True

        # Handle JSON serialization of special types
        json_encoders = {
            # Add custom encoders if needed, e.g.:
            datetime.datetime: lambda v: v.isoformat(),
            # Could add others for specific ArangoDB types
        }

        # Validate all assignments
        validate_assignment = True

        # Frozen=True if we want immutability
        frozen = False


class ArangoChatAgentConfig(ChatAgentConfig):
    arango_settings: ArangoSettings = ArangoSettings()
    system_message: str = DEFAULT_ARANGO_CHAT_SYSTEM_MESSAGE
    kg_schema: str | Dict[str, List[Dict[str, Any]]] | None = None
    database_created: bool = False
    prepopulate_schema: bool = True
    use_functions_api: bool = True
    max_num_results: int = 10  # how many results to return from AQL query
    max_schema_fields: int = 500  # max fields to show in schema
    max_tries: int = 10  # how many attempts to answer user question
    use_tools: bool = False
    schema_sample_pct: float = 0
    # whether the agent is used in a continuous chat with user,
    # as opposed to returning a result from the task.run()
    chat_mode: bool = False
    addressing_prefix: str = ""


class ArangoChatAgent(ChatAgent):
    def __init__(self, config: ArangoChatAgentConfig):
        super().__init__(config)
        self.config: ArangoChatAgentConfig = config
        self.init_state()
        self._validate_config()
        self._import_arango()
        self._initialize_db()
        self._init_tools_sys_message()

    def init_state(self) -> None:
        super().init_state()
        self.current_retrieval_aql_query: str = ""
        self.current_schema_params: ArangoSchemaTool = ArangoSchemaTool()
        self.num_tries = 0  # how many attempts to answer user question

    def user_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        response = super().user_response(msg)
        if response is None:
            return None
        response_str = response.content if response is not None else ""
        if response_str != "":
            self.num_tries = 0  # reset number of tries if user responds
        return response

    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        if self.num_tries > self.config.max_tries:
            if self.config.chat_mode:
                return self.create_llm_response(
                    content=f"""
                    {self.config.addressing_prefix}User
                    I give up, since I have exceeded the 
                    maximum number of tries ({self.config.max_tries}).
                    Feel free to give me some hints!
                    """
                )
            else:
                return self.create_llm_response(
                    tool_messages=[
                        DoneTool(
                            content=f"""
                            Exceeded maximum number of tries ({self.config.max_tries}).
                            """
                        )
                    ]
                )

        if isinstance(message, ChatDocument) and message.metadata.sender == Entity.USER:
            message.content = (
                message.content
                + "\n"
                + """
                (REMEMBER, Do NOT use more than ONE TOOL/FUNCTION at a time!
                you must WAIT for a helper to send you the RESULT(S) before
                making another TOOL/FUNCTION call)
                """
            )

        response = super().llm_response(message)
        if (
            response is not None
            and self.config.chat_mode
            and self.config.addressing_prefix in response.content
            and self.has_tool_message_attempt(response)
        ):
            # response contains both a user-addressing and a tool, which
            # is not allowed, so remove the user-addressing prefix
            response.content = response.content.replace(
                self.config.addressing_prefix, ""
            )

        return response

    def _validate_config(self) -> None:
        assert isinstance(self.config, ArangoChatAgentConfig)
        if (
            self.config.arango_settings.client is None
            or self.config.arango_settings.db is None
        ):
            if not all(
                [
                    self.config.arango_settings.url,
                    self.config.arango_settings.username,
                    self.config.arango_settings.password,
                    self.config.arango_settings.database,
                ]
            ):
                raise ValueError("ArangoDB connection info must be provided")

    def _import_arango(self) -> None:
        global ArangoClient
        try:
            from arango.client import ArangoClient
        except ImportError:
            raise LangroidImportError("python-arango", "arango")

    def _has_any_data(self) -> bool:
        for c in self.db.collections():  # type: ignore
            if c["name"].startswith("_"):
                continue
            if self.db.collection(c["name"]).count() > 0:  # type: ignore
                return True
        return False

    def _initialize_db(self) -> None:
        try:
            logger.info("Initializing ArangoDB client connection...")
            self.client = self.config.arango_settings.client or ArangoClient(
                hosts=self.config.arango_settings.url
            )

            logger.info("Connecting to database...")
            self.db = self.config.arango_settings.db or self.client.db(
                self.config.arango_settings.database,
                username=self.config.arango_settings.username,
                password=self.config.arango_settings.password,
            )

            logger.info("Checking for existing data in collections...")
            # Check if any non-system collection has data
            self.config.database_created = self._has_any_data()

            # If database has data, get schema
            if self.config.database_created:
                logger.info("Database has existing data, retrieving schema...")
                # this updates self.config.kg_schema
                self.arango_schema_tool(None)
            else:
                logger.info("No existing data found in database")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise ConnectionError(f"Failed to initialize ArangoDB connection: {e}")

    def close(self) -> None:
        if self.client:
            self.client.close()

    @staticmethod
    def cleanup_graph_db(db) -> None:  # type: ignore
        # First delete graphs to properly handle edge collections
        for graph in db.graphs():
            graph_name = graph["name"]
            if not graph_name.startswith("_"):  # Skip system graphs
                try:
                    db.delete_graph(graph_name)
                except Exception as e:
                    print(f"Failed to delete graph {graph_name}: {e}")

        # Clear existing collections
        for collection in db.collections():
            if not collection["name"].startswith("_"):  # Skip system collections
                try:
                    db.delete_collection(collection["name"])
                except Exception as e:
                    print(f"Failed to delete collection {collection['name']}: {e}")

    def with_retry(
        self, func: Callable[[], T], max_retries: int = 3, delay: float = 1.0
    ) -> T:
        """Execute a function with retries on connection error"""
        for attempt in range(max_retries):
            try:
                return func()
            except ArangoError:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    f"Connection failed (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                # Reconnect if needed
                self._initialize_db()
        return func()  # Final attempt after loop if not raised

    def read_query(
        self, query: str, bind_vars: Optional[Dict[Any, Any]] = None
    ) -> QueryResult:
        """Execute a read query with connection retry."""
        if not self.db:
            return QueryResult(
                success=False, data="No database connection is established."
            )

        def execute_read() -> QueryResult:
            try:
                cursor = self.db.aql.execute(query, bind_vars=bind_vars)
                records = [doc for doc in cursor]  # type: ignore
                records = records[: self.config.max_num_results]
                logger.warning(f"Records retrieved: {records}")
                return QueryResult(success=True, data=records if records else [])
            except Exception as e:
                if isinstance(e, ServerConnectionError):
                    raise
                logger.error(f"Failed to execute query: {query}\n{e}")
                error_message = self.retry_query(e, query)
                return QueryResult(success=False, data=error_message)

        try:
            return self.with_retry(execute_read)  # type: ignore
        except Exception as e:
            return QueryResult(
                success=False, data=f"Failed after max retries: {str(e)}"
            )

    def write_query(
        self, query: str, bind_vars: Optional[Dict[Any, Any]] = None
    ) -> QueryResult:
        """Execute a write query with connection retry."""
        if not self.db:
            return QueryResult(
                success=False, data="No database connection is established."
            )

        def execute_write() -> QueryResult:
            try:
                self.db.aql.execute(query, bind_vars=bind_vars)
                return QueryResult(success=True)
            except Exception as e:
                if isinstance(e, ServerConnectionError):
                    raise
                logger.error(f"Failed to execute query: {query}\n{e}")
                error_message = self.retry_query(e, query)
                return QueryResult(success=False, data=error_message)

        try:
            return self.with_retry(execute_write)  # type: ignore
        except Exception as e:
            return QueryResult(
                success=False, data=f"Failed after max retries: {str(e)}"
            )

    def aql_retrieval_tool(self, msg: AQLRetrievalTool) -> str:
        """Handle AQL query for data retrieval"""
        if not self.tried_schema:
            return f"""
            You need to use `{arango_schema_tool_name}` first to get the 
            database schema before using `{aql_retrieval_tool_name}`. This ensures
            you know the correct collection names and edge definitions.
            """
        elif not self.config.database_created:
            return """
            You need to create the database first using `{aql_creation_tool_name}`.
            """
        self.num_tries += 1
        query = msg.aql_query
        if query == self.current_retrieval_aql_query:
            return """
            You have already tried this query, so you will get the same results again!
            If you need to retry, please MODIFY the query to get different results.
            """
        self.current_retrieval_aql_query = query
        logger.info(f"Executing AQL query: {query}")
        response = self.read_query(query)

        if isinstance(response.data, list) and len(response.data) == 0:
            return """
            No results found. Check if your collection names are correct - 
            they are case-sensitive. Use exact names from the schema.
            Try modifying your query based on the RETRY-SUGGESTIONS 
            in your instructions.
            """
        return str(response.data)

    def aql_creation_tool(self, msg: AQLCreationTool) -> str:
        """Handle AQL query for creating data"""
        self.num_tries += 1
        query = msg.aql_query
        logger.info(f"Executing AQL query: {query}")
        response = self.write_query(query)

        if response.success:
            self.config.database_created = True
            return "AQL query executed successfully"
        return str(response.data)

    def arango_schema_tool(
        self,
        msg: ArangoSchemaTool | None,
    ) -> Dict[str, List[Dict[str, Any]]] | str:
        """Get database schema. If collections=None, include all collections.
        If properties=False, show only connection info,
        else show all properties and example-docs.
        """

        if (
            msg is not None
            and msg.collections == self.current_schema_params.collections
            and msg.properties == self.current_schema_params.properties
        ):
            return """
            You have already tried this schema TOOL, so you will get the same results 
            again! Please MODIFY the tool params `collections` or `properties` to get
            different results.
            """

        if msg is not None:
            collections = msg.collections
            properties = msg.properties
        else:
            collections = None
            properties = True
        self.tried_schema = True
        if (
            self.config.kg_schema is not None
            and len(self.config.kg_schema) > 0
            and msg is None
        ):
            # we are trying to pre-populate full schema before the agent runs,
            # so get it if it's already available
            # (Note of course that this "full schema" may actually be incomplete)
            return self.config.kg_schema

        # increment tries only if the LLM is asking for the schema,
        # in which case msg will not be None
        self.num_tries += msg is not None

        try:
            # Get graph schemas (keeping full graph info)
            graph_schema = [
                {"graph_name": g["name"], "edge_definitions": g["edge_definitions"]}
                for g in self.db.graphs()  # type: ignore
            ]

            # Get collection schemas
            collection_schema = []
            for collection in self.db.collections():  # type: ignore
                if collection["name"].startswith("_"):
                    continue

                col_name = collection["name"]
                if collections and col_name not in collections:
                    continue

                col_type = collection["type"]
                col_size = self.db.collection(col_name).count()

                if col_size == 0:
                    continue

                if properties:
                    # Full property collection with sampling
                    lim = self.config.schema_sample_pct * col_size  # type: ignore
                    limit_amount = ceil(lim / 100.0) or 1
                    sample_query = f"""
                        FOR doc in {col_name}
                        LIMIT {limit_amount}
                        RETURN doc
                    """

                    properties_list = []
                    example_doc = None

                    def simplify_doc(doc: Any) -> Any:
                        if isinstance(doc, list) and len(doc) > 0:
                            return [simplify_doc(doc[0])]
                        if isinstance(doc, dict):
                            return {k: simplify_doc(v) for k, v in doc.items()}
                        return doc

                    for doc in self.db.aql.execute(sample_query):  # type: ignore
                        if example_doc is None:
                            example_doc = simplify_doc(doc)
                        for key, value in doc.items():
                            prop = {"name": key, "type": type(value).__name__}
                            if prop not in properties_list:
                                properties_list.append(prop)

                    collection_schema.append(
                        {
                            "collection_name": col_name,
                            "collection_type": col_type,
                            f"{col_type}_properties": properties_list,
                            f"example_{col_type}": example_doc,
                        }
                    )
                else:
                    # Basic info + from/to for edges only
                    collection_info = {
                        "collection_name": col_name,
                        "collection_type": col_type,
                    }
                    if col_type == "edge":
                        # Get a sample edge to extract from/to fields
                        sample_edge = next(
                            self.db.aql.execute(  # type: ignore
                                f"FOR e IN {col_name} LIMIT 1 RETURN e"
                            ),
                            None,
                        )
                        if sample_edge:
                            collection_info["from_collection"] = sample_edge[
                                "_from"
                            ].split("/")[0]
                            collection_info["to_collection"] = sample_edge["_to"].split(
                                "/"
                            )[0]

                    collection_schema.append(collection_info)

            schema = {
                "Graph Schema": graph_schema,
                "Collection Schema": collection_schema,
            }
            schema_str = json.dumps(schema, indent=2)
            logger.warning(f"Schema retrieved:\n{schema_str}")
            with open("logs/arango-schema.json", "w") as f:
                f.write(schema_str)
            if (n_fields := count_fields(schema)) > self.config.max_schema_fields:
                logger.warning(
                    f"""
                    Schema has {n_fields} fields, which exceeds the maximum of
                    {self.config.max_schema_fields}. Showing a trimmed version
                    that only includes edge info and no other properties.
                    """
                )
                schema = trim_schema(schema)
                n_fields = count_fields(schema)
                logger.warning(f"Schema trimmed down to {n_fields} fields.")
                schema_str = (
                    json.dumps(schema)
                    + "\n"
                    + f"""
                    
                    CAUTION: The requested schema was too large, so 
                    the schema has been trimmed down to show only all collection names,
                    their types, 
                    and edge relationships (from/to collections) without any properties.
                    To find out more about the schema, you can EITHER:
                    - Use the `{arango_schema_tool_name}` tool again with the 
                      `properties` arg set to True, and `collections` arg set to
                        specific collections you want to know more about, OR
                    - Use the `{aql_retrieval_tool_name}` tool to learn more about
                      the schema by querying the database.
                      
                    """
                )
                if msg is None:
                    self.config.kg_schema = schema_str
                return schema_str
            self.config.kg_schema = schema
            return schema

        except Exception as e:
            logger.error(f"Schema retrieval failed: {str(e)}")
            return f"Failed to retrieve schema: {str(e)}"

    def _init_tools_sys_message(self) -> None:
        """Initialize system msg and enable tools"""
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
        # self.config.prepopulate_schema is True or False, because
        # even when schema provided, the agent may later want to get the schema,
        # e.g. if the db evolves, or schema was trimmed due to size, or
        # if it needs to bring in the schema into recent context.

        self.enable_message(
            [
                ArangoSchemaTool,
                AQLRetrievalTool,
                AQLCreationTool,
                ForwardTool,
            ]
        )
        if not self.config.chat_mode:
            self.enable_message(DoneTool)

    def _format_message(self) -> str:
        if self.db is None:
            raise ValueError("Database connection not established")

        assert isinstance(self.config, ArangoChatAgentConfig)
        return (
            SCHEMA_TOOLS_SYS_MSG
            if not self.config.prepopulate_schema
            else SCHEMA_PROVIDED_SYS_MSG.format(schema=self.arango_schema_tool(None))
        )

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ForwardTool | None:
        """When LLM sends a no-tool msg, assume user is the intended recipient,
        and if in interactive mode, forward the msg to the user.
        """
        done_tool_name = DoneTool.default_value("request")
        forward_tool_name = ForwardTool.default_value("request")
        aql_retrieval_tool_instructions = AQLRetrievalTool.instructions()
        # TODO the aql_retrieval_tool_instructions may be empty/minimal
        # when using self.config.use_functions_api = True.
        tools_instruction = f"""
          For example you may want to use the TOOL
          `{aql_retrieval_tool_name}`  according to these instructions:
           {aql_retrieval_tool_instructions}
        """
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
                      {tools_instruction}
                    """
                return f"""
                The intent of your response is not clear:
                - if you intended this to be the FINAL answer to the user's query,
                    then use the `{done_tool_name}` to indicate so,
                    with the `content` set to the answer or result.
                - otherwise, use one of the available tools to make progress 
                    to arrive at the final answer.
                    {tools_instruction}
                """
        return None

    def retry_query(self, e: Exception, query: str) -> str:
        """Generate error message for failed AQL query"""
        logger.error(f"AQL Query failed: {query}\nException: {e}")

        error_message = f"""\
        {ARANGO_ERROR_MSG}: '{query}'
        {str(e)}
        Please try again with a corrected query.
        """

        return error_message
