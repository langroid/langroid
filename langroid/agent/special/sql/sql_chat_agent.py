"""
Agent that allows interaction with an SQL database using SQLAlchemy library.
The agent can execute SQL queries in the database and return the result.

Functionality includes:
- adding table and column context
- asking a question about a SQL schema
"""

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from rich.console import Console

from langroid.exceptions import LangroidImportError
from langroid.mytypes import Entity
from langroid.utils.constants import SEND_TO

try:
    from sqlalchemy import MetaData, Row, create_engine, inspect, text
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import ResourceClosedError, SQLAlchemyError
    from sqlalchemy.orm import Session, sessionmaker
except ImportError as e:
    raise LangroidImportError(extra="sql", error=str(e))

try:
    # sqlglot is required for the statement-type allowlist enforced in
    # `_validate_query`. Importing it at module load ensures the security
    # guarantee cannot be silently bypassed by a partial/stale install.
    import sqlglot
    from sqlglot import expressions as sqlglot_exp
except ImportError as e:
    raise LangroidImportError(extra="sql", error=str(e))

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
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
from langroid.agent.tools.orchestration import (
    DonePassTool,
    DoneTool,
    ForwardTool,
    PassTool,
)
from langroid.language_models.base import Role
from langroid.vector_store.base import VectorStoreConfig

logger = logging.getLogger(__name__)

console = Console()

DEFAULT_SQL_CHAT_SYSTEM_MESSAGE = """
{mode}

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

"""

ADDRESSING_INSTRUCTION = """
IMPORTANT - Whenever you are NOT writing a SQL query, make sure you address the user
using {prefix}User (NO SPACE between {prefix} and User). 
You MUST use the EXACT syntax {prefix}User !!!

In other words, you ALWAYS write EITHER:
 - a SQL query using the `run_query` tool, 
 - OR address the user using {prefix}User
"""

DONE_INSTRUCTION = f"""
When you are SURE you have the CORRECT answer to a user's query or request, 
use the `{DoneTool.name()}` with `content` set to the answer or result.
If you DO NOT think you have the answer to the user's query or request,
you SHOULD NOT use the `{DoneTool.name()}` tool.
Instead, you must CONTINUE to improve your queries (tools) to get the correct answer,
and finally use the `{DoneTool.name()}` tool to send the correct answer to the user.
"""


SQL_ERROR_MSG = "There was an error in your SQL Query"


# Dialect-specific SQL patterns that enable code execution, arbitrary file
# access, or other escapes from the database engine. Matched against the raw
# query text (case-insensitive) as a defense-in-depth layer in addition to the
# sqlglot-based statement-type allowlist.
_DANGEROUS_SQL_PATTERNS: List["re.Pattern[str]"] = [
    # PostgreSQL: COPY ... FROM/TO PROGRAM executes shell commands as the DB
    # server OS user. This is the primitive used in CVE-2026-25879.
    re.compile(r"\bcopy\b[\s\S]*\bprogram\b", re.IGNORECASE),
    # PostgreSQL server-side filesystem access
    re.compile(r"\bpg_read_server_files?\b", re.IGNORECASE),
    re.compile(r"\bpg_read_binary_file\b", re.IGNORECASE),
    re.compile(r"\bpg_ls_dir\b", re.IGNORECASE),
    re.compile(r"\blo_(import|export)\b", re.IGNORECASE),
    # MySQL/MariaDB filesystem
    re.compile(r"\binto\s+(outfile|dumpfile)\b", re.IGNORECASE),
    re.compile(r"\bload_file\s*\(", re.IGNORECASE),
    re.compile(r"\bload\s+data\b", re.IGNORECASE),
    # SQLite: load_extension enables loading arbitrary shared objects;
    # ATTACH DATABASE can read/write arbitrary files.
    re.compile(r"\bload_extension\s*\(", re.IGNORECASE),
    re.compile(r"\battach\s+database\b", re.IGNORECASE),
    # SQL Server: command execution and OLE automation
    re.compile(r"\bxp_cmdshell\b", re.IGNORECASE),
    re.compile(r"\bsp_oacreate\b", re.IGNORECASE),
    re.compile(r"\bsp_oamethod\b", re.IGNORECASE),
    re.compile(r"\bopenrowset\b", re.IGNORECASE),
    re.compile(r"\bbulk\s+insert\b", re.IGNORECASE),
    # Generic: stored-program creation and procedural language extensions
    re.compile(
        r"\bcreate\s+(or\s+replace\s+)?(function|procedure|trigger)\b", re.IGNORECASE
    ),
    re.compile(r"\bcreate\s+extension\b", re.IGNORECASE),
]


# Default set of SQL statement types the agent is allowed to execute when
# `allow_dangerous_operations` is False. SELECT-only is safe for Q&A workloads.
_DEFAULT_ALLOWED_STATEMENTS: List[str] = ["SELECT"]


class SQLChatAgentConfig(ChatAgentConfig):
    system_message: str = DEFAULT_SQL_CHAT_SYSTEM_MESSAGE
    user_message: None | str = None
    cache: bool = True  # cache results
    debug: bool = False
    use_helper: bool = True
    is_helper: bool = False
    stream: bool = True  # allow streaming where needed
    database_uri: str = ""  # Database URI
    database_session: None | Session = None  # Database session
    vecdb: None | VectorStoreConfig = None
    context_descriptions: Dict[str, Dict[str, Union[str, Dict[str, str]]]] = {}
    use_schema_tools: bool = False
    multi_schema: bool = False
    # whether the agent is used in a continuous chat with user,
    # as opposed to returning a result from the task.run()
    chat_mode: bool = False
    addressing_prefix: str = ""
    max_result_rows: int | None = None  # limit query results to this
    max_retained_tokens: int | None = None  # limit history of query results to this

    # --- Security controls (see CVE-2026-25879) ---------------------------
    # By default, the agent only executes SELECT statements and rejects any
    # query that matches a known dangerous pattern (e.g. PostgreSQL
    # `COPY ... FROM PROGRAM`, MySQL `INTO OUTFILE`, SQLite `load_extension`,
    # MSSQL `xp_cmdshell`). The LLM-generated SQL is influenceable by prompt
    # injection — including injection via data the LLM reads back from the
    # database — so executing it without restrictions is unsafe when the DB
    # role has elevated privileges.
    #
    # To enable writes: extend allowed_statement_types, e.g. ["SELECT",
    # "INSERT", "UPDATE", "DELETE"]. To disable all checks (only do this with
    # a least-privilege DB role and trusted prompts): set
    # allow_dangerous_operations=True.
    allowed_statement_types: List[str] = list(_DEFAULT_ALLOWED_STATEMENTS)
    allow_dangerous_operations: bool = False

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
        self.final_instructions = ""

        # Caution - this updates the self.config.system_message!
        self._init_system_message()
        super().__init__(config)
        self._init_tools()
        if self.config.is_helper:
            self.system_tool_format_instructions += self.final_instructions

        if self.config.use_helper:
            # helper_config.system_message is now the fully-populated sys msg of
            # the main SQLAgent.
            self.helper_config = self.config.model_copy()
            self.helper_config.is_helper = True
            self.helper_config.use_helper = False
            self.helper_config.chat_mode = False
            self.helper_agent = SQLHelperAgent(self.helper_config)

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

    def _init_system_message(self) -> None:
        """Initialize the system message."""
        message = self._format_message()
        self.config.system_message = self.config.system_message.format(mode=message)

        if not self.config.allow_dangerous_operations:
            allowed = sorted(
                t.strip().upper() for t in self.config.allowed_statement_types
            )
            self.config.system_message += (
                f"\n\nIMPORTANT - SECURITY POLICY:\n"
                f"You may ONLY issue SQL queries whose top-level statement "
                f"type is one of: {allowed}. Any other statement type "
                f"(e.g. DDL, COPY, EXEC, multi-statement scripts that "
                f"include a disallowed type) will be REJECTED by the "
                f"executor and not run.\n"
            )

        if self.config.chat_mode:
            self.config.addressing_prefix = self.config.addressing_prefix or SEND_TO
            self.config.system_message += ADDRESSING_INSTRUCTION.format(
                prefix=self.config.addressing_prefix
            )
        else:
            self.config.system_message += DONE_INSTRUCTION

    def _init_tools(self) -> None:
        """Initialize sys msg and tools."""
        # Create a custom RunQueryTool class with the desired max_retained_tokens
        if self.config.max_retained_tokens is not None:

            class CustomRunQueryTool(RunQueryTool):
                _max_retained_tokens = self.config.max_retained_tokens

            self.enable_message([CustomRunQueryTool, ForwardTool])
        else:
            self.enable_message([RunQueryTool, ForwardTool])

        if self.config.use_schema_tools:
            self._enable_schema_tools()
        if not self.config.chat_mode:
            self.enable_message(DoneTool)
            self.enable_message(DonePassTool)

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
        self.used_run_query = False
        return super().llm_response(message)

    def user_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        self.llm_responded = False
        self.used_run_query = False
        return super().user_response(msg)

    def _clarify_answer_instruction(self) -> str:
        """
        Prompt to use when asking LLM to clarify intent of
        an already-generated response
        """
        if self.config.chat_mode:
            return f"""
                you must use the TOOL `{ForwardTool.name()}` with the `agent` 
                parameter set to "User"
                """
        else:
            return f"you must use the TOOL `{DonePassTool.name()}`"

    def _clarifying_message(self) -> str:
        tools_instruction = f"""
          For example you may want to use the TOOL
          `{RunQueryTool.name()}` to further explore the database contents
        """
        if self.config.use_schema_tools:
            tools_instruction += """
            OR you may want to use one of the schema tools to 
            explore the database schema
            """
        return f"""
            The intent of your response is not clear:
            - if you intended this to be the FINAL answer to the user's query,
                {self._clarify_answer_instruction()}
            - otherwise, use one of the available tools to make progress 
                to arrive at the final answer.
                {tools_instruction}
            """

    def handle_message_fallback(
        self, message: str | ChatDocument
    ) -> str | ForwardTool | ChatDocument | None:
        """
        We'd end up here if the current msg has no tool.
        If this is from LLM, we may need to handle the scenario where
        it may have "forgotten" to generate a tool.
        """
        if (
            not isinstance(message, ChatDocument)
            or message.metadata.sender != Entity.LLM
        ):
            return None
        if self.config.chat_mode:
            # send any Non-tool msg to the user
            return ForwardTool(agent="User")
        # Agent intent not clear => use the helper agent to
        # do what this agent should have done, e.g. generate tool, etc.
        # This is likelier to succeed since this agent has no "baggage" of
        # prior conversation, other than the system msg, and special
        # "Intent-interpretation" instructions.
        if self._json_schema_available() and self.config.strict_recovery:
            AnyTool = self._get_any_tool_message(optional=False)
            self.set_output_format(
                AnyTool,
                force_tools=True,
                use=True,
                handle=True,
                instructions=True,
            )
            recovery_message = self._strict_recovery_instructions(
                AnyTool, optional=False
            )
            result = self.llm_response(recovery_message)
            # remove the recovery_message (it has User role) from the chat history,
            # else it may cause the LLM to directly use the AnyTool.
            self.delete_last_message(role=Role.USER)  # delete last User-role msg
            return result
        elif self.config.use_helper:
            response = self.helper_agent.llm_response(message)
            tools = self.try_get_tool_messages(response)
            if tools:
                return response
        # fall back on the clarification message
        return self._clarifying_message()

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

    def _available_tool_names(self) -> str:
        return ",".join(self.llm_tools_usable)

    def _tool_result_llm_answer_prompt(self) -> str:
        """
        Prompt to use at end of tool result,
        to guide LLM, for the case where it wants to answer the user's query
        """
        if self.config.chat_mode:
            assert self.config.addressing_prefix != ""
            return """
                You must EXPLICITLY address the User with 
                the addressing prefix according to your instructions,
                to convey your answer to the User.
                """
        else:
            return f"""
                you must use the `{DoneTool.name()}` with the `content` 
                set to the answer or result
                """

    def _sqlglot_dialect(self) -> Optional[str]:
        """Map the SQLAlchemy dialect name to a sqlglot dialect name."""
        if self.engine is None:
            return None
        name: str = str(self.engine.dialect.name)
        # sqlglot uses 'postgres', not 'postgresql'; 'tsql' for MSSQL.
        mapping: Dict[str, str] = {"postgresql": "postgres", "mssql": "tsql"}
        return mapping.get(name, name)

    def _validate_query(self, query: str) -> Optional[str]:
        """
        Check whether `query` is permitted under the agent's security config.

        Returns None if the query may be executed, otherwise an error message
        explaining why it was rejected (to be relayed to the LLM).
        """
        if self.config.allow_dangerous_operations:
            return None

        for pat in _DANGEROUS_SQL_PATTERNS:
            if pat.search(query):
                logger.warning(
                    "SQLChatAgent rejected query matching dangerous pattern "
                    f"{pat.pattern!r}: {query!r}"
                )
                return (
                    f"Query REJECTED for safety: it matches a pattern "
                    f"({pat.pattern!r}) that enables code execution, "
                    f"filesystem access, or other unsafe operations. "
                    f"Rewrite the query without using this construct, or ask "
                    f"the operator to set `allow_dangerous_operations=True` "
                    f"on the SQLChatAgent config."
                )

        allowed = {t.strip().upper() for t in self.config.allowed_statement_types}
        try:
            statements = sqlglot.parse(query, read=self._sqlglot_dialect())
        except Exception as e:
            logger.warning(f"sqlglot failed to parse query {query!r}: {e}")
            return (
                f"Query REJECTED for safety: could not be parsed to verify it "
                f"is a {sorted(allowed)} statement ({e}). Rewrite the query "
                f"more simply, or ask the operator to set "
                f"`allow_dangerous_operations=True` on the SQLChatAgent config."
            )

        kind_map = {
            sqlglot_exp.Select: "SELECT",
            sqlglot_exp.Insert: "INSERT",
            sqlglot_exp.Update: "UPDATE",
            sqlglot_exp.Delete: "DELETE",
            sqlglot_exp.Merge: "MERGE",
            sqlglot_exp.Create: "CREATE",
            sqlglot_exp.Drop: "DROP",
            sqlglot_exp.Alter: "ALTER",
            sqlglot_exp.TruncateTable: "TRUNCATE",
            sqlglot_exp.Command: "COMMAND",
        }
        for stmt in statements:
            if stmt is None:
                continue
            kind = next(
                (v for k, v in kind_map.items() if isinstance(stmt, k)),
                type(stmt).__name__.upper(),
            )
            if kind not in allowed:
                logger.warning(
                    f"SQLChatAgent rejected {kind} statement (allowed: "
                    f"{sorted(allowed)}): {query!r}"
                )
                return (
                    f"Query REJECTED for safety: statement type {kind!r} is "
                    f"not in the allowed list {sorted(allowed)}. Rewrite the "
                    f"query as one of the allowed statement types, or ask "
                    f"the operator to extend `allowed_statement_types` "
                    f"(or set `allow_dangerous_operations=True`) on the "
                    f"SQLChatAgent config."
                )
        return None

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

        rejection = self._validate_query(query)
        if rejection is not None:
            return f"""
        Below is the result from your use of the TOOL `{RunQueryTool.name()}`:
        ==== result ====
        {rejection}
        ================

        Try a different query that complies with the policy above.
        """

        try:
            logger.info(f"Executing SQL query: {query}")

            query_result = session.execute(text(query))
            session.commit()
            try:
                # attempt to fetch results: should work for normal SELECT queries
                rows = query_result.fetchall()
                n_rows = len(rows)
                if self.config.max_result_rows and n_rows > self.config.max_result_rows:
                    rows = rows[: self.config.max_result_rows]
                    logger.warning(
                        f"SQL query produced {n_rows} rows, "
                        f"limiting to {self.config.max_result_rows}"
                    )

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

        final_message = f"""
        Below is the result from your use of the TOOL `{RunQueryTool.name()}`:
        ==== result ====
        {response_message}
        ================
        
        If you are READY to ANSWER the ORIGINAL QUERY:
        {self._tool_result_llm_answer_prompt()}
        OTHERWISE:
             continue using one of your available TOOLs:
             {",".join(self.llm_tools_usable)}
        """
        return final_message

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


class SQLHelperAgent(SQLChatAgent):

    def _clarifying_message(self) -> str:
        tools_instruction = f"""
          For example the Agent may have forgotten to use the TOOL
          `{RunQueryTool.name()}` to further explore the database contents
        """
        if self.config.use_schema_tools:
            tools_instruction += """
            OR the agent may have forgotten to use one of the schema tools to 
            explore the database schema
            """

        return f"""
            The intent of the Agent's response is not clear:
            - if you think the Agent intended this as ANSWER to the 
                user's query,
                {self._clarify_answer_instruction()}
            - otherwise, the Agent may have forgotten to 
              use one of the available tools to make progress 
                to arrive at the final answer.
                {tools_instruction}
            """

    def _init_system_message(self) -> None:
        """Set up helper sys msg"""

        # Note that self.config.system_message is already set to the
        # parent SQLAgent's system_message
        self.config.system_message = f"""
                You role is to help INTERPRET the INTENT of an 
                AI agent in a conversation. This Agent was supposed to generate
                a TOOL/Function-call but forgot to do so, and this is where 
                you can help, by trying to generate the appropriate TOOL
                based on your best guess of the Agent's INTENT.
                
                Below are the instructions that were given to this Agent: 
                ===== AGENT INSTRUCTIONS =====
                {self.config.system_message}
                ===== END OF AGENT INSTRUCTIONS =====
                """

        # note that the initial msg in chat history will contain:
        # - system message
        # - tool instructions
        # so the final_instructions will be at the end of this initial msg

        self.final_instructions = f"""        
        You must take note especially of the TOOLs that are
        available to the Agent. Your reasoning process should be as follows:
        
        - If the Agent's message appears to be an ANSWER to the original query,
          {self._clarify_answer_instruction()}.
          CAUTION - You must be absolutely sure that the Agent's message is 
          an ACTUAL ANSWER to the user's query, and not a failed attempt to use 
          a TOOL without JSON, e.g. something like "run_query" or "done_tool"
          without any actual JSON formatting.
           
        - Else, if you think the Agent intended to use some type of SQL
          query tool to READ or UPDATE the table(s), 
          AND it is clear WHICH TOOL is intended as well as the 
          TOOL PARAMETERS, then you must generate the JSON-Formatted
          TOOL with the parameters set based on your understanding.
          Note that the `{RunQueryTool.name()}` is not ONLY for querying the tables,
          but also for UPDATING the tables.
           
        - Else, use the `{PassTool.name()}` to pass the message unchanged.
            CAUTION - ONLY use `{PassTool.name()}` if you think the Agent's response
            is NEITHER an ANSWER, nor an intended SQL QUERY.
        """

    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        if message is None:
            return None
        message_str = message if isinstance(message, str) else message.content
        instruc_msg = f"""
        Below is the MESSAGE from the SQL Agent. 
        Remember your instructions on how to respond based on your understanding
        of the INTENT of this message:        
        {self.final_instructions}
        
        === AGENT MESSAGE =========
        {message_str}
        === END OF AGENT MESSAGE ===
        """
        # user response_forget to avoid accumulating the chat history
        return super().llm_response_forget(instruc_msg)
