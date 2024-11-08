from langroid.agent.tool_message import ToolMessage


class AQLRetrievalTool(ToolMessage):
    request: str = "aql_retrieval_tool"
    purpose: str = """
        To send the <aql_query> to retrieve data from the 
        graph database based on provided text description and schema.
    """
    aql_query: str


aql_retrieval_tool_name = AQLRetrievalTool.default_value("request")


class AQLCreationTool(ToolMessage):
    request: str = "aql_creation_tool"
    purpose: str = """
        To send the <aql_query> to create documents/edges in the graph database.
    """
    aql_query: str


aql_creation_tool_name = AQLCreationTool.default_value("request")


class ArangoSchemaTool(ToolMessage):
    request: str = "arango_schema_tool"
    purpose: str = """To get the schema of the Arango graph database."""


arango_schema_tool_name = ArangoSchemaTool.default_value("request")
