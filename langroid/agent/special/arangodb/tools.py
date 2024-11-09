from langroid.agent.tool_message import ToolMessage


class AQLRetrievalTool(ToolMessage):
    request: str = "aql_retrieval_tool"
    purpose: str = """
        To send an <aql_query> in response to a user's request/question, 
        and WAIT for results of the <aql_query> BEFORE continuing with response.
        You will receive RESULTS from this tool, and ONLY THEN you can continue.
    """
    aql_query: str


aql_retrieval_tool_name = AQLRetrievalTool.default_value("request")


class AQLCreationTool(ToolMessage):
    request: str = "aql_creation_tool"
    purpose: str = """
        To send the <aql_query> to create documents/edges in the graph database.
        IMPORTANT: YOU MUST WAIT FOR THE RESULT OF THE TOOL BEFORE CONTINUING.
        You will receive RESULTS from this tool, and ONLY THEN you can continue.
    """
    aql_query: str


aql_creation_tool_name = AQLCreationTool.default_value("request")


class ArangoSchemaTool(ToolMessage):
    request: str = "arango_schema_tool"
    purpose: str = """
        To get the schema of the Arango graph database.
        IMPORTANT: YOU MUST WAIT FOR THE RESULT OF THE TOOL BEFORE CONTINUING.
        You will receive RESULTS from this tool, and ONLY THEN you can continue.
    """


arango_schema_tool_name = ArangoSchemaTool.default_value("request")
