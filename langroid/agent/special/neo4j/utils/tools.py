from langroid.agent.tool_message import ToolMessage


class CypherQueryTool(ToolMessage):
    request: str = "make_query"
    purpose: str = """Use this tool to send me the Generated Cypher query based on 
    text description and schema that I will provide you."""
    cypherQuery: str


class GraphSchemaTool(ToolMessage):
    request: str = "get_schema"
    purpose: str = """Use this tool to get me the schema of the graph database."""
