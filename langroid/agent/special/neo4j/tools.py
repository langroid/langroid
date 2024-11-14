from langroid.agent import ToolMessage


class CypherRetrievalTool(ToolMessage):
    request: str = "cypher_retrieval_tool"
    purpose: str = """To send the <cypher_query> to retrieve 
        data from the graph database based on provided text description and schema.
        """
    cypher_query: str


cypher_retrieval_tool_name = CypherRetrievalTool.default_value("request")


class CypherCreationTool(ToolMessage):
    request: str = "cypher_creation_tool"
    purpose: str = """
        To send the <cypher_query> to create 
        entities/relationships in the graph database.
        """
    cypher_query: str


cypher_creation_tool_name = CypherCreationTool.default_value("request")


class GraphSchemaTool(ToolMessage):
    request: str = "graph_schema_tool"
    purpose: str = """To get the schema of the graph database."""


graph_schema_tool_name = GraphSchemaTool.default_value("request")
