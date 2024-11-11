from typing import List, Tuple

from langroid.agent.tool_message import ToolMessage


class AQLRetrievalTool(ToolMessage):
    request: str = "aql_retrieval_tool"
    purpose: str = """
        To send an <aql_query> in response to a user's request/question, 
        OR to find SCHEMA information,
        and WAIT for results of the <aql_query> BEFORE continuing with response.
        You will receive RESULTS from this tool, and ONLY THEN you can continue.
    """
    aql_query: str

    _max_result_tokens = 500
    _max_retained_tokens = 200

    @classmethod
    def examples(cls) -> List[ToolMessage | Tuple[str, ToolMessage]]:
        """Few-shot examples to include in tool instructions."""
        return [
            (
                "I want to see who Bob's Father is",
                cls(
                    aql_query="""
                    FOR v, e, p IN 1..1 OUTBOUND 'users/Bob' GRAPH 'family_tree'
                    FILTER p.edges[0].type == 'father'
                    RETURN v
                    """
                ),
            ),
            (
                "I want to know the properties of the Actor node",
                cls(
                    aql_query="""
                    FOR doc IN Actor
                    LIMIT 1
                    RETURN ATTRIBUTES(doc)                    
                    """
                ),
            ),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        When using this TOOL/Function-call, you must WAIT to receive the RESULTS 
        of the AQL query, before continuing your response!
        DO NOT ASSUME YOU KNOW THE RESULTs BEFORE RECEIVING THEM.        
        """


aql_retrieval_tool_name = AQLRetrievalTool.default_value("request")


class AQLCreationTool(ToolMessage):
    request: str = "aql_creation_tool"
    purpose: str = """
        To send the <aql_query> to create documents/edges in the graph database.
        IMPORTANT: YOU MUST WAIT FOR THE RESULT OF THE TOOL BEFORE CONTINUING.
        You will receive RESULTS from this tool, and ONLY THEN you can continue.
    """
    aql_query: str

    @classmethod
    def examples(cls) -> List[ToolMessage | Tuple[str, ToolMessage]]:
        """Few-shot examples to include in tool instructions."""
        return [
            (
                "Create a new document in the collection 'users'",
                cls(
                    aql_query="""
                    INSERT {
                      "name": "Alice",
                      "age": 30
                    } INTO users
                    """
                ),
            ),
        ]


aql_creation_tool_name = AQLCreationTool.default_value("request")


class ArangoSchemaTool(ToolMessage):
    request: str = "arango_schema_tool"
    purpose: str = """
        To get the schema of the Arango graph database,
        or some part of it. Follow these instructions:
        1. Set <properties> to True to get the properties of the collections,
        and False if you only want to see the graph structure and get only the
        from/to relations of the edges.
        2. Set <collections> to a list of collection names if you want to see,
        or leave it as None to see all ALL collections.
        IMPORTANT: YOU MUST WAIT FOR THE RESULT OF THE TOOL BEFORE CONTINUING.
        You will receive RESULTS from this tool, and ONLY THEN you can continue.
    """

    properties: bool = True
    collections: List[str] | None = None

    _max_result_tokens = 500


arango_schema_tool_name = ArangoSchemaTool.default_value("request")
