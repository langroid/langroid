from typing import List, Tuple

from langroid.agent.tool_message import ToolMessage


class RunQueryTool(ToolMessage):
    request: str = "run_query"
    purpose: str = """
            To run <query> on the database 'db' and 
            return the results to answer a question.
            """
    query: str

    @classmethod
    def examples(cls) -> List["ToolMessage" | Tuple[str, "ToolMessage"]]:
        return [
            cls(
                query="SELECT * FROM movies WHERE genre = 'comedy'",
            ),
            (
                "Find all movies with a rating of 5",
                cls(
                    query="SELECT * FROM movies WHERE rating = 5",
                ),
            ),
        ]


class GetTableNamesTool(ToolMessage):
    request: str = "get_table_names"
    purpose: str = """
            To retrieve the names of all <tables> in the database 'db'.
            """


class GetTableSchemaTool(ToolMessage):
    request: str = "get_table_schema"
    purpose: str = """
            To retrieve the schema of all provided <tables> in the database 'db'.
            """
    tables: List[str]

    @classmethod
    def example(cls) -> "GetTableSchemaTool":
        return cls(
            tables=["employees", "departments", "sales"],
        )


class GetColumnDescriptionsTool(ToolMessage):
    request: str = "get_column_descriptions"
    purpose: str = """
            To retrieve the description of one or more <columns> from the respective 
            <table> in the database 'db'.
            """
    table: str
    columns: str

    @classmethod
    def example(cls) -> "GetColumnDescriptionsTool":
        return cls(
            table="employees",
            columns="name, department_id",
        )
