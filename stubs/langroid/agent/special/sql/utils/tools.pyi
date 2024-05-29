from langroid.agent.tool_message import ToolMessage as ToolMessage

class RunQueryTool(ToolMessage):
    request: str
    purpose: str
    query: str

class GetTableNamesTool(ToolMessage):
    request: str
    purpose: str

class GetTableSchemaTool(ToolMessage):
    request: str
    purpose: str
    tables: list[str]
    @classmethod
    def example(cls) -> GetTableSchemaTool: ...

class GetColumnDescriptionsTool(ToolMessage):
    request: str
    purpose: str
    table: str
    columns: str
    @classmethod
    def example(cls) -> GetColumnDescriptionsTool: ...
