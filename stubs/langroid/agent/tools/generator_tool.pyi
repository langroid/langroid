from langroid.agent.tool_message import ToolMessage as ToolMessage

class GeneratorTool(ToolMessage):
    request: str
    purpose: str
    rules: str
    def handle(self) -> None: ...
