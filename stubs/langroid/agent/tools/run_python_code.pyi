from langroid.agent.tool_message import ToolMessage as ToolMessage

class RunPythonCodeTool(ToolMessage):
    request: str
    purpose: str
    code: str
    @classmethod
    def examples(cls) -> list["ToolMessage"]: ...
    def handle(self) -> str: ...
