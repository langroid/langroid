from typing import Protocol

from langroid.agent.tool_message import ToolMessage as ToolMessage
from langroid.language_models.base import LLMMessage as LLMMessage

def extract_between(value: str, start_word: str, end_word: str) -> str: ...

class HasMessageHistory(Protocol):
    message_history: list[LLMMessage]

class ExtractTool(ToolMessage):
    request: str
    purpose: str
    jinja_template: str
    @classmethod
    def instructions(cls) -> str: ...
    def handle(self): ...
