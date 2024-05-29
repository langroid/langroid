from _typeshed import Incomplete

from langroid.agent.chat_agent import ChatAgent as ChatAgent
from langroid.agent.chat_document import (
    ChatDocMetaData as ChatDocMetaData,
)
from langroid.agent.chat_document import (
    ChatDocument as ChatDocument,
)
from langroid.agent.tool_message import ToolMessage as ToolMessage
from langroid.mytypes import Entity as Entity
from langroid.utils.pydantic_utils import has_field as has_field

class AddRecipientTool(ToolMessage):
    request: str
    purpose: str
    intended_recipient: str
    saved_content: str

    class Config:
        schema_extra: Incomplete

    def response(self, agent: ChatAgent) -> ChatDocument: ...

class RecipientTool(ToolMessage):
    request: str
    purpose: str
    intended_recipient: str
    content: str
    @classmethod
    def create(
        cls, recipients: list[str], default: str = ""
    ) -> type["RecipientTool"]: ...
    @classmethod
    def instructions(cls) -> str: ...
    def response(self, agent: ChatAgent) -> str | ChatDocument: ...
    @staticmethod
    def handle_message_fallback(
        agent: ChatAgent, msg: str | ChatDocument
    ) -> str | ChatDocument | None: ...
