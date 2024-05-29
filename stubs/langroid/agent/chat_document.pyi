from enum import Enum

from _typeshed import Incomplete
from pydantic import BaseModel

from langroid.agent.tool_message import ToolMessage as ToolMessage
from langroid.language_models.base import (
    LLMFunctionCall as LLMFunctionCall,
)
from langroid.language_models.base import (
    LLMMessage as LLMMessage,
)
from langroid.language_models.base import (
    LLMResponse as LLMResponse,
)
from langroid.language_models.base import (
    LLMTokenUsage as LLMTokenUsage,
)
from langroid.language_models.base import (
    Role as Role,
)
from langroid.mytypes import (
    DocMetaData as DocMetaData,
)
from langroid.mytypes import (
    Document as Document,
)
from langroid.mytypes import (
    Entity as Entity,
)
from langroid.parsing.agent_chats import parse_message as parse_message
from langroid.parsing.parse_json import (
    extract_top_level_json as extract_top_level_json,
)
from langroid.parsing.parse_json import (
    top_level_json_field as top_level_json_field,
)
from langroid.utils.output.printing import shorten_text as shorten_text

class ChatDocAttachment(BaseModel):
    class Config:
        extra: Incomplete

class StatusCode(str, Enum):
    OK: str
    ERROR: str
    DONE: str
    STALLED: str
    INF_LOOP: str
    KILL: str
    MAX_TURNS: str
    MAX_COST: str
    MAX_TOKENS: str
    TIMEOUT: str
    NO_ANSWER: str
    USER_QUIT: str

class ChatDocMetaData(DocMetaData):
    parent: ChatDocument | None
    sender: Entity
    tool_ids: list[str]
    parent_responder: None | Entity
    block: None | Entity
    sender_name: str
    recipient: str
    usage: LLMTokenUsage | None
    cached: bool
    displayed: bool
    has_citation: bool
    status: StatusCode | None

class ChatDocLoggerFields(BaseModel):
    sender_entity: Entity
    sender_name: str
    recipient: str
    block: Entity | None
    tool_type: str
    tool: str
    content: str
    @classmethod
    def tsv_header(cls) -> str: ...

class ChatDocument(Document):
    function_call: LLMFunctionCall | None
    tool_messages: list[ToolMessage]
    metadata: ChatDocMetaData
    attachment: None | ChatDocAttachment
    def get_json_tools(self) -> list[str]: ...
    def log_fields(self) -> ChatDocLoggerFields: ...
    def tsv_str(self) -> str: ...
    def pop_tool_ids(self) -> None: ...
    @staticmethod
    def from_LLMResponse(
        response: LLMResponse, displayed: bool = False
    ) -> ChatDocument: ...
    @staticmethod
    def from_str(msg: str) -> ChatDocument: ...
    @staticmethod
    def to_LLMMessage(message: str | ChatDocument) -> LLMMessage: ...
