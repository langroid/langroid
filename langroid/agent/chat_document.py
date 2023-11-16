import json
from typing import List, Optional, Union

from pydantic import BaseModel, Extra

from langroid.language_models.base import (
    LLMFunctionCall,
    LLMMessage,
    LLMResponse,
    LLMTokenUsage,
    Role,
)
from langroid.mytypes import DocMetaData, Document, Entity
from langroid.parsing.agent_chats import parse_message
from langroid.parsing.json import extract_top_level_json, top_level_json_field
from langroid.utils.output.printing import shorten_text


class ChatDocAttachment(BaseModel):
    # any additional data that should be attached to the document
    class Config:
        extra = Extra.allow


class ChatDocMetaData(DocMetaData):
    parent: Optional["ChatDocument"] = None
    sender: Entity
    tool_ids: List[str] = []  # stack of tool_ids; used by OpenAIAssistant
    # when result returns to parent, pretend message is from this entity
    parent_responder: None | Entity = None
    block: None | Entity = None
    sender_name: str = ""
    recipient: str = ""
    usage: Optional[LLMTokenUsage]
    cached: bool = False
    displayed: bool = False


class ChatDocLoggerFields(BaseModel):
    sender_entity: Entity = Entity.USER
    sender_name: str = ""
    recipient: str = ""
    block: Entity | None = None
    tool_type: str = ""
    tool: str = ""
    content: str = ""

    @classmethod
    def tsv_header(cls) -> str:
        field_names = cls().dict().keys()
        return "\t".join(field_names)


class ChatDocument(Document):
    function_call: Optional[LLMFunctionCall] = None
    metadata: ChatDocMetaData
    attachment: None | ChatDocAttachment = None

    def __str__(self) -> str:
        fields = self.log_fields()
        tool_str = ""
        if fields.tool_type != "":
            tool_str = f"{fields.tool_type}[{fields.tool}]: "
        recipient_str = ""
        if fields.recipient != "":
            recipient_str = f"=>{fields.recipient}: "
        return (
            f"{fields.sender_entity}[{fields.sender_name}] "
            f"{recipient_str}{tool_str}{fields.content}"
        )

    def get_json_tools(self) -> List[str]:
        """
        Get names of attempted JSON tool usages in the content
            of the message.
        Returns:
            List[str]: list of JSON tool names
        """
        jsons = extract_top_level_json(self.content)
        tools = []
        for j in jsons:
            json_data = json.loads(j)
            tool = json_data.get("request")
            if tool is not None:
                tools.append(tool)
        return tools

    def log_fields(self) -> ChatDocLoggerFields:
        """
        Fields for logging in csv/tsv logger
        Returns:
            List[str]: list of fields
        """
        tool_type = ""  # FUNC or TOOL
        tool = ""  # tool name or function name
        if self.function_call is not None:
            tool_type = "FUNC"
            tool = self.function_call.name
        elif self.get_json_tools() != []:
            tool_type = "TOOL"
            tool = self.get_json_tools()[0]
        recipient = self.metadata.recipient
        content = self.content
        sender_entity = self.metadata.sender
        sender_name = self.metadata.sender_name
        return ChatDocLoggerFields(
            sender_entity=sender_entity,
            sender_name=sender_name,
            recipient=recipient,
            block=self.metadata.block,
            tool_type=tool_type,
            tool=tool,
            content=content,
        )

    def tsv_str(self) -> str:
        fields = self.log_fields()
        fields.content = shorten_text(fields.content, 80)
        field_values = fields.dict().values()
        return "\t".join(str(v) for v in field_values)

    def pop_tool_ids(self) -> None:
        """
        Pop the last tool_id from the stack of tool_ids.
        """
        if len(self.metadata.tool_ids) > 0:
            self.metadata.tool_ids.pop()

    @staticmethod
    def from_LLMResponse(
        response: LLMResponse,
        displayed: bool = False,
    ) -> "ChatDocument":
        """
        Convert LLMResponse to ChatDocument.
        Args:
            response (LLMResponse): LLMResponse to convert.
            displayed (bool): Whether this response was displayed to the user.
        Returns:
            ChatDocument: ChatDocument representation of this LLMResponse.
        """
        recipient, message = response.get_recipient_and_message()
        return ChatDocument(
            content=message,
            function_call=response.function_call,
            metadata=ChatDocMetaData(
                source=Entity.LLM,
                sender=Entity.LLM,
                usage=response.usage,
                displayed=displayed,
                cached=response.cached,
                recipient=recipient,
            ),
        )

    @staticmethod
    def from_str(msg: str) -> "ChatDocument":
        # first check whether msg is structured as TO <recipient>: <message>
        recipient, message = parse_message(msg)
        if recipient == "":
            # check if any top level json specifies a 'recipient'
            recipient = top_level_json_field(msg, "recipient")
            message = msg  # retain the whole msg in this case
        return ChatDocument(
            content=message,
            metadata=ChatDocMetaData(
                source=Entity.USER,
                sender=Entity.USER,
                recipient=recipient,
            ),
        )

    @staticmethod
    def to_LLMMessage(message: Union[str, "ChatDocument"]) -> LLMMessage:
        """
        Convert to LLMMessage for use with LLM.

        Args:
            message (str|ChatDocument): Message to convert.
        Returns:
            LLMMessage: LLMMessage representation of this str or ChatDocument.

        """
        sender_name = None
        sender_role = Role.USER
        fun_call = None
        tool_id = ""
        if isinstance(message, ChatDocument):
            content = message.content
            fun_call = message.function_call
            sender_name = message.metadata.sender_name
            tool_ids = message.metadata.tool_ids
            tool_id = tool_ids[-1] if len(tool_ids) > 0 else ""
            if message.metadata.sender == Entity.SYSTEM:
                sender_role = Role.SYSTEM
            if (
                message.metadata.parent is not None
                and message.metadata.parent.function_call is not None
            ):
                sender_role = Role.FUNCTION
                sender_name = message.metadata.parent.function_call.name
            elif message.metadata.sender == Entity.LLM:
                sender_role = Role.ASSISTANT
        else:
            # LLM can only respond to text content, so extract it
            content = message

        return LLMMessage(
            role=sender_role,
            tool_id=tool_id,
            content=content,
            function_call=fun_call,
            name=sender_name,
        )


ChatDocMetaData.update_forward_refs()
