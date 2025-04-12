from __future__ import annotations

import copy
import json
from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast

from langroid.agent.tool_message import ToolMessage
from langroid.agent.xml_tool_message import XMLToolMessage
from langroid.language_models.base import (
    LLMFunctionCall,
    LLMMessage,
    LLMResponse,
    LLMTokenUsage,
    OpenAIToolCall,
    Role,
    ToolChoiceTypes,
)
from langroid.mytypes import DocMetaData, Document, Entity
from langroid.parsing.agent_chats import parse_message
from langroid.parsing.parse_json import extract_top_level_json, top_level_json_field
from langroid.pydantic_v1 import BaseModel, Extra
from langroid.utils.object_registry import ObjectRegistry
from langroid.utils.output.printing import shorten_text
from langroid.utils.types import to_string


class ChatDocAttachment(BaseModel):
    # any additional data that should be attached to the document
    class Config:
        extra = Extra.allow


class StatusCode(str, Enum):
    """Codes meant to be returned by task.run(). Some are not used yet."""

    OK = "OK"
    ERROR = "ERROR"
    DONE = "DONE"
    STALLED = "STALLED"
    INF_LOOP = "INF_LOOP"
    KILL = "KILL"
    FIXED_TURNS = "FIXED_TURNS"  # reached intended number of turns
    MAX_TURNS = "MAX_TURNS"  # hit max-turns limit
    MAX_COST = "MAX_COST"
    MAX_TOKENS = "MAX_TOKENS"
    TIMEOUT = "TIMEOUT"
    NO_ANSWER = "NO_ANSWER"
    USER_QUIT = "USER_QUIT"


class ChatDocMetaData(DocMetaData):
    parent_id: str = ""  # msg (ChatDocument) to which this is a response
    child_id: str = ""  # ChatDocument that has response to this message
    agent_id: str = ""  # ChatAgent that generated this message
    msg_idx: int = -1  # index of this message in the agent `message_history`
    sender: Entity  # sender of the message
    # tool_id corresponding to single tool result in ChatDocument.content
    oai_tool_id: str | None = None
    tool_ids: List[str] = []  # stack of tool_ids; used by OpenAIAssistant
    block: None | Entity = None
    sender_name: str = ""
    recipient: str = ""
    usage: Optional[LLMTokenUsage] = None
    cached: bool = False
    displayed: bool = False
    has_citation: bool = False
    status: Optional[StatusCode] = None

    @property
    def parent(self) -> Optional["ChatDocument"]:
        return ChatDocument.from_id(self.parent_id)

    @property
    def child(self) -> Optional["ChatDocument"]:
        return ChatDocument.from_id(self.child_id)


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
    """
    Represents a message in a conversation among agents. All responders of an agent
    have signature ChatDocument -> ChatDocument (modulo None, str, etc),
    and so does the Task.run() method.

    Attributes:
        oai_tool_calls (Optional[List[OpenAIToolCall]]):
            Tool-calls from an OpenAI-compatible API
        oai_tool_id2results (Optional[OrderedDict[str, str]]):
            Results of tool-calls from OpenAI (dict is a map of tool_id -> result)
        oai_tool_choice: ToolChoiceTypes | Dict[str, str]: Param controlling how the
            LLM should choose tool-use in its response
            (auto, none, required, or a specific tool)
        function_call (Optional[LLMFunctionCall]):
            Function-call from an OpenAI-compatible API
                (deprecated by OpenAI, in favor of tool-calls)
        tool_messages (List[ToolMessage]): Langroid ToolMessages extracted from
            - `content` field (via JSON parsing),
            - `oai_tool_calls`, or
            - `function_call`
        metadata (ChatDocMetaData): Metadata for the message, e.g. sender, recipient.
        attachment (None | ChatDocAttachment): Any additional data attached.
    """

    reasoning: str = ""  # reasoning produced by a reasoning LLM
    content_any: Any = None  # to hold arbitrary data returned by responders
    oai_tool_calls: Optional[List[OpenAIToolCall]] = None
    oai_tool_id2result: Optional[OrderedDict[str, str]] = None
    oai_tool_choice: ToolChoiceTypes | Dict[str, Dict[str, str] | str] = "auto"
    function_call: Optional[LLMFunctionCall] = None
    # tools that are explicitly added by agent response/handler,
    # or tools recognized in the ChatDocument as handle-able tools
    tool_messages: List[ToolMessage] = []
    # all known tools in the msg that are in an agent's llm_tools_known list,
    # even if non-used/handled
    all_tool_messages: List[ToolMessage] = []

    metadata: ChatDocMetaData
    attachment: None | ChatDocAttachment = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        ObjectRegistry.register_object(self)

    @property
    def parent(self) -> Optional["ChatDocument"]:
        return ChatDocument.from_id(self.metadata.parent_id)

    @property
    def child(self) -> Optional["ChatDocument"]:
        return ChatDocument.from_id(self.metadata.child_id)

    @staticmethod
    def deepcopy(doc: ChatDocument) -> ChatDocument:
        new_doc = copy.deepcopy(doc)
        new_doc.metadata.id = ObjectRegistry.new_id()
        new_doc.metadata.child_id = ""
        new_doc.metadata.parent_id = ""
        ObjectRegistry.register_object(new_doc)
        return new_doc

    @staticmethod
    def from_id(id: str) -> Optional["ChatDocument"]:
        return cast(ChatDocument, ObjectRegistry.get(id))

    @staticmethod
    def delete_id(id: str) -> None:
        """Remove ChatDocument with given id from ObjectRegistry,
        and all its descendants.
        """
        chat_doc = ChatDocument.from_id(id)
        # first delete all descendants
        while chat_doc is not None:
            next_chat_doc = chat_doc.child
            ObjectRegistry.remove(chat_doc.id())
            chat_doc = next_chat_doc

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

    def get_tool_names(self) -> List[str]:
        """
        Get names of attempted tool usages (JSON or non-JSON) in the content
            of the message.
        Returns:
            List[str]: list of *attempted* tool names
            (We say "attempted" since we ONLY look at the `request` component of the
            tool-call representation, and we're not fully parsing it into the
            corresponding tool message class)

        """
        tool_candidates = XMLToolMessage.find_candidates(self.content)
        if len(tool_candidates) == 0:
            tool_candidates = extract_top_level_json(self.content)
            if len(tool_candidates) == 0:
                return []
            tools = [json.loads(tc).get("request") for tc in tool_candidates]
        else:
            tool_dicts = [
                XMLToolMessage.extract_field_values(tc) for tc in tool_candidates
            ]
            tools = [td.get("request") for td in tool_dicts if td is not None]
        return [str(tool) for tool in tools if tool is not None]

    def log_fields(self) -> ChatDocLoggerFields:
        """
        Fields for logging in csv/tsv logger
        Returns:
            List[str]: list of fields
        """
        tool_type = ""  # FUNC or TOOL
        tool = ""  # tool name or function name
        oai_tools = (
            []
            if self.oai_tool_calls is None
            else [t for t in self.oai_tool_calls if t.function is not None]
        )
        if self.function_call is not None:
            tool_type = "FUNC"
            tool = self.function_call.name
        elif len(oai_tools) > 0:
            tool_type = "OAI_TOOL"
            tool = ",".join(t.function.name for t in oai_tools)  # type: ignore
        else:
            try:
                json_tools = self.get_tool_names()
            except Exception:
                json_tools = []
            if json_tools != []:
                tool_type = "TOOL"
                tool = json_tools[0]
        recipient = self.metadata.recipient
        content = self.content
        sender_entity = self.metadata.sender
        sender_name = self.metadata.sender_name
        if tool_type == "FUNC":
            content += str(self.function_call)
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
    def _clean_fn_call(fc: LLMFunctionCall | None) -> None:
        # Sometimes an OpenAI LLM (esp gpt-4o) may generate a function-call
        # with odditities:
        # (a) the `name` is set, as well as `arguments.request` is set,
        #  and in langroid we use the `request` value as the `name`.
        #  In this case we override the `name` with the `request` value.
        # (b) the `name` looks like "functions blah" or just "functions"
        #   In this case we strip the "functions" part.
        if fc is None:
            return
        fc.name = fc.name.replace("functions", "").strip()
        if fc.arguments is not None:
            request = fc.arguments.get("request")
            if request is not None and request != "":
                fc.name = request
                fc.arguments.pop("request")

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
        message = message.strip()
        if message in ["''", '""']:
            message = ""
        if response.function_call is not None:
            ChatDocument._clean_fn_call(response.function_call)
        if response.oai_tool_calls is not None:
            # there must be at least one if it's not None
            for oai_tc in response.oai_tool_calls:
                ChatDocument._clean_fn_call(oai_tc.function)
        return ChatDocument(
            content=message,
            reasoning=response.reasoning,
            content_any=message,
            oai_tool_calls=response.oai_tool_calls,
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
            content_any=message,
            metadata=ChatDocMetaData(
                source=Entity.USER,
                sender=Entity.USER,
                recipient=recipient,
            ),
        )

    @staticmethod
    def to_LLMMessage(
        message: Union[str, "ChatDocument"],
        oai_tools: Optional[List[OpenAIToolCall]] = None,
    ) -> List[LLMMessage]:
        """
        Convert to list of LLMMessage, to incorporate into msg-history sent to LLM API.
        Usually there will be just a single LLMMessage, but when the ChatDocument
        contains results from multiple OpenAI tool-calls, we would have a sequence
        LLMMessages, one per tool-call result.

        Args:
            message (str|ChatDocument): Message to convert.
            oai_tools (Optional[List[OpenAIToolCall]]): Tool-calls currently awaiting
                response, from the ChatAgent's latest message.
        Returns:
            List[LLMMessage]: list of LLMMessages corresponding to this ChatDocument.
        """
        sender_name = None
        sender_role = Role.USER
        fun_call = None
        oai_tool_calls = None
        tool_id = ""  # for OpenAI Assistant
        chat_document_id: str = ""
        if isinstance(message, str):
            message = ChatDocument.from_str(message)
        content = message.content or to_string(message.content_any) or ""
        fun_call = message.function_call
        oai_tool_calls = message.oai_tool_calls
        if message.metadata.sender == Entity.USER and fun_call is not None:
            # This may happen when a (parent agent's) LLM generates a
            # a Function-call, and it ends up being sent to the current task's
            # LLM (possibly because the function-call is mis-named or has other
            # issues and couldn't be handled by handler methods).
            # But a function-call can only be generated by an entity with
            # Role.ASSISTANT, so we instead put the content of the function-call
            # in the content of the message.
            content += " " + str(fun_call)
            fun_call = None
        if message.metadata.sender == Entity.USER and oai_tool_calls is not None:
            # same reasoning as for function-call above
            content += " " + "\n\n".join(str(tc) for tc in oai_tool_calls)
            oai_tool_calls = None
        sender_name = message.metadata.sender_name
        tool_ids = message.metadata.tool_ids
        tool_id = tool_ids[-1] if len(tool_ids) > 0 else ""
        chat_document_id = message.id()
        if message.metadata.sender == Entity.SYSTEM:
            sender_role = Role.SYSTEM
        if (
            message.metadata.parent is not None
            and message.metadata.parent.function_call is not None
        ):
            # This is a response to a function call, so set the role to FUNCTION.
            sender_role = Role.FUNCTION
            sender_name = message.metadata.parent.function_call.name
        elif oai_tools is not None and len(oai_tools) > 0:
            pending_tool_ids = [tc.id for tc in oai_tools]
            # The ChatAgent has pending OpenAI tool-call(s),
            # so the current ChatDocument contains
            # results for some/all/none of them.

            if len(oai_tools) == 1:
                # Case 1:
                # There was exactly 1 pending tool-call, and in this case
                # the result would be a plain string in `content`
                return [
                    LLMMessage(
                        role=Role.TOOL,
                        tool_call_id=oai_tools[0].id,
                        content=content,
                        chat_document_id=chat_document_id,
                    )
                ]

            elif (
                message.metadata.oai_tool_id is not None
                and message.metadata.oai_tool_id in pending_tool_ids
            ):
                # Case 2:
                # ChatDocument.content has result of a single tool-call
                return [
                    LLMMessage(
                        role=Role.TOOL,
                        tool_call_id=message.metadata.oai_tool_id,
                        content=content,
                        chat_document_id=chat_document_id,
                    )
                ]
            elif message.oai_tool_id2result is not None:
                # Case 2:
                # There were > 1 tool-calls awaiting response,
                assert (
                    len(message.oai_tool_id2result) > 1
                ), "oai_tool_id2result must have more than 1 item."
                return [
                    LLMMessage(
                        role=Role.TOOL,
                        tool_call_id=tool_id,
                        content=result,
                        chat_document_id=chat_document_id,
                    )
                    for tool_id, result in message.oai_tool_id2result.items()
                ]
        elif message.metadata.sender == Entity.LLM:
            sender_role = Role.ASSISTANT

        return [
            LLMMessage(
                role=sender_role,
                tool_id=tool_id,  # for OpenAI Assistant
                content=content,
                function_call=fun_call,
                tool_calls=oai_tool_calls,
                name=sender_name,
                chat_document_id=chat_document_id,
            )
        ]


LLMMessage.update_forward_refs()
ChatDocMetaData.update_forward_refs()
