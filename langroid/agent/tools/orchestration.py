"""
Various tools to for agents to be able to control flow of Task, e.g.
termination, routing to another agent, etc.
"""

from typing import Any, List, Tuple

from langroid.agent.chat_agent import ChatAgent
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.mytypes import Entity
from langroid.pydantic_v1 import Extra
from langroid.utils.types import to_string


class AgentDoneTool(ToolMessage):
    """Tool for AGENT entity (i.e. agent_response or downstream tool handling fns) to
    signal the current task is done."""

    purpose: str = """
    To signal the current task is done, along with an optional message <content>
    of arbitrary type (default None) and an 
    optional list of <tools> (default empty list).
    """
    request: str = "agent_done_tool"
    content: Any = None
    tools: List[ToolMessage] = []
    # only meant for agent_response or tool-handlers, not for LLM generation:
    _allow_llm_use: bool = False

    def response(self, agent: ChatAgent) -> ChatDocument:
        content_str = "" if self.content is None else to_string(self.content)
        return agent.create_agent_response(
            content=content_str,
            content_any=self.content,
            tool_messages=[self] + self.tools,
        )


class DoneTool(ToolMessage):
    """Tool for Agent Entity (i.e. agent_response) or LLM entity (i.e. llm_response) to
    signal the current task is done, with some content as the result."""

    purpose: str = """
    To signal the current task is done, along with an optional message <content>
    of arbitrary type (default None).
    """
    request: str = "done_tool"
    content: str = ""

    def response(self, agent: ChatAgent) -> ChatDocument:
        return agent.create_agent_response(
            content=self.content,
            content_any=self.content,
            tool_messages=[self],
        )

    @classmethod
    def instructions(cls) -> str:
        tool_name = cls.default_value("request")
        return f"""
        When you determine your task is finished, 
        use the tool `{tool_name}` to signal this,
        along with any message or result, in the `content` field. 
        """


class ResultTool(ToolMessage):
    """Class to use as a wrapper for sending arbitrary results from an Agent's
    agent_response or tool handlers, to:
    (a) trigger completion of the current task (similar to (Agent)DoneTool), and
    (b) be returned as the result of the current task, i.e. this tool would appear
         in the resulting ChatDocument's `tool_messages` list.
    See test_tool_handlers_and_results in test_tool_messages.py, and
    examples/basic/tool-extract-short-example.py.

    Note:
        - when defining a tool handler or agent_response, you can directly return
            ResultTool(field1 = val1, ...),
            where the values can be arbitrary data structures, including nested
            Pydantic objs, or you can define a subclass of ResultTool with the
            fields you want to return.
        - This is a special ToolMessage that is NOT meant to be used or handled
            by an agent.
        - AgentDoneTool is more restrictive in that you can only send a `content`
            or `tools` in the result.
    """

    request: str = "result_tool"
    purpose: str = "Ignored; Wrapper for a structured message"
    id: str = ""  # placeholder for OpenAI-API tool_call_id

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = False
        validate_all = True
        validate_assignment = True
        # do not include these fields in the generated schema
        # since we don't require the LLM to specify them
        schema_extra = {"exclude": {"purpose", "id"}}

    def handle(self) -> AgentDoneTool:
        return AgentDoneTool(tools=[self])


class FinalResultTool(ToolMessage):
    """Class to use as a wrapper for sending arbitrary results from an Agent's
    agent_response or tool handlers, to:
    (a) trigger completion of the current task as well as all parent tasks, and
    (b) be returned as the final result of the root task, i.e. this tool would appear
         in the final ChatDocument's `tool_messages` list.
    See test_tool_handlers_and_results in test_tool_messages.py, and
    examples/basic/tool-extract-short-example.py.

    Note:
        - when defining a tool handler or agent_response, you can directly return
            FinalResultTool(field1 = val1, ...),
            where the values can be arbitrary data structures, including nested
            Pydantic objs, or you can define a subclass of FinalResultTool with the
            fields you want to return.
        - This is a special ToolMessage that is NOT meant to be used or handled
            by an agent.
    """

    request: str = ""
    purpose: str = "Ignored; Wrapper for a structured message"
    id: str = ""  # placeholder for OpenAI-API tool_call_id
    _allow_llm_use: bool = False

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = False
        validate_all = True
        validate_assignment = True
        # do not include these fields in the generated schema
        # since we don't require the LLM to specify them
        schema_extra = {"exclude": {"purpose", "id"}}


class PassTool(ToolMessage):
    """Tool for "passing" on the received msg (ChatDocument),
    so that an as-yet-unspecified agent can handle it.
    Similar to ForwardTool, but without specifying the recipient agent.
    """

    purpose: str = """
    To pass the current message so that other agents can handle it.
    """
    request: str = "pass_tool"

    def response(self, agent: ChatAgent, chat_doc: ChatDocument) -> ChatDocument:
        """When this tool is enabled for an Agent, this will result in a method
        added to the Agent with signature:
        `forward_tool(self, tool: PassTool, chat_doc: ChatDocument) -> ChatDocument:`
        """
        # if PassTool is in chat_doc, pass its parent, else pass chat_doc itself
        doc = chat_doc
        while True:
            tools = agent.get_tool_messages(doc)
            if not any(isinstance(t, type(self)) for t in tools):
                break
            if doc.parent is None:
                break
            doc = doc.parent
        assert doc is not None, "PassTool: parent of chat_doc must not be None"
        new_doc = ChatDocument.deepcopy(doc)
        new_doc.metadata.sender = Entity.AGENT
        return new_doc

    @classmethod
    def instructions(cls) -> str:
        return """
        Use the `pass_tool` to PASS the current message 
        so that another agent can handle it.
        """


class DonePassTool(PassTool):
    """Tool to signal DONE, AND Pass incoming/current msg as result.
    Similar to PassTool, except we append a DoneTool to the result tool_messages.
    """

    purpose: str = """
    To signal the current task is done, with results set to the current/incoming msg.
    """
    request: str = "done_pass_tool"

    def response(self, agent: ChatAgent, chat_doc: ChatDocument) -> ChatDocument:
        # use PassTool to get the right ChatDocument to pass...
        new_doc = PassTool.response(self, agent, chat_doc)
        tools = agent.get_tool_messages(new_doc)
        # ...then return an AgentDoneTool with content, tools from this ChatDocument
        return AgentDoneTool(content=new_doc.content, tools=tools)  # type: ignore

    @classmethod
    def instructions(cls) -> str:
        return """
        When you determine your task is finished,
        and want to pass the current message as the result of the task,  
        use the `done_pass_tool` to signal this.
        """


class ForwardTool(PassTool):
    """Tool for forwarding the received msg (ChatDocument) to another agent or entity.
    Similar to PassTool, but with a specified recipient agent.
    """

    purpose: str = """
    To forward the current message to an <agent>, where <agent> 
    could be the name of an agent, or an entity such as "user", "llm".
    """
    request: str = "forward_tool"
    agent: str

    def response(self, agent: ChatAgent, chat_doc: ChatDocument) -> ChatDocument:
        """When this tool is enabled for an Agent, this will result in a method
        added to the Agent with signature:
        `forward_tool(self, tool: ForwardTool, chat_doc: ChatDocument) -> ChatDocument:`
        """
        # if chat_doc contains ForwardTool, then we forward its parent ChatDocument;
        # else forward chat_doc itself
        new_doc = PassTool.response(self, agent, chat_doc)
        new_doc.metadata.recipient = self.agent
        return new_doc

    @classmethod
    def instructions(cls) -> str:
        return """
        If you need to forward the current message to another agent, 
        use the `forward_tool` to do so, 
        setting the `recipient` field to the name of the recipient agent.
        """


class SendTool(ToolMessage):
    """Tool for agent or LLM to send content to a specified agent.
    Similar to RecipientTool.
    """

    purpose: str = """
    To send message <content> to agent specified in <to> field.
    """
    request: str = "send_tool"
    to: str
    content: str = ""

    def response(self, agent: ChatAgent) -> ChatDocument:
        return agent.create_agent_response(
            self.content,
            recipient=self.to,
        )

    @classmethod
    def instructions(cls) -> str:
        return """
        If you need to send a message to another agent, 
        use the `send_tool` to do so, with these field values:
        - `to` field = name of the recipient agent,
        - `content` field = the message to send.
        """

    @classmethod
    def examples(cls) -> List["ToolMessage" | Tuple[str, "ToolMessage"]]:
        return [
            cls(to="agent1", content="Hello, agent1!"),
            (
                """
                I need to send the content 'Who built the Gemini model?', 
                to the 'Searcher' agent.
                """,
                cls(to="Searcher", content="Who built the Gemini model?"),
            ),
        ]


class AgentSendTool(ToolMessage):
    """Tool for Agent (i.e. agent_response) to send content or tool_messages
    to a specified agent. Similar to SendTool except that AgentSendTool is only
    usable by agent_response (or handler of another tool), to send content or
    tools to another agent. SendTool does not allow sending tools.
    """

    purpose: str = """
    To send message <content> and <tools> to agent specified in <to> field. 
    """
    request: str = "agent_send_tool"
    to: str
    content: str = ""
    tools: List[ToolMessage] = []
    _allow_llm_use: bool = False

    def response(self, agent: ChatAgent) -> ChatDocument:
        return agent.create_agent_response(
            self.content,
            tool_messages=self.tools,
            recipient=self.to,
        )
