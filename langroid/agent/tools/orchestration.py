"""
Various tools to for agents to be able to control flow of Task, e.g.
termination, routing to another agent, etc.
"""

from typing import List, Tuple

from langroid.agent.chat_agent import ChatAgent
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.mytypes import Entity


class AgentDoneTool(ToolMessage):
    """Tool for AGENT entity (i.e. agent_response or downstream tool handling fns) to
    signal the current task is done."""

    purpose: str = """
    To signal the current task is done, along with an optional message <content>
    (default empty string) and an optional list of <tools> (default empty list).
    """
    request: str = "agent_done_tool"
    content: str = ""
    tools: List[ToolMessage] = []
    _handle_only: bool = True

    def response(self, agent: ChatAgent) -> ChatDocument:
        return agent.create_agent_response(
            self.content,
            tool_messages=[self] + self.tools,
        )


class DoneTool(ToolMessage):
    """Tool for Agent Entity (i.e. agent_response) or LLM entity (i.e. llm_response) to
    signal the current task is done, with some content as the result."""

    purpose = """
    To signal the current task is done, along with an optional message <content>
    (default empty string).
    """
    request = "done_tool"
    content: str = ""

    def response(self, agent: ChatAgent) -> ChatDocument:
        return agent.create_agent_response(
            self.content,
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


class PassTool(ToolMessage):
    """Tool for "passing" on the received msg (ChatDocument),
    so that an as-yet-unspecified agent can handle it.
    Similar to ForwardTool, but without specifying the recipient agent.
    """

    purpose = """
    To pass the current message so that other agents can handle it.
    """
    request = "pass_tool"

    def response(self, agent: ChatAgent, chat_doc: ChatDocument) -> ChatDocument:
        """When this tool is enabled for an Agent, this will result in a method
        added to the Agent with signature:
        `forward_tool(self, tool: PassTool, chat_doc: ChatDocument) -> ChatDocument:`
        """
        # if PassTool is in chat_doc, pass its parent, else pass chat_doc itself
        tools = agent.get_tool_messages(chat_doc)
        doc = (
            chat_doc.parent
            if any(isinstance(t, type(self)) for t in tools)
            else chat_doc
        )
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

    purpose = """
    To signal the current task is done, with results set to the current/incoming msg.
    """
    request = "done_pass_tool"

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
    """Tool for forwarding the received msg (ChatDocument) to another agent.
    Similar to PassTool, but with a specified recipient agent.
    """

    purpose: str = """
    To forward the current message to an <agent>.
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
    _handle_only: bool = True

    def response(self, agent: ChatAgent) -> ChatDocument:
        return agent.create_agent_response(
            self.content,
            tool_messages=self.tools,
            recipient=self.to,
        )
