"""
Various tools to for agents to be able to control flow of Task, e.g.
termination, routing to another agent, etc.
"""

from typing import List

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

    purpose = """
    To forward the current message to an <agent>.
    """
    request = "forward_tool"
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