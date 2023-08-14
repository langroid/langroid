"""
The `recipient_tool` is used to send a message to a specific recipient.
Various methods from the RecipientTool and AddRecipientTool class
are inserted into the Agent as methods (see `langroid/agent/base.py`,
the method `_get_tool_list()`).

See usage examples in `tests/main/test_multi_agent_complex.py` and
`tests/main/test_recipient_tool.py`.

Previously we were using RecipientValidatorAgent to enforce proper
recipient specifiction, but the preferred method is to use the
`RecipientTool` class.  This has numerous advantages:
- it uses the tool/function-call mechanism to specify a recipient in a JSON-structured
    string, which is more consistent with the rest of the system, and does not require
    inventing a new syntax like `TO:<recipient>` (which the RecipientValidatorAgent
    uses).
- it removes the need for any special parsing of the message content, since we leverage
    the built-in JSON tool-matching in `Agent.handle_message()` and downstream code.
- it does not require setting the `parent_responder` field in the `ChatDocument`
    metadata, which is somewhat hacky.
- it appears to be less brittle than requiring the LLM to use TO:<recipient> syntax:
  The LLM almost never forgets to use the RecipientTool as instructed.
- The RecipientTool class acts as a specification of the required syntax, and also
  contains mechanisms to enforce this syntax.
- For a developer who needs to enforce recipient specification for an agent, they only
  need to do `agent.enable_message(RecipientTool)`, and the rest is taken care of.
"""
from rich import print

from langroid.agent.chat_agent import ChatAgent
from langroid.agent.chat_document import (
    ChatDocAttachment,
    ChatDocMetaData,
    ChatDocument,
)
from langroid.agent.tool_message import ToolMessage
from langroid.mytypes import Entity


class RecipientValidatorAttachment(ChatDocAttachment):
    content: str = ""


class AddRecipientTool(ToolMessage):
    """
    Used by LLM to add a recipient to the previous message, when it has
    forgotten to specify a recipient. This avoids having to re-generate the
    previous message (and thus saves token-cost and time).
    """

    request: str = "add_recipient"
    purpose: str = (
        "To add a <recipient> when I forgot to specify it, "
        "to clarify who the message is intended for."
    )
    recipient: str
    attachment: None | RecipientValidatorAttachment = None

    class Config:
        # do not include these fields in the generated schema
        # since we don't require the LLM to specify them
        schema_extra = {"exclude": {"attachment", "purpose"}}

    def response(self, agent: ChatAgent) -> ChatDocument:
        """
        Returns:
            (ChatDocument): with content set to self.content and
                metadata.recipient set to self.recipient.
        """
        print(f"[red]RecipientTool: Added recipient {self.recipient} to message.")
        if self.__class__.attachment is None:
            raise ValueError("AddRecipientTool: attachment is None")
        return ChatDocument(
            content=self.__class__.attachment.content,  # use class-level attrib value
            metadata=ChatDocMetaData(
                recipient=self.recipient,
                # we are constructing this so it looks as it msg is from LLM
                sender=Entity.LLM,
            ),
        )


class RecipientTool(ToolMessage):
    """
    Used by LLM to send a message to a specific recipient.

    Useful in cases where an LLM is talking to 2 or more
    agents, and needs to specify which agent (task) its message is intended for.
    The recipient name should be the name of a task (which is normally the name of
    the agent that the task wraps, although the task can have its own name).

    To use this tool/function-call, LLM must generate a JSON structure
    with these fields:
    {
        "request": "recipient_tool", # this is the function name when using fn-calling
        "recipient": <name_of_recipient_task>,
        "content": <content>
    }
    """

    request: str = "recipient_message"
    purpose: str = "To address a message <content> to a specific <recipient>."
    recipient: str
    content: str

    def response(self, agent: ChatAgent) -> str | ChatDocument:
        """
        When LLM has correctly used this tool, set the agent's `recipient_tool_used`
        field to True, and construct a ChatDocument with an explicit recipient,
        and make it look like it is from the LLM.

        Returns:
            (ChatDocument): with content set to self.content and
                metadata.recipient set to self.recipient.
        """

        if self.recipient == "":
            # save the content in the attachment, so that
            # we can construct the ChatDocument once the LLM specifies a recipient.
            # This avoids having to re-generate the entire message, saving time + cost.
            AddRecipientTool.attachment = RecipientValidatorAttachment(
                content=self.content
            )
            agent.enable_message(AddRecipientTool)
            return ChatDocument(
                content="""
                Empty recipient field!
                Please use the 'add_recipient' tool/function-call to specify who your 
                message is intended for.
                DO NOT REPEAT your original message; ONLY specify the recipient via this
                tool/function-call.
                """,
                attachment=RecipientValidatorAttachment(content=self.content),
                metadata=ChatDocMetaData(
                    sender=Entity.AGENT,
                    recipient=Entity.LLM,
                ),
            )

        print("[red]RecipientTool: Validated properly addressed message")

        return ChatDocument(
            content=self.content,
            metadata=ChatDocMetaData(
                recipient=self.recipient,
                # we are constructing this so it looks as it msg is from LLM
                sender=Entity.LLM,
            ),
        )

    @staticmethod
    def handle_message_fallback(
        agent: ChatAgent, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """
        Response of agent if this tool is not used, e.g.
        the LLM simply sends a message without using this tool.
        This method has two purposes:
        (a) Alert the LLM that it has forgotten to specify a recipient, and prod it
            to use the `add_recipient` tool to specify just the recipient
            (and not re-generate the entire message).
        (b) Save the content of the message in the agent's `content` field,
            so the agent can construct a ChatDocument with this content once LLM
            later specifies a recipient using the `add_recipient` tool.

        This method is used to set the agent's handle_message_fallback() method.

        Returns:
            (str): reminder to LLM to use the `add_recipient` tool.
        """
        # Note: once the LLM specifies a missing recipient, the task loop
        # mechanism will not allow any of the "native" responders to respond,
        # since the recipient will differ from the task name.
        # So if this method is called, we can be sure that the recipient has not
        # been specified.
        if isinstance(msg, str):
            return None
        if msg.metadata.sender != Entity.LLM:
            return None
        content = msg if isinstance(msg, str) else msg.content
        # save the content in the attachment, so that
        # we can construct the ChatDocument once the LLM specifies a recipient.
        # This avoids having to re-generate the entire message, saving time + cost.
        AddRecipientTool.attachment = RecipientValidatorAttachment(content=content)
        agent.enable_message(AddRecipientTool)
        print("[red]RecipientTool: Recipient not specified, asking LLM to clarify.")
        return ChatDocument(
            content="""
            Please use the 'add_recipient' tool/function-call to specify who your 
            message is intended for.
            DO NOT REPEAT your original message; ONLY specify the recipient via this
            tool/function-call.
            """,
            attachment=RecipientValidatorAttachment(content=content),
            metadata=ChatDocMetaData(
                sender=Entity.AGENT,
                recipient=Entity.LLM,
            ),
        )
