"""
The `recipient_tool` is used to send a message to a specific recipient.
Various methods from the RecipientTool and AddRecipientTool class
are inserted into the Agent as methods (see `langroid/agent/base.py`,
the method `_get_tool_list()`).

See usage examples in `tests/main/test_multi_agent_complex.py` and
`tests/main/test_recipient_tool.py`.

A simpler alternative to this tool is `SendTool`, see here:
https://github.com/langroid/langroid/blob/main/langroid/agent/tools/orchestration.py

You can also define your own XML-based variant of this tool:
https://github.com/langroid/langroid/blob/main/examples/basic/xml-tool.py
which uses XML rather than JSON, and can be more reliable than JSON,
especially with weaker LLMs.

"""

from typing import List, Type

from rich import print

from langroid.agent.chat_agent import ChatAgent
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.mytypes import Entity
from langroid.utils.pydantic_utils import has_field


class AddRecipientTool(ToolMessage):
    """
    Used by LLM to add a recipient to the previous message, when it has
    forgotten to specify a recipient. This avoids having to re-generate the
    previous message (and thus saves token-cost and time).
    """

    request: str = "add_recipient"
    purpose: str = (
        "To clarify that the <intended_recipient> when I forgot to specify it, "
        "to clarify who the message is intended for."
    )
    intended_recipient: str
    _saved_content: str = ""

    def response(self, agent: ChatAgent) -> ChatDocument:
        """
        Returns:
            (ChatDocument): with content set to self.content and
                metadata.recipient set to self.recipient.
        """
        print(
            "[red]RecipientTool: "
            f"Added recipient {self.intended_recipient} to message."
        )
        if self.__class__._saved_content == "":
            recipient_request_name = RecipientTool.default_value("request")
            content = f"""
                Recipient specified but content is empty!
                This could be because the `{self.request}` tool/function was used 
                before using `{recipient_request_name}` tool/function.
                Resend the message using `{recipient_request_name}` tool/function.
                """
        else:
            content = self.__class__._saved_content  # use class-level attrib value
            # erase content since we just used it.
            self.__class__._saved_content = ""
        return ChatDocument(
            content=content,
            metadata=ChatDocMetaData(
                recipient=self.intended_recipient,
                # we are constructing this so it looks as it msg is from LLM
                sender=Entity.LLM,
            ),
        )


class RecipientTool(ToolMessage):
    """
    Used by LLM to send a message to a specific recipient.

    Useful in cases where an LLM is talking to 2 or more
    agents (or an Agent and human user), and needs to specify which agent (task)
    its message is intended for. The recipient name should be the name of a task
    (which is normally the name of the agent that the task wraps, although the task
    can have its own name).

    To use this tool/function-call, LLM must generate a JSON structure
    with these fields:
    {
        "request": "recipient_message", # also the function name when using fn-calling
        "intended_recipient": <name_of_recipient_task_or_entity>,
        "content": <content>
    }
    The effect of this is that `content` will be sent to the `intended_recipient` task.
    """

    request: str = "recipient_message"
    purpose: str = "To send message <content> to a specific <intended_recipient>."
    intended_recipient: str
    content: str

    @classmethod
    def create(cls, recipients: List[str], default: str = "") -> Type["RecipientTool"]:
        """Create a restricted version of RecipientTool that
        only allows certain recipients, and possibly sets a default recipient."""

        class RecipientToolRestricted(cls):  # type: ignore
            allowed_recipients = recipients
            default_recipient = default

        return RecipientToolRestricted

    @classmethod
    def instructions(cls) -> str:
        """
        Generate instructions for using this tool/function.
        These are intended to be appended to the system message of the LLM.
        """
        recipients = []
        if has_field(cls, "allowed_recipients"):
            recipients = cls.default_value("allowed_recipients")
        if len(recipients) > 0:
            recipients_str = ", ".join(recipients)
            return f"""
            Since you will be talking to multiple recipients, 
            you must clarify who your intended recipient is, using 
            the `{cls.default_value("request")}` tool/function-call, by setting the 
            'intended_recipient' field to one of the following:
            {recipients_str},
            and setting the 'content' field to your message.
            """
        else:
            return f"""
            Since you will be talking to multiple recipients, 
            you must clarify who your intended recipient is, using 
            the `{cls.default_value("request")}` tool/function-call, by setting the 
            'intended_recipient' field to the name of the recipient, 
            and setting the 'content' field to your message.
            """

    def response(self, agent: ChatAgent) -> str | ChatDocument:
        """
        When LLM has correctly used this tool,
        construct a ChatDocument with an explicit recipient,
        and make it look like it is from the LLM.

        Returns:
            (ChatDocument): with content set to self.content and
                metadata.recipient set to self.intended_recipient.
        """
        default_recipient = self.__class__.default_value("default_recipient")
        if self.intended_recipient == "" and default_recipient not in ["", None]:
            self.intended_recipient = default_recipient
        elif self.intended_recipient == "":
            # save the content as a class-variable, so that
            # we can construct the ChatDocument once the LLM specifies a recipient.
            # This avoids having to re-generate the entire message, saving time + cost.
            AddRecipientTool._saved_content = self.content
            agent.enable_message(AddRecipientTool)
            return ChatDocument(
                content="""
                Empty recipient field!
                Please use the 'add_recipient' tool/function-call to specify who your 
                message is intended for.
                DO NOT REPEAT your original message; ONLY specify the recipient via this
                tool/function-call.
                """,
                metadata=ChatDocMetaData(
                    sender=Entity.AGENT,
                    recipient=Entity.LLM,
                ),
            )

        print("[red]RecipientTool: Validated properly addressed message")

        return ChatDocument(
            content=self.content,
            metadata=ChatDocMetaData(
                recipient=self.intended_recipient,
                # we are constructing this so it looks as if msg is from LLM
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
        if (
            isinstance(msg, str)
            or msg.metadata.sender != Entity.LLM
            or msg.metadata.recipient != ""  # there IS an explicit recipient
        ):
            return None
        content = msg if isinstance(msg, str) else msg.content
        # save the content as a class-variable, so that
        # we can construct the ChatDocument once the LLM specifies a recipient.
        # This avoids having to re-generate the entire message, saving time + cost.
        AddRecipientTool._saved_content = content
        agent.enable_message(AddRecipientTool)
        print("[red]RecipientTool: Recipient not specified, asking LLM to clarify.")
        return ChatDocument(
            content="""
            Please use the 'add_recipient' tool/function-call to specify who your 
            `intended_recipient` is.
            DO NOT REPEAT your original message; ONLY specify the 
            `intended_recipient` via this tool/function-call.
            """,
            metadata=ChatDocMetaData(
                sender=Entity.AGENT,
                recipient=Entity.LLM,
            ),
        )
