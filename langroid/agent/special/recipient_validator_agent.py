"""
Agent to detect un-specified recipient and ask for clarification,
and when received, modify the pending message so that it looks as if
the parent task's LLM generated the right message in the first place.

Note that this is deprecated in favor of using the `RecipientTool`, defined in
`langroid.agent.tools.recipient_tool.py`.
See usage examples in `tests/main/test_multi_agent_complex.py` and
`tests/main/test_recipient_tool.py`.

The advantages of using the `RecipientTool` are:
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

import logging
from typing import List, Optional

from rich import print
from rich.console import Console

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import (
    ChatDocAttachment,
    ChatDocMetaData,
    ChatDocument,
)
from langroid.mytypes import Entity

console = Console()

logger = logging.getLogger(__name__)
# TODO - this is currently hardocded to validate the TO:<recipient> format
# but we could have a much more general declarative grammar-based validator


class RecipientValidatorConfig(ChatAgentConfig):
    recipients: List[str]
    tool_recipient: str | None = None
    name = "RecipientValidator"


class RecipientValidatorAttachment(ChatDocAttachment):
    content: str = ""


class RecipientValidator(ChatAgent):
    def __init__(self, config: RecipientValidatorConfig):
        super().__init__(config)
        self.config: RecipientValidatorConfig = config
        self.llm = None
        self.vecdb = None

    def user_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        # don't get user input
        return None

    def agent_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        """
        Check whether the incoming message is in the expected format.
        Used to check whether the output of the LLM of the calling agent is
        in the expected format.

        Args:
            msg (str|ChatDocument): the incoming message (pending message of the task)

        Returns:
            Optional[ChatDocument]:
            - if msg is in expected format, return None (no objections)
            - otherwise, a ChatDocument that either contains a request to
                LLM to clarify/fix the msg, or a fixed version of the LLM's original
                message.
        """
        if msg is None:
            return None
        if isinstance(msg, str):
            msg = ChatDocument.from_str(msg)

        recipient = msg.metadata.recipient
        has_func_call = msg.function_call is not None
        content = msg.content

        if recipient != "":
            # there is a clear recipient, return None (no objections)
            return None

        attachment: None | ChatDocAttachment = None
        responder: None | Entity = None
        sender_name = self.config.name
        if (
            has_func_call or "TOOL" in content
        ) and self.config.tool_recipient is not None:
            # assume it is meant for Coder, so simply set the recipient field,
            # and the parent task loop continues as normal
            # TODO- but what if it is not a legit function call
            recipient = self.config.tool_recipient
        elif content in self.config.recipients:
            # the incoming message is a clarification response from LLM
            recipient = content
            if msg.attachment is not None and isinstance(
                msg.attachment, RecipientValidatorAttachment
            ):
                content = msg.attachment.content
            else:
                logger.warning("ValidatorAgent: Did not find content to correct")
                content = ""
            # we've used the attachment, don't need anymore
            attachment = RecipientValidatorAttachment(content="")
            # we are rewriting an LLM message from parent, so
            # pretend it is from LLM
            responder = Entity.LLM
            sender_name = ""
        else:
            # save the original message so when the Validator
            # receives the LLM clarification,
            # it can use it as the `content` field
            attachment = RecipientValidatorAttachment(content=content)
            recipient_str = ", ".join(self.config.recipients)
            content = f"""
            Who is this message for? 
            Please simply respond with one of these names:
            {recipient_str}
            """
            console.print(f"[red]{self.indent}", end="")
            print(f"[red]Validator: {content}")

        return ChatDocument(
            content=content,
            function_call=msg.function_call if has_func_call else None,
            attachment=attachment,
            metadata=ChatDocMetaData(
                source=Entity.AGENT,
                sender=Entity.AGENT,
                parent_responder=responder,
                sender_name=sender_name,
                recipient=recipient,
            ),
        )
