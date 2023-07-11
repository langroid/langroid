import logging
from typing import Optional

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import (
    ChatDocAttachment,
    ChatDocMetaData,
    ChatDocument,
)
from langroid.mytypes import Entity
from langroid.parsing.agent_chats import parse_message

logger = logging.getLogger(__name__)
# TODO - this is currently hardocded to validate the TO:<recipient> format
# but we could have a much more general declarative grammar-based validator


class MessageValidatorAgent(ChatAgent):
    def __init__(self, config: ChatAgentConfig):
        super().__init__(config)
        self.config = config
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

        has_func_call = False
        if isinstance(msg, ChatDocument):
            recipient = msg.metadata.recipient
            has_func_call = msg.function_call is not None
            content = msg.content
        else:
            recipient, content = parse_message(msg)

        if recipient != "":
            # there is a clear recipient, return None (no objections)
            return None

        attachment: None | ChatDocAttachment = None
        responder: None | Entity = None
        sender_name = self.config.name
        if has_func_call or "TOOL" in content:
            # assume it is meant for Coder, so simply set the recipient field,
            # and the parent task loop continues as normal
            # TODO- but what if it is not a legit function call
            recipient = "Coder"
        elif content in ["DockerExpert", "Coder"]:
            # the incoming message is a clarification response from LLM
            recipient = content
            try:
                content = msg.attachment.content
                # we've used the attachment, don't need anymore
                attachment = ChatDocAttachment(content="")
            except Exception as e:
                content = msg.content
                logger.warning(f"MessageValidatorAgent: {str(e)}")
            # we are rewriting an LLM message from parent, so
            # pretend it is from LLM
            responder = Entity.LLM
            sender_name = ""
        else:
            # save the original message so when the Validator
            # receives the LLM clarification,
            # it can use it as the `content` field
            attachment = ChatDocAttachment(content=content)
            content = """
            Is this message for DockerExpert, or for Coder?
            Please simply respond with "DockerExpert" or "Coder"
            """
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
