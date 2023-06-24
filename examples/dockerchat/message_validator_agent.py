from typing import Optional
from llmagent.agent.chat_agent import ChatAgent, ChatAgentConfig
from llmagent.agent.base import Entity
from llmagent.agent.chat_document import ChatDocument, ChatDocMetaData
from llmagent.parsing.agent_chats import parse_message

import logging

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
            msg (str): the incoming message (pending message of the task)
            sender_name (str): the name of the sender

        Returns:
            ChatDocument: None if message is in the expected format, otherwise
                a ChatDocument with an instruction on how to rewrite the message.
                (this is intended to be sent to the LLM of the calling agent).

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

        if has_func_call or "TOOL" in content:
            # assume it is meant for Coder
            # TODO- but what if it is not a legit function call
            recipient = "Coder"
        elif "DockerExpert" in content:
            content = msg.metadata.parent.metadata.parent.content
            recipient = "DockerExpert"
        elif "Coder" in content:
            content = msg.metadata.parent.content
            recipient = "Coder"
        else:
            # recipient = "DockerExpert"
            # logger.warning("TO[] not specified; assuming message is for DockerExpert")
            # we don't know who it is for, return a message asking for clarification
            content = """
            Is this message for DockerExpert, or for Coder?
            Please simply respond with "DockerExpert" or "Coder"
            """
        return ChatDocument(
            content=content,
            function_call=msg.function_call if has_func_call else None,
            metadata=ChatDocMetaData(
                source=Entity.AGENT,
                sender=Entity.AGENT,
                sender_name=self.config.name,
                recipient=recipient,
            ),
        )
