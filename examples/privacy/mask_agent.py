"""
Agent to detect and annotate sensitive information in text,
meant to be used with a local (llama2, etc) model.
"""

import logging
from typing import List, Optional

from rich import print
from rich.console import Console

from langroid.language_models.base import LocalModelConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
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


class MaskAgentConfig(ChatAgentConfig):
    name = "MaskAgent"
    sensitive_categories: List[str]



class MaskAgentAttachment(ChatDocAttachment):
    content: str = ""


class MaskAgent(ChatAgent):
    def __init__(self, config: MaskAgentConfig):
        super().__init__(config)
        self.config: MaskAgentConfig = config
        local_model_config = LocalModelConfig(
            api_base="http://localhost:8000/v1",
            model="local",
            context_length=1024,
            # use our built-in chat completion formatting,
            # and hit the /completions endpoint instead of /chat/completions
            use_completion_for_chat=True,
        )
        self.llm = OpenAIGPTConfig(local=local_model_config)
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
                    msg.attachment, MaskAgentAttachment
            ):
                content = msg.attachment.content
            else:
                logger.warning("ValidatorAgent: Did not find content to correct")
                content = ""
            # we've used the attachment, don't need anymore
            attachment = MaskAgentAttachment(content="")
            # we are rewriting an LLM message from parent, so
            # pretend it is from LLM
            responder = Entity.LLM
            sender_name = ""
        else:
            # save the original message so when the Validator
            # receives the LLM clarification,
            # it can use it as the `content` field
            attachment = MaskAgentAttachment(content=content)
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
