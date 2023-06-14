from typing import Optional
from llmagent.agent.chat_agent import ChatAgent, ChatAgentConfig
from llmagent.agent.base import ChatDocMetaData, ChatDocument, Entity
from llmagent.parsing.agent_chats import parse_message

# TODO - this is currently hardocded to validate the TO:<recipient> format
# but we could have a much more general declarative grammar-based validator


class MessageValidatorAgent(ChatAgent):
    def __init__(self, config: ChatAgentConfig):
        super().__init__(config)
        self.config = config
        self.llm = None
        self.vecdb = None

    def user_response(
        self, msg: Optional[str] = None, sender_name: str = ""
    ) -> Optional[ChatDocument]:
        # don't get user input
        return None

    def agent_response(
        self, input_str: Optional[str] = None, sender_name: str = ""
    ) -> Optional[ChatDocument]:
        """
        Check whether the incoming message is in the expected format.
        Used to check whether the output of the LLM of the calling agent is
        in the expected format.

        Args:
            input_str (str): the incoming message (pending message of the task)
            sender_name (str): the name of the sender

        Returns:
            ChatDocument: None if message is in the expected format, otherwise
                a ChatDocument with an instruction on how to rewrite the message.
                (this is intended to be sent to the LLM of the calling agent).

        """
        if input_str is None:
            return None
        recipient, content = parse_message(input_str)
        if recipient == "":
            return ChatDocument(
                content="""
                    Expected a format of TO[<recipient>]:<message>.
                    Please rewrite your message in this format
                    """,
                metadata=ChatDocMetaData(
                    source=Entity.AGENT,
                    sender=Entity.AGENT,
                    sender_name=self.config.name,
                ),
            )
        else:
            # no objections, let the task loop continue
            return None
