from typing import Optional
from llmagent.agent.chat_agent import ChatAgent, ChatAgentConfig
from llmagent.agent.base import Entity
from llmagent.agent.chat_document import ChatDocument, ChatDocMetaData
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
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        # don't get user input
        return None

    def agent_response(
        self,
        input_str: Optional[str | ChatDocument] = None,
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

        has_func_call = False
        if isinstance(input_str, ChatDocument):
            recipient, content = input_str.recipient_message()
            has_func_call = input_str.function_call is not None
        else:
            recipient, content = parse_message(input_str)

        if has_func_call:
            error = """
            Expected `function_call` to have a "to" field. 
            Please resend with an appropriate "to" field, to clarify
            who the `function_call` is intended for.
            """
        else:
            error = """
            Please rewrite your message so it starts with 'TO[<recipient>]:'
            to clarify who the message is intended for. 
            """
        if recipient == "":
            return ChatDocument(
                content=error,
                metadata=ChatDocMetaData(
                    source=Entity.AGENT,
                    sender=Entity.AGENT,
                    sender_name=self.config.name,
                ),
            )
        else:
            # no objections, let the task loop continue
            return None
