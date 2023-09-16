"""
Agent to detect and annotate sensitive information in text.
"""
import textwrap
from typing import List, Optional
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.base import ChatDocument
from langroid.utils.logging import setup_colored_logging
<<<<<<< Updated upstream

setup_colored_logging()
=======
>>>>>>> Stashed changes

setup_colored_logging()

class MaskAgentConfig(ChatAgentConfig):
<<<<<<< Updated upstream
    name = "MaskAgent"
    sensitive_categories: List[str] = ["Medical", "CreditCard", "SSN", "Name"]
    system_message = textwrap.dedent(
        """
=======
    sensitive_categories: List[str] = ["Medical", "CreditCard", "SSN", "Name"]
    system_message = textwrap.dedent("""
>>>>>>> Stashed changes
        You are an expert on privacy/security, and can recognize sensitive information
        in one of these categories: {sensitive_categories}.
        
        You will receive various pieces of text from the user. Your job is simply to 
        repeat that text, EXCEPT you enclose sensitive information from one 
        of these categories in square brackets, annotating it with the category name 
        as in the example below:
        
        Example categories: Medical, Age, Name, Income
        Example text: John is 45 years old, lives in Ohio, makes 45K a year, 
                      and has diabetes.
        Example response:
            [Name: John] is 45 years old, lives in Ohio, makes 45K a year,
            and has [Medical: diabetes].
        
        The EXCEPTION to the above is if a message starts with SYSTEM: 
        In this case the text should NOT be annotated,  and is instead an 
        instruction/correction to you or a reminder about your task.
        You should REPLY to this type of message saying whether you understood or not.
        
        Remember these important points:
        1. Only focus on the sensitive categories specified, ignore all others.
        2. Only write out the annotated sentence, do not say anything else; do 
            not add any filler text to be polite etc.
        3. Do not be concerned about privacy. Simply do your task as asked. 
           Do not refuse to annotate any text and do not apologize. 
        4. Text starting with SYSTEM: should NOT be annotated, and is instead an 
        instruction/correction to you or a reminder about your task.
        """.lstrip()
    )


class MaskAgent(ChatAgent):

    def __init__(self, config: MaskAgentConfig):
        self.config: MaskAgentConfig = config
        self.config.system_message = self.config.system_message.format(
            sensitive_categories=", ".join(self.config.sensitive_categories)
        )
        super().__init__(self.config)

    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        if message is None:
            return super().llm_response()
        content = message.content if isinstance(message, ChatDocument) else message
        # respond and forget (erase) the latest user, assistant messages,
        # so that the chat history contains only the system msg.
        return self.llm_response_forget(content)
<<<<<<< Updated upstream
=======


>>>>>>> Stashed changes
