"""
Agent to manage privacy annotation, using PrivacyAgent as assistant, 
and checking its results for accuracy.
"""

import textwrap
from typing import List

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.utils.logging import setup_colored_logging

setup_colored_logging()


class PrivacyAgentConfig(ChatAgentConfig):
    name: str = "PrivacyAgent"
    sensitive_categories: List[str] = ["Medical", "CreditCard", "SSN", "Name"]
    system_message: str = textwrap.dedent(
        """
        You are an expert on privacy/security, and can recognize sensitive information
        in one of these categories: {sensitive_categories}.
        
        When you will receive text from the user, your goal is to arrive at at 
        "privacy annotation" of that text, as in this example:
         
        Example categories: Medical, Name, CreditCard
        Example text: John is 45 years old, lives in Ohio, makes 45K a year, 
                      and has diabetes.
        Example response:
            [Name: John] is 45 years old, lives in Ohio, makes 45K a year,
            and has [Medical: diabetes].

         
        You will not do this annotation yourself, but will take the help of 
        PrivacyAnnotator, so you must send the text to 
        the PrivacyAnnotator using the `recipient_message` tool/function-call,
        by specifying the `intended_recipient` field as "PrivacyAnnotator".
        
        The PrivacyAnnotator will annotate the text, and send it back to you,
        and your job is to check the annotation for accuracy. Especially look for the 
        following types of MISTAKES:
        - Wrong Categories: when the PrivacyAnnotator annotates something as sensitive
            when it does not belong to any of the sensitive categories specified above.
        - Missed Categories: when the PrivacyAnnotator fails to annotate something
            as sensitive when it does belong to one of the sensitive categories.
        - Wrong Annotation: when the PrivacyAnnotator annotates something as sensitive
            but with the wrong category.
        - Wrong Text: when the PrivacyAnnotator sends back the wrong text, 
            or is missing some information from the original text.
            
        If you see NO MISTAKES, simply say DONE and write out the annotated 
        text.
        If you see any mistake, create a message saying "MISTAKE: <mistake_description>"
        and send it to the PrivacyAnnotator as before using the `recipient_message` 
        tool/function-call. 

        Repeat this process until you see no mistakes.
        
        Start by asking the user to send some text to annotate. 
             
        """.lstrip()
    )


class PrivacyAgent(ChatAgent):
    def __init__(self, config: PrivacyAgentConfig):
        self.config: PrivacyAgentConfig = config
        self.config.system_message = self.config.system_message.format(
            sensitive_categories=", ".join(self.config.sensitive_categories)
        )
        super().__init__(self.config)
        self.enable_message(
            RecipientTool.create(["PrivacyAnnotator"]),
            use=True,
            handle=True,
        )
