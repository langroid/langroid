from pydantic import BaseModel
from typing import Dict


class Message(BaseModel):
    """Represents a single message with a topic and content.

    Attributes:
        topic (str): The topic of the message.
        message (str): The content of the message.
    """
    topic: str
    message: str


class SystemMessages(BaseModel):
    """Represents a collection of system messages.

    Attributes:
        messages (Dict[str, Message]): A dictionary where the key is the message
        identifier (e.g., 'pro_ai') and the value is a `Message` object.
    """
    messages: Dict[str, Message]
