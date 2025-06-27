import json
import logging
from typing import Any, Dict

from langroid.pydantic_v1 import BaseModel
from langroid.utils.logging import setup_logger

logger = setup_logger(__name__, level=logging.INFO, terminal=True)


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


def load_system_messages(file_path: str) -> SystemMessages:
    """Load and validate system messages from a JSON file.

    Reads the JSON file containing system messages, maps each entry to a
    `Message` object, and wraps the result in a `SystemMessages` object.

    Args:
        file_path (str): The path to the JSON file containing system messages.

    Returns:
        SystemMessages: A `SystemMessages` object containing validated messages.

    Raises:
        IOError: If the file cannot be read or found.
        json.JSONDecodeError: If the JSON file is not properly formatted.
        Exception: For any other unexpected errors during processing.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data: Any = json.load(f)
        # Map dictionaries to Message objects
        messages = {key: Message(**value) for key, value in data.items()}
        return SystemMessages(messages=messages)
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise IOError(f"Could not find the file: {file_path}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file: {file_path}")
        raise json.JSONDecodeError(
            f"Invalid JSON format in file: {file_path}", e.doc, e.pos
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error loading system messages: {e}")
        raise
