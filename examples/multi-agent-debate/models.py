from pydantic import BaseModel
from typing import Dict


class Message(BaseModel):
    topic: str
    message: str


class SystemMessages(BaseModel):
    messages: Dict[str, Message]
