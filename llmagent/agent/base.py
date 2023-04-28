from abc import ABC, abstractmethod
from llmagent.agent.config import AgentConfig
from llmagent.language_models.base import LanguageModel
from llmagent.vector_store.base import VectorStore
from llmagent.parsing.parser import Parser
from pydantic import BaseModel
from dataclasses import asdict
from llmagent.mytypes import Document
from rich import print

class Message(BaseModel):
    role: str
    content: str

    def message(self):
        return {"role": self.role, "content": self.content}


class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.chat_history = [] # list of (prompt, response) tuples
        self.response:Document = None # last response

        self.llm = LanguageModel.create(config.llm)
        self.vecdb = VectorStore.create(config.vecdb)
        self.parser = Parser(**config.parsing)

    def update_history(self, prompt, output):
        self.chat_history.append((prompt, output))

    def get_history(self):
        return self.chat_history

    def respond(self, query:str):
        response = self.llm.generate(query, self.config.llm.max_tokens)
        print("[green]", response)
