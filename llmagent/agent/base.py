from abc import ABC
from pydantic import BaseModel, BaseSettings
from halo import Halo
from llmagent.mytypes import Document
from rich import print

from llmagent.language_models.base import LanguageModel
from llmagent.vector_store.base import VectorStore
from llmagent.parsing.parser import Parser
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig


class AgentConfig(BaseSettings):
    """
    General config settings for an LLM agent. This is nested, combining configs of
    various components, in a hierarchy. Let us see how this works.
    """

    name: str = "llmagent"
    debug: bool = False
    vecdb: VectorStoreConfig = VectorStoreConfig()
    llm: LLMConfig = LLMConfig()
    parsing: ParsingConfig = ParsingConfig()
    prompts: PromptsConfig = PromptsConfig()


class Message(BaseModel):
    role: str
    content: str

    def message(self):
        return {"role": self.role, "content": self.content}


class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.chat_history = []  # list of (prompt, response) tuples
        self.response: Document = None  # last response

        self.llm = LanguageModel.create(config.llm)
        self.vecdb = VectorStore.create(config.vecdb)
        self.parser = Parser(config.parsing)

    def update_history(self, prompt, output):
        self.chat_history.append((prompt, output))

    def get_history(self):
        return self.chat_history

    def respond(self, query: str) -> Document:
        """
        Respond to a query.
        Args:
            query:

        Returns:
            Document
        """
        with Halo(text="LLM query...", spinner="dots"):
            response = self.llm.generate(query, self.config.llm.max_tokens)
        print("[green]" + response.message)
        return Document(content=response.message, metadata={"source": "LLM"})
