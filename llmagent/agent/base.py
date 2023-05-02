from abc import ABC
from typing import List
from contextlib import ExitStack
from pydantic import BaseModel, BaseSettings
from halo import Halo
from llmagent.mytypes import Document
from rich import print

from llmagent.language_models.base import LanguageModel, LLMMessage
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
    stream: bool = False  # stream LLM output?
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
            query: query string
        Returns:
            Document
        """

        with ExitStack() as stack:  # for conditionally using Halo spinner
            if not self.llm.get_stream():
                # show Halo spinner only if not streaming!
                cm = Halo(text="LLM responding to message...", spinner="dots")
                stack.enter_context(cm)
            response = self.llm.generate(query, self.config.llm.max_tokens)
        if not self.llm.get_stream():
            print("[green]" + response.message)

        return Document(
            content=response.message, metadata=dict(source="LLM", usage=response.usage)
        )

    def respond_messages(self, messages: List[LLMMessage]) -> Document:
        """
        Respond to a series of messages, e.g. with OpenAI ChatCompletion
        Args:
            messages: seq of messages (with role, content fields) sent to LLM
        Returns:
            Document (i.e. with fields "content", "metadata")
        """
        with ExitStack() as stack:  # for conditionally using Halo spinner
            if not self.llm.get_stream():
                # show Halo spinner only if not streaming!
                cm = Halo(text="LLM responding to messages...", spinner="dots")
                stack.enter_context(cm)
            response = self.llm.chat(messages, self.config.llm.max_tokens)
        if not self.llm.get_stream():
            print("[green]" + response.message)
        return Document(
            content=response.message, metadata=dict(source="LLM", usage=response.usage)
        )
