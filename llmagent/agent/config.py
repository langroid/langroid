"""General config settings for an LLM agent"""

from dataclasses import dataclass, field
from llmagent.embedding_models.config import EmbeddingModelsConfig
from llmagent.vector_store.config import VectorStoreConfig
from llmagent.language_models.config import LLMConfig
from llmagent.parsing.config import ParsingConfig
from llmagent.prompts.config import PromptsConfig

@dataclass
class AgentConfig:
    """
    General config settings for an LLM agent. This is nested, combining configs of
    various components, in a hierarchy. Let us see how this works.
    """
    name: str = "llmagent"
    debug: bool = False
    embeddings: EmbeddingModelsConfig = field(default_factory=EmbeddingModelsConfig)
    vecdb: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)






