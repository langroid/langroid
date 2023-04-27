"""General config settings for an LLM agent"""

from pydantic import BaseModel, Field
from llmagent.embedding_models.config import EmbeddingModelsConfig
from llmagent.vector_store.config import VectorStoreConfig
from llmagent.language_models.config import LLMConfig
from llmagent.parsing.config import ParsingConfig
from llmagent.prompts.config import PromptsConfig

class AgentConfig(BaseModel):
    """
    General config settings for an LLM agent. This is nested, combining configs of
    various components, in a hierarchy. Let us see how this works.
    """
    name: str = "llmagent"
    debug: bool = False
    embeddings: EmbeddingModelsConfig = Field(default_factory=EmbeddingModelsConfig)
    vecdb: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)






