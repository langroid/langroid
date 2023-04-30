from dataclasses import dataclass, field
from llmagent.agent.base import AgentConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.config import PromptsConfig
from hydra.core.config_store import ConfigStore
from typing import List

@dataclass
class CustomAgentConfig(AgentConfig):
    max_tokens: int = 10000
    vecdb: VectorStoreConfig = field(
        default_factory=lambda:
        QdrantDBConfig(
            type="qdrant",
            collection_name="test",
            storage_path=".qdrant/test/",
            embedding=OpenAIEmbeddingsConfig(
                model_type="openai",
                model_name="text-embedding-ada-002",
                dims=1536,
            )))
            # embedding=SentenceTransformerEmbeddingsConfig(
            #     model_type="sentence-transformer",
            #     model_name="all-MiniLM-L6-v2",
            #     dims=384,
            # )
        ##))
    llm: LLMConfig = field(
        default_factory=lambda:
        LLMConfig(
            type="openai",
        ))

    parsing: ParsingConfig = field(
        default_factory=lambda:
        ParsingConfig(
            chunk_size=500,
            chunk_overlap=50,
        ))

    prompts: PromptsConfig = field(
        default_factory=lambda:
        PromptsConfig(
            max_tokens=1000,
        ))

    urls: List[str] = field(default_factory=lambda: [
        "https://news.ycombinator.com/item?id=35629033",
        "https://www.newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-jpeg-of-the-web",
        "https://www.wired.com/1995/04/maes/",
        "https://cthiriet.com/articles/scaling-laws",
        "https://www.jasonwei.net/blog/emergence",
        "https://www.quantamagazine.org/the-unpredictable-abilities-emerging-from-large-ai-models-20230316/",
        "https://ai.googleblog.com/2022/11/characterizing-emergent-phenomena-in.html",
    ])


cs = ConfigStore.instance()
cs.store(name="tests.configs.config", node=CustomAgentConfig)