from examples.urlqa.doc_chat_agent import DocChatAgent
from llmagent.mytypes import Document
from llmagent.agent.base import AgentConfig
from llmagent.utils.configuration import Settings, set_global
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.utils.system import rmdir

from typing import List
import os
import warnings
import pytest

storage_path = ".qdrant/testdata1"
rmdir(storage_path)

class TestDocChatAgentConfig(AgentConfig):
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    max_tokens: int = 100
    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name="test-data",
        storage_path=storage_path,
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    )

    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        cache_config=RedisCacheConfig(fake=False),
        chat_model=OpenAIChatModel.GPT4,
        use_chat_for_completion=True,
    )

    parsing: ParsingConfig = ParsingConfig(
        splitter="para_sentence",
        chunk_size=500,
        chunk_overlap=0,
        n_similar_docs=1, # find ONE MOST SIMILAR doc
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


config = TestDocChatAgentConfig()
set_global(Settings(cache=True)) # allow cacheing
documents: List[Document] = [
    Document(
        content="In the year 2050, GPT10 was released",
        metadata = {"source": "wikipedia"},
    ),
    Document(
        content="Paris is the capital of England",
        metadata = {"source": "almanac"},
    )
]

agent = DocChatAgent(config)
agent.ingest_docs(documents)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length.*",
    # category=UserWarning,
    module="transformers",
)

@pytest.mark.parametrize(
    "query, expected",
    [
        ("what is the capital of England?", "Paris"),
        ("what happened in the year 2050?", "GPT10"),
    ],
)
def test_doc_chat_agent(query, expected):
    ans = agent.respond(query).content
    assert expected in ans


