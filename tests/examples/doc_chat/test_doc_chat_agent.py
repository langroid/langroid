from examples.urlqa.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from llmagent.mytypes import Document
from llmagent.utils.configuration import Settings, set_global
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.parsing.parser import ParsingConfig, Splitter
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.utils.system import rmdir
from llmagent.parsing.utils import generate_random_text


from typing import List
import os
import warnings
import pytest

storage_path = ".qdrant/testdata1"
rmdir(storage_path)


class _TestDocChatAgentConfig(DocChatAgentConfig):
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    max_tokens: int = 100
    conversation_mode = True
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
        stream=True,
        cache_config=RedisCacheConfig(fake=False),
        chat_model=OpenAIChatModel.GPT4,
        use_chat_for_completion=True,
    )

    parsing: ParsingConfig = ParsingConfig(
        splitter=Splitter.TOKENS,
        chunk_size=500,
        n_similar_docs=2,
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


config = _TestDocChatAgentConfig()
set_global(Settings(cache=True))  # allow cacheing
documents: List[Document] = (
    [
        Document(
            content="""
        In the year 2050, GPT10 was released. In 2057,
        paperclips were seen all over the world. Global
        warming was solved in 2060. In 2061, the world
        was taken over by paperclips. 
        
        In 2045, the Tour de France was still going on.
        They were still using bicycles. There was one more ice age in 2040.
        """,
            metadata={"source": "wikipedia"},
        ),
        Document(
            content="""
        We are living in an alternate universe where Paris is the capital of England.
        The capital of England used to be London. 
        The capital of France used to be Paris.
        Charlie Chaplin was a great comedian.
        In 2050, all countries merged into Lithuania.
        """,
            metadata={"source": "almanac"},
        ),
    ]
    + [
        Document(content=generate_random_text(5), metadata={"source": "random"})
        for _ in range(10)
    ]
)


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
        ("what happened in the year 2050?", "GPT10, Lithuania"),
        ("Who was Charlie Chaplin?", "comedian"),
        # ("What was the old capital of England?", "London"), this often fails!!
        ("What was the old capital of France?", "Paris"),
        ("When was global warming solved?", "2060"),
        ("what is the capital of England?", "Paris"),
        ("What do we know about paperclips?", "2057, 2061"),
    ],
)
def test_doc_chat_agent(test_settings: Settings, query: str, expected: str):
    # set_global(Settings(debug=options.show, cache=not options.nocache))
    # note that the (query, ans) pairs are accumulated into the
    # internal dialog history of the agent.
    set_global(test_settings)
    ans = agent.respond(query).content
    expected = [e.strip() for e in expected.split(",")]
    assert all([e in ans for e in expected])
