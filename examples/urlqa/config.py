from examples.urlqa.doc_chat_agent import DocChatAgentConfig
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig

from typing import List


class URLQAConfig(DocChatAgentConfig):
    debug: bool = False
    max_context_tokens = 500
    conversation_mode = True
    cache: bool = True  # cache results
    gpt4: bool = False  # use GPT-4
    stream: bool = True  # allow streaming where needed
    max_tokens: int = 10000
    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name="llmagent-urls",
        storage_path=".qdrant/data/",
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    )

    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
    )
    parsing: ParsingConfig = ParsingConfig(
        splitter="tokens",
        chunk_size=200,
        chunk_overlap=50,
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )

    urls: List[str] = [
        "https://news.ycombinator.com/item?id=35629033",
        "https://www.newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-jpeg-of-the-web",
        "https://www.wired.com/1995/04/maes/",
        "https://cthiriet.com/articles/scaling-laws",
        "https://www.jasonwei.net/blog/emergence",
        "https://www.quantamagazine.org/the-unpredictable-abilities-emerging-from-large-ai-models-20230316/",
        "https://ai.googleblog.com/2022/11/characterizing-emergent-phenomena-in.html",
    ]
