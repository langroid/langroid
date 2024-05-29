from typing import Any

import pandas as pd
from _typeshed import Incomplete
from sentence_transformers import SentenceTransformer as SentenceTransformer

from langroid.agent.batch import run_batch_tasks as run_batch_tasks
from langroid.agent.chat_agent import (
    ChatAgent as ChatAgent,
)
from langroid.agent.chat_agent import (
    ChatAgentConfig as ChatAgentConfig,
)
from langroid.agent.chat_document import (
    ChatDocMetaData as ChatDocMetaData,
)
from langroid.agent.chat_document import (
    ChatDocument as ChatDocument,
)
from langroid.agent.special.relevance_extractor_agent import (
    RelevanceExtractorAgent as RelevanceExtractorAgent,
)
from langroid.agent.special.relevance_extractor_agent import (
    RelevanceExtractorAgentConfig as RelevanceExtractorAgentConfig,
)
from langroid.agent.task import Task as Task
from langroid.agent.tools.retrieval_tool import RetrievalTool as RetrievalTool
from langroid.embedding_models.models import (
    OpenAIEmbeddingsConfig as OpenAIEmbeddingsConfig,
)
from langroid.embedding_models.models import (
    SentenceTransformerEmbeddingsConfig as SentenceTransformerEmbeddingsConfig,
)
from langroid.language_models.base import StreamingIfAllowed as StreamingIfAllowed
from langroid.language_models.openai_gpt import (
    OpenAIChatModel as OpenAIChatModel,
)
from langroid.language_models.openai_gpt import (
    OpenAIGPTConfig as OpenAIGPTConfig,
)
from langroid.mytypes import (
    DocMetaData as DocMetaData,
)
from langroid.mytypes import (
    Document as Document,
)
from langroid.mytypes import (
    Entity as Entity,
)
from langroid.parsing.document_parser import DocumentType as DocumentType
from langroid.parsing.parser import (
    Parser as Parser,
)
from langroid.parsing.parser import (
    ParsingConfig as ParsingConfig,
)
from langroid.parsing.parser import (
    PdfParsingConfig as PdfParsingConfig,
)
from langroid.parsing.parser import (
    Splitter as Splitter,
)
from langroid.parsing.repo_loader import RepoLoader as RepoLoader
from langroid.parsing.search import (
    find_closest_matches_with_bm25 as find_closest_matches_with_bm25,
)
from langroid.parsing.search import (
    find_fuzzy_matches_in_docs as find_fuzzy_matches_in_docs,
)
from langroid.parsing.search import (
    preprocess_text as preprocess_text,
)
from langroid.parsing.table_loader import describe_dataframe as describe_dataframe
from langroid.parsing.url_loader import URLLoader as URLLoader
from langroid.parsing.urls import (
    get_list_from_user as get_list_from_user,
)
from langroid.parsing.urls import (
    get_urls_paths_bytes_indices as get_urls_paths_bytes_indices,
)
from langroid.parsing.utils import batched as batched
from langroid.prompts.prompts_config import PromptsConfig as PromptsConfig
from langroid.prompts.templates import (
    SUMMARY_ANSWER_PROMPT_GPT4 as SUMMARY_ANSWER_PROMPT_GPT4,
)
from langroid.utils.constants import NO_ANSWER as NO_ANSWER
from langroid.utils.output import show_if_debug as show_if_debug
from langroid.utils.output import status as status
from langroid.utils.pydantic_utils import (
    dataframe_to_documents as dataframe_to_documents,
)
from langroid.utils.pydantic_utils import (
    extract_fields as extract_fields,
)
from langroid.vector_store.base import (
    VectorStore as VectorStore,
)
from langroid.vector_store.base import (
    VectorStoreConfig as VectorStoreConfig,
)
from langroid.vector_store.lancedb import LanceDBConfig as LanceDBConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig as QdrantDBConfig

def apply_nest_asyncio() -> None: ...

logger: Incomplete
DEFAULT_DOC_CHAT_INSTRUCTIONS: str
DEFAULT_DOC_CHAT_SYSTEM_MESSAGE: str
has_sentence_transformers: bool

def extract_markdown_references(md_string: str) -> list[int]: ...
def format_footnote_text(content: str, width: int = 80) -> str: ...

class DocChatAgentConfig(ChatAgentConfig):
    system_message: str
    user_message: str
    summarize_prompt: str
    add_fields_to_content: list[str]
    filter_fields: list[str]
    retrieve_only: bool
    extraction_granularity: int
    filter: str | None
    conversation_mode: bool
    assistant_mode: bool
    hypothetical_answer: bool
    n_query_rephrases: int
    n_neighbor_chunks: int
    n_fuzzy_neighbor_words: int
    use_fuzzy_match: bool
    use_bm25_search: bool
    cross_encoder_reranking_model: str
    rerank_diversity: bool
    rerank_periphery: bool
    rerank_after_adding_context: bool
    embed_batch_size: int
    cache: bool
    debug: bool
    stream: bool
    split: bool
    relevance_extractor_config: None | RelevanceExtractorAgentConfig
    doc_paths: list[str | bytes]
    default_paths: list[str]
    parsing: ParsingConfig
    hf_embed_config: Incomplete
    oai_embed_config: Incomplete
    vecdb_config: VectorStoreConfig
    vecdb: VectorStoreConfig | None
    llm: OpenAIGPTConfig
    prompts: PromptsConfig

class DocChatAgent(ChatAgent):
    config: Incomplete
    original_docs: Incomplete
    original_docs_length: int
    from_dataframe: bool
    df_description: str
    chunked_docs: Incomplete
    chunked_docs_clean: Incomplete
    response: Incomplete
    def __init__(self, config: DocChatAgentConfig) -> None: ...
    vecdb: Incomplete
    def clear(self) -> None: ...
    def ingest(self) -> None: ...
    def ingest_doc_paths(
        self,
        paths: str | bytes | list[str | bytes],
        metadata: (
            list[dict[str, Any]] | dict[str, Any] | DocMetaData | list[DocMetaData]
        ) = [],
        doc_type: str | DocumentType | None = None,
    ) -> list[Document]: ...
    def ingest_docs(
        self,
        docs: list[Document],
        split: bool = True,
        metadata: (
            list[dict[str, Any]] | dict[str, Any] | DocMetaData | list[DocMetaData]
        ) = [],
    ) -> int: ...
    def retrieval_tool(self, msg: RetrievalTool) -> str: ...
    @staticmethod
    def document_compatible_dataframe(
        df: pd.DataFrame, content: str = "content", metadata: list[str] = []
    ) -> tuple[pd.DataFrame, list[str]]: ...
    def ingest_dataframe(
        self, df: pd.DataFrame, content: str = "content", metadata: list[str] = []
    ) -> int: ...
    def set_filter(self, filter: str) -> None: ...
    def setup_documents(
        self, docs: list[Document] = [], filter: str | None = None
    ) -> None: ...
    def get_field_values(self, fields: list[str]) -> dict[str, str]: ...
    def doc_length(self, docs: list[Document]) -> int: ...
    def user_docs_ingest_dialog(self) -> None: ...
    def llm_response(
        self, query: None | str | ChatDocument = None
    ) -> ChatDocument | None: ...
    async def llm_response_async(
        self, query: None | str | ChatDocument = None
    ) -> ChatDocument | None: ...
    @staticmethod
    def doc_string(docs: list[Document]) -> str: ...
    def get_summary_answer(
        self, question: str, passages: list[Document]
    ) -> ChatDocument: ...
    def llm_hypothetical_answer(self, query: str) -> str: ...
    def llm_rephrase_query(self, query: str) -> list[str]: ...
    def get_similar_chunks_bm25(
        self, query: str, multiple: int
    ) -> list[tuple[Document, float]]: ...
    def get_fuzzy_matches(self, query: str, multiple: int) -> list[Document]: ...
    def rerank_with_cross_encoder(
        self, query: str, passages: list[Document]
    ) -> list[Document]: ...
    def rerank_with_diversity(self, passages: list[Document]) -> list[Document]: ...
    def rerank_to_periphery(self, passages: list[Document]) -> list[Document]: ...
    def add_context_window(
        self, docs_scores: list[tuple[Document, float]]
    ) -> list[tuple[Document, float]]: ...
    def get_semantic_search_results(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]: ...
    def get_relevant_chunks(
        self, query: str, query_proxies: list[str] = []
    ) -> list[Document]: ...
    def get_relevant_extracts(self, query): ...
    def get_verbatim_extracts(
        self, query: str, passages: list[Document]
    ) -> list[Document]: ...
    def answer_from_docs(self, query: str) -> ChatDocument: ...
    def summarize_docs(
        self, instruction: str = "Give a concise summary of the following text:"
    ) -> None | ChatDocument: ...
    def justify_response(self) -> ChatDocument | None: ...
