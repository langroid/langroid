from typing import Type

import pytest
from pydantic import BaseModel, Field, HttpUrl

from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.agent.special.lance_doc_chat_agent import LanceDocChatAgent
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import Document
from langroid.parsing.parser import ParsingConfig, Splitter
from langroid.utils.configuration import Settings, set_global
from langroid.utils.system import rmdir
from langroid.vector_store.lancedb import LanceDBConfig


# Mock or simplified document and metadata models, now including HttpUrl
class MovieMetadata(BaseModel):
    year: int = Field(..., description="The year the movie was released.")
    website: HttpUrl = Field(..., description="The official movie website URL.")


class MovieDoc(Document):
    content: str = Field(..., description="A short description of the movie.")
    metadata: MovieMetadata


# Sample documents
movie_docs = [
    MovieDoc(
        content="A journey through space and time.",
        metadata=MovieMetadata(
            year=1999,
            website="https://example.com/movie",
        ),
    ),
]

embed_cfg = OpenAIEmbeddingsConfig()


@pytest.mark.parametrize("doc_cls", [MovieDoc])
def test_document_ingestion_with_lance_db(
    test_settings: Settings, doc_cls: Type[Document]
):
    set_global(test_settings)

    ldb_dir = ".lancedb/data/test-2"
    rmdir(ldb_dir)
    ldb_cfg = LanceDBConfig(
        cloud=False,
        collection_name="test-lance-2",
        storage_path=ldb_dir,
        embedding=embed_cfg,
        document_class=doc_cls,
    )

    cfg = DocChatAgentConfig(
        # turn cross-encoder off this off since it needs sentence-transformers
        cross_encoder_reranking_model="",
        vecdb=ldb_cfg,
        parsing=ParsingConfig(
            splitter=Splitter.SIMPLE,
            n_similar_docs=3,
        ),
    )

    agent = LanceDocChatAgent(cfg)

    # Ingest documents into LanceDB through the agent
    agent.ingest_docs(movie_docs)
    print(f"Ingested {len(movie_docs)} documents into LanceDB.")
