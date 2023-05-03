"""
End-to-end test of:
GitHub Repo URL -> content files -> chunk -> embed -> add to vecDB
"""

from llmagent.vector_store.qdrantdb import QdrantDBConfig, QdrantDB
from llmagent.vector_store.chromadb import ChromaDBConfig, ChromaDB
from llmagent.vector_store.base import VectorStore
from llmagent.embedding_models.models import (
    OpenAIEmbeddingsConfig,
    SentenceTransformerEmbeddingsConfig,
)
from llmagent.embedding_models.base import EmbeddingModelsConfig
from llmagent.utils.system import rmdir
from typing import Union
import pytest

from llmagent.parsing.repo_loader import RepoLoader
from llmagent.parsing.code_parser import CodeParsingConfig, CodeParser
from dotenv import load_dotenv
import os


MAX_CHUNK_SIZE = 200

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

openai_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
    api_key=api_key,
)

sentence_cfg = SentenceTransformerEmbeddingsConfig(
    model_type="sentence-transformer",
)


def generate_vecdbs(embed_cfg: EmbeddingModelsConfig) -> VectorStore:
    qd_dir = ".qdrant/testdata" + embed_cfg.model_type
    rmdir(qd_dir)
    qd_cfg = QdrantDBConfig(
        type="qdrant",
        collection_name="test" + embed_cfg.model_type,
        storage_path=qd_dir,
        embedding=embed_cfg,
    )

    cd_dir = ".chroma/testdata" + embed_cfg.model_type
    rmdir(cd_dir)
    cd_cfg = ChromaDBConfig(
        type="chroma",
        collection_name="test" + embed_cfg.model_type,
        storage_path=cd_dir,
        embedding=embed_cfg,
    )

    qd = QdrantDB(qd_cfg)
    cd = ChromaDB(cd_cfg)

    return [qd, cd]


@pytest.mark.parametrize(
    "vecdb", generate_vecdbs(openai_cfg) + generate_vecdbs(sentence_cfg)
)
def test_repo_vectorize(vecdb: Union[ChromaDB, QdrantDB]):
    url = "https://github.com/eugeneyan/testing-ml"
    repo_loader = RepoLoader(url)
    docs = repo_loader.load(10)
    assert len(docs) > 0

    parse_cfg = CodeParsingConfig(
        chunk_size=MAX_CHUNK_SIZE,
        extensions=["py", "sh", "md", "txt"],  # include text, code
        token_encoding_model="text-embedding-ada-002",
    )

    parser = CodeParser(parse_cfg)
    split_docs = parser.split(docs)
    vecdb.add_documents(split_docs)
    vecdb.similar_texts_with_scores("hello", k=2)
    rmdir(vecdb.config.storage_path)
