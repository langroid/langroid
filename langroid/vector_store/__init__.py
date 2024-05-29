from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import base

    from . import qdrantdb
    from . import meilisearch
    from . import lancedb

    from .base import VectorStoreConfig, VectorStore
    from .qdrantdb import QdrantDBConfig, QdrantDB
    from .meilisearch import MeiliSearch, MeiliSearchConfig
    from .lancedb import LanceDB, LanceDBConfig

    from . import chromadb
    from .chromadb import ChromaDBConfig, ChromaDB

    chromadb  # silence linters
    ChromaDB
    ChromaDBConfig

else:

    base = LazyLoad("langroid.vector_store.base")
    qdrantdb = LazyLoad("langroid.vector_store.qdrantdb")
    VectorStore = LazyLoad("langroid.vector_store.base.VectorStore")
    VectorStoreConfig = LazyLoad("langroid.vector_store.base.VectorStoreConfig")
    QdrantDB = LazyLoad("langroid.vector_store.qdrantdb.QdrantDB")
    QdrantDBConfig = LazyLoad("langroid.vector_store.qdrantdb.QdrantDBConfig")

    lancedb = LazyLoad("langroid.vector_store.lancedb")
    LanceDB = LazyLoad("langroid.vector_store.lancedb.LanceDB")
    LanceDBConfig = LazyLoad("langroid.vector_store.lancedb.LanceDBConfig")

    meilisearch = LazyLoad("langroid.vector_store.meilisearch")
    MeiliSearch = LazyLoad("langroid.vector_store.meilisearch.MeiliSearch")
    MeiliSearchConfig = LazyLoad("langroid.vector_store.meilisearch.MeiliSearchConfig")

    chromadb = LazyLoad("langroid.vector_store.chromadb")
    ChromaDB = LazyLoad("langroid.vector_store.chromadb.ChromaDB")
    ChromaDBConfig = LazyLoad("langroid.vector_store.chromadb.ChromaDBConfig")

__all__ = [
    "base",
    "VectorStore",
    "VectorStoreConfig",
    "qdrantdb",
    "QdrantDBConfig",
    "QdrantDB",
    "meilisearch",
    "MeiliSearch",
    "MeiliSearchConfig",
    "lancedb",
    "LanceDB",
    "LanceDBConfig",
    "chromadb",
    "ChromaDB",
    "ChromaDBConfig",
]
