from . import base

from . import qdrantdb
from . import meilisearch
from . import lancedb

from .base import VectorStoreConfig, VectorStore
from .qdrantdb import QdrantDBConfig, QdrantDB
from .meilisearch import MeiliSearch, MeiliSearchConfig
from .lancedb import LanceDB, LanceDBConfig

has_chromadb = False
try:
    from . import chromadb
    from .chromadb import ChromaDBConfig, ChromaDB

    chromadb  # silence linters
    ChromaDB
    ChromaDBConfig
    has_chromadb = True
except ImportError:
    pass

__all__ = [
    "base",
    "VectorStore",
    "VectorStoreConfig",
    "qdrantdb",
    "meilisearch",
    "lancedb",
    "QdrantDBConfig",
    "QdrantDB",
    "MeiliSearch",
    "MeiliSearchConfig",
    "LanceDB",
    "LanceDBConfig",
]

if has_chromadb:
    __all__.extend(["chromadb", "ChromaDBConfig", "ChromaDB"])
