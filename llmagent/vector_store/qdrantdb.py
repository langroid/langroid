from llmagent.vector_store.base import VectorStore, VectorStoreConfig
from llmagent.embedding_models.base import (
    EmbeddingModelsConfig,
    EmbeddingModel,
)
from llmagent.mytypes import Document
from llmagent.utils.configuration import settings
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    CollectionStatus,
    Batch,
    SearchParams,
)
from qdrant_client.conversions.common_types import ScoredPoint
from typing import List, Tuple
from chromadb.api.types import EmbeddingFunction
import logging

logger = logging.getLogger(__name__)


class QdrantDBConfig(VectorStoreConfig):
    type: str = "qdrant"
    collection_name: str = "qdrant-llmagent"
    storage_path: str = ".qdrant/data"
    embedding: EmbeddingModelsConfig = EmbeddingModelsConfig(
        model_type="openai",
    )
    distance: str = Distance.COSINE
    # host: str = "127.0.0.1"
    # port: int = 6333
    # compose_file: str = "llmagent/vector_store/docker-compose-qdrant.yml"


class QdrantDB(VectorStore):
    def __init__(self, config: QdrantDBConfig):
        super().__init__()
        self.config = config
        emb_model = EmbeddingModel.create(config.embedding)
        self.embedding_fn: EmbeddingFunction = emb_model.embedding_fn()
        self.embedding_dim = emb_model.embedding_dims
        self.host = config.host
        self.port = config.port
        self.client = QdrantClient(
            path=config.storage_path,
            # host=self.host,
            # port=self.port,
            # prefer_grpc=True,
        )

        self.client.recreate_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        collection_info = self.client.get_collection(
            collection_name=config.collection_name
        )
        assert collection_info.status == CollectionStatus.GREEN
        assert collection_info.vectors_count == 0
        if settings.debug:
            level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)
            logger.info(collection_info)
            logger.setLevel(level)

    def add_documents(self, documents: List[Document]) -> None:
        embedding_vecs = self.embedding_fn([doc.content for doc in documents])
        ids = list(range(len(documents)))
        self.client.upsert(
            collection_name=self.config.collection_name,
            points=Batch(
                ids=ids,  # TODO do we need ids?
                vectors=embedding_vecs,
                payloads=documents,
            ),
        )

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = None,
        where: str = None,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_fn([text])[0]
        # TODO filter may not work yet
        filter = Filter() if where is None else Filter.from_json(where)
        search_result: List[ScoredPoint] = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=embedding,
            query_filter=filter,
            limit=k,
            search_params=SearchParams(
                hnsw_ef=128,
                exact=False,  # use Apx NN, not exact NN
            ),
        )
        scores = [match.score for match in search_result]
        docs = [Document(**(match.payload)) for match in search_result]
        if settings.debug:
            logger.info(f"Found {len(docs)} matches, max score: {max(scores)}")
        doc_score_pairs = list(zip(docs, scores))
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs
