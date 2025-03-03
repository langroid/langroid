import hashlib
import json
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypeVar

from dotenv import load_dotenv

from langroid.embedding_models.base import (
    EmbeddingModelsConfig,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import Document, Embeddings
from langroid.utils.configuration import settings
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from qdrant_client.http.models import SparseVector


T = TypeVar("T")


def from_optional(x: Optional[T], default: T) -> T:
    if x is None:
        return default

    return x


def is_valid_uuid(uuid_to_test: str) -> bool:
    """
    Check if a given string is a valid UUID.
    """
    try:
        uuid_obj = uuid.UUID(uuid_to_test)
        return str(uuid_obj) == uuid_to_test
    except Exception:
        pass
    # Check for valid unsigned 64-bit integer
    try:
        int_value = int(uuid_to_test)
        return 0 <= int_value <= 18446744073709551615
    except ValueError:
        return False


class QdrantDBConfig(VectorStoreConfig):
    cloud: bool = True
    docker: bool = False
    collection_name: str | None = "temp"
    storage_path: str = ".qdrant/data"
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    use_sparse_embeddings: bool = False
    sparse_embedding_model: str = "naver/splade-v3-distilbert"
    sparse_limit: int = 3
    distance: str = "cosine"


class QdrantDB(VectorStore):
    def __init__(self, config: QdrantDBConfig = QdrantDBConfig()):
        super().__init__(config)
        self.config: QdrantDBConfig = config
        from qdrant_client import QdrantClient

        if self.config.use_sparse_embeddings:
            try:
                from transformers import AutoModelForMaskedLM, AutoTokenizer
            except ImportError:
                raise ImportError(
                    """
                    To use sparse embeddings,
                    you must install langroid with the [transformers] extra, e.g.:
                    pip install "langroid[transformers]"
                    """
                )

            self.sparse_tokenizer = AutoTokenizer.from_pretrained(
                self.config.sparse_embedding_model
            )
            self.sparse_model = AutoModelForMaskedLM.from_pretrained(
                self.config.sparse_embedding_model
            )
        self.host = config.host
        self.port = config.port
        load_dotenv()
        key = os.getenv("QDRANT_API_KEY")
        url = os.getenv("QDRANT_API_URL")
        if config.docker:
            if url is None:
                logger.warning(
                    f"""The QDRANT_API_URL env variable must be set to use
                    QdrantDB in local docker mode. Please set this
                    value in your .env file.
                    Switching to local storage at {config.storage_path}
                    """
                )
                config.cloud = False
            else:
                config.cloud = True
        elif config.cloud and None in [key, url]:
            logger.warning(
                f"""QDRANT_API_KEY, QDRANT_API_URL env variable must be set to use
                QdrantDB in cloud mode. Please set these values
                in your .env file.
                Switching to local storage at {config.storage_path}
                """
            )
            config.cloud = False

        if config.cloud:
            self.client = QdrantClient(
                url=url,
                api_key=key,
                timeout=config.timeout,
            )
        else:
            try:
                self.client = QdrantClient(
                    path=config.storage_path,
                )
            except Exception as e:
                new_storage_path = config.storage_path + ".new"
                logger.warning(
                    f"""
                    Error connecting to local QdrantDB at {config.storage_path}:
                    {e}
                    Switching to {new_storage_path}
                    """
                )
                self.client = QdrantClient(
                    path=new_storage_path,
                )

        # Note: Only create collection if a non-null collection name is provided.
        # This is useful to delay creation of vecdb until we have a suitable
        # collection name (e.g. we could get it from the url or folder path).
        if config.collection_name is not None:
            self.create_collection(
                config.collection_name, replace=config.replace_collection
            )

    def clear_empty_collections(self) -> int:
        coll_names = self.list_collections()
        n_deletes = 0
        for name in coll_names:
            info = self.client.get_collection(collection_name=name)
            if info.points_count == 0:
                n_deletes += 1
                self.client.delete_collection(collection_name=name)
        return n_deletes

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        """Clear all collections with the given prefix."""

        if not really:
            logger.warning("Not deleting all collections, set really=True to confirm")
            return 0
        coll_names = [
            c for c in self.list_collections(empty=True) if c.startswith(prefix)
        ]
        if len(coll_names) == 0:
            logger.warning(f"No collections found with prefix {prefix}")
            return 0
        n_empty_deletes = 0
        n_non_empty_deletes = 0
        for name in coll_names:
            info = self.client.get_collection(collection_name=name)
            points_count = from_optional(info.points_count, 0)

            n_empty_deletes += points_count == 0
            n_non_empty_deletes += points_count > 0
            self.client.delete_collection(collection_name=name)
        logger.warning(
            f"""
            Deleted {n_empty_deletes} empty collections and
            {n_non_empty_deletes} non-empty collections.
            """
        )
        return n_empty_deletes + n_non_empty_deletes

    def list_collections(self, empty: bool = False) -> List[str]:
        """
        Returns:
            List of collection names that have at least one vector.

        Args:
            empty (bool, optional): Whether to include empty collections.
        """

        colls = list(self.client.get_collections())[0][1]
        if empty:
            return [coll.name for coll in colls]
        counts = []
        for coll in colls:
            try:
                counts.append(
                    from_optional(
                        self.client.get_collection(
                            collection_name=coll.name
                        ).points_count,
                        0,
                    )
                )
            except Exception:
                logger.warning(f"Error getting collection {coll.name}")
                counts.append(0)
        return [coll.name for coll, count in zip(colls, counts) if (count or 0) > 0]

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Create a collection with the given name, optionally replacing an existing
            collection if `replace` is True.
        Args:
            collection_name (str): Name of the collection to create.
            replace (bool): Whether to replace an existing collection
                with the same name. Defaults to False.
        """
        from qdrant_client.http.models import (
            CollectionStatus,
            Distance,
            SparseIndexParams,
            SparseVectorParams,
            VectorParams,
        )

        self.config.collection_name = collection_name
        if self.client.collection_exists(collection_name=collection_name):
            coll = self.client.get_collection(collection_name=collection_name)
            if (
                coll.status == CollectionStatus.GREEN
                and from_optional(coll.points_count, 0) > 0
            ):
                logger.warning(f"Non-empty Collection {collection_name} already exists")
                if not replace:
                    logger.warning("Not replacing collection")
                    return
                else:
                    logger.warning("Recreating fresh collection")
            self.client.delete_collection(collection_name=collection_name)

        vectors_config = {
            "": VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
            )
        }
        sparse_vectors_config = None
        if self.config.use_sparse_embeddings:
            sparse_vectors_config = {
                "text-sparse": SparseVectorParams(index=SparseIndexParams())
            }
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        collection_info = self.client.get_collection(collection_name=collection_name)
        assert collection_info.status == CollectionStatus.GREEN
        assert collection_info.vectors_count in [0, None]
        if settings.debug:
            level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)
            logger.info(collection_info)
            logger.setLevel(level)

    def get_sparse_embeddings(self, inputs: List[str]) -> List["SparseVector"]:
        from qdrant_client.http.models import SparseVector

        if not self.config.use_sparse_embeddings:
            return []
        import torch

        tokens = self.sparse_tokenizer(
            inputs, return_tensors="pt", truncation=True, padding=True
        )
        output = self.sparse_model(**tokens)
        vectors = torch.max(
            torch.log(torch.relu(output.logits) + torch.tensor(1.0))
            * tokens.attention_mask.unsqueeze(-1),
            dim=1,
        )[0].squeeze(dim=1)
        sparse_embeddings = []
        for vec in vectors:
            cols = vec.nonzero().squeeze().cpu().tolist()
            weights = vec[cols].cpu().tolist()
            sparse_embeddings.append(
                SparseVector(
                    indices=cols,
                    values=weights,
                )
            )
        return sparse_embeddings

    def add_documents(self, documents: Sequence[Document]) -> None:
        from qdrant_client.http.models import (
            Batch,
            CollectionStatus,
            SparseVector,
        )

        # Add id to metadata if not already present
        super().maybe_add_ids(documents)
        # Fix the ids due to qdrant finickiness
        for doc in documents:
            doc.metadata.id = str(self._to_int_or_uuid(doc.metadata.id))
        colls = self.list_collections(empty=True)
        if len(documents) == 0:
            return
        document_dicts = [doc.dict() for doc in documents]
        embedding_vecs = self.embedding_fn([doc.content for doc in documents])
        sparse_embedding_vecs = self.get_sparse_embeddings(
            [doc.content for doc in documents]
        )
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot ingest docs")
        if self.config.collection_name not in colls:
            self.create_collection(self.config.collection_name, replace=True)
        ids = [self._to_int_or_uuid(d.id()) for d in documents]
        # don't insert all at once, batch in chunks of b,
        # else we get an API error
        b = self.config.batch_size
        for i in range(0, len(ids), b):
            vectors: Dict[str, Embeddings | List[SparseVector]] = {
                "": embedding_vecs[i : i + b]
            }
            if self.config.use_sparse_embeddings:
                vectors["text-sparse"] = sparse_embedding_vecs[i : i + b]
            coll_found: bool = False
            for _ in range(3):
                # poll until collection is ready
                if (
                    self.client.collection_exists(self.config.collection_name)
                    and self.client.get_collection(self.config.collection_name).status
                    == CollectionStatus.GREEN
                ):
                    coll_found = True
                    break
                time.sleep(1)

            if not coll_found:
                raise ValueError(
                    f"""
                    QdrantDB Collection {self.config.collection_name} 
                    not found or not ready
                    """
                )

            self.client.upsert(
                collection_name=self.config.collection_name,
                points=Batch(
                    ids=ids[i : i + b],
                    vectors=vectors,
                    payloads=document_dicts[i : i + b],
                ),
            )

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name=collection_name)

    def _to_int_or_uuid(self, id: str) -> int | str:
        try:
            int_val = int(id)
            if is_valid_uuid(id):
                return int_val
        except ValueError:
            pass

        # If doc_id is already a valid UUID, return it as is
        if isinstance(id, str) and is_valid_uuid(id):
            return id

        # Otherwise, generate a UUID from the doc_id
        # Convert doc_id to string if it's not already
        id_str = str(id)

        # Hash the document ID using SHA-1
        hash_object = hashlib.sha1(id_str.encode())
        hash_digest = hash_object.hexdigest()

        # Truncate or manipulate the hash to fit into a UUID (128 bits)
        uuid_str = hash_digest[:32]

        # Format this string into a UUID format
        formatted_uuid = uuid.UUID(uuid_str)

        return str(formatted_uuid)

    def get_all_documents(self, where: str = "") -> List[Document]:
        from qdrant_client.http.models import (
            Filter,
        )

        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        docs = []
        offset = 0
        filter = Filter() if where == "" else Filter.parse_obj(json.loads(where))
        while True:
            results, next_page_offset = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=filter,
                offset=offset,
                limit=10_000,  # try getting all at once, if not we keep paging
                with_payload=True,
                with_vectors=False,
            )
            docs += [
                self.config.document_class(**record.payload)  # type: ignore
                for record in results
            ]
            # ignore
            if next_page_offset is None:
                break
            offset = next_page_offset  # type: ignore
        return docs

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        _ids = [self._to_int_or_uuid(id) for id in ids]
        records = self.client.retrieve(
            collection_name=self.config.collection_name,
            ids=_ids,
            with_vectors=False,
            with_payload=True,
        )
        # Note the records may NOT be in the order of the ids,
        # so we re-order them here.
        id2payload = {record.id: record.payload for record in records}
        ordered_payloads = [id2payload[id] for id in _ids if id in id2payload]
        docs = [Document(**payload) for payload in ordered_payloads]  # type: ignore
        return docs

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
        neighbors: int = 0,
    ) -> List[Tuple[Document, float]]:
        from qdrant_client.conversions.common_types import ScoredPoint
        from qdrant_client.http.models import (
            Filter,
            NamedSparseVector,
            NamedVector,
            SearchRequest,
        )

        embedding = self.embedding_fn([text])[0]
        # TODO filter may not work yet
        if where is None or where == "":
            filter = Filter()
        else:
            filter = Filter.parse_obj(json.loads(where))
        requests = [
            SearchRequest(
                vector=NamedVector(
                    name="",
                    vector=embedding,
                ),
                limit=k,
                with_payload=True,
                filter=filter,
            )
        ]
        if self.config.use_sparse_embeddings:
            sparse_embedding = self.get_sparse_embeddings([text])[0]
            requests.append(
                SearchRequest(
                    vector=NamedSparseVector(
                        name="text-sparse",
                        vector=sparse_embedding,
                    ),
                    limit=self.config.sparse_limit,
                    with_payload=True,
                    filter=filter,
                )
            )
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot search")
        search_result_lists: List[List[ScoredPoint]] = self.client.search_batch(
            collection_name=self.config.collection_name, requests=requests
        )

        search_result = [
            match for result in search_result_lists for match in result
        ]  # 2D list -> 1D list
        scores = [match.score for match in search_result if match is not None]
        docs = [
            self.config.document_class(**(match.payload))  # type: ignore
            for match in search_result
            if match is not None
        ]
        if len(docs) == 0:
            logger.warning(f"No matches found for {text}")
            return []
        doc_score_pairs = list(zip(docs, scores))
        max_score = max(ds[1] for ds in doc_score_pairs)
        if settings.debug:
            logger.info(f"Found {len(doc_score_pairs)} matches, max score: {max_score}")
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs
