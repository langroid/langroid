import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic_settings import SettingsConfigDict

from langroid.embedding_models.base import EmbeddingModelsConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.exceptions import LangroidImportError
from langroid.mytypes import Document
from langroid.utils.configuration import settings
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)

_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STATIC_FIELDS = {"id", "vector", "content", "metadata"}
_OUTPUT_FIELDS = ["id", "content", "metadata", "doc_extra"]


class MilvusDBConfig(VectorStoreConfig):
    model_config = SettingsConfigDict(env_prefix="MILVUS_")

    collection_name: str | None = "temp"
    uri: str = ""
    token: str | None = None
    db_name: str | None = None
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    metric_type: str = "COSINE"
    consistency_level: str = "Strong"
    id_field_max_length: int = 512
    text_field_max_length: int = 65535


class MilvusDB(VectorStore):
    def __init__(self, config: Optional[MilvusDBConfig] = None):
        if config is None:
            config = MilvusDBConfig()
        super().__init__(config)
        self.config: MilvusDBConfig = config
        load_dotenv()
        self.config.uri = self.config.uri or os.getenv("MILVUS_URI", "./milvus.db")
        self.config.token = self.config.token or os.getenv("MILVUS_TOKEN")
        self.config.db_name = self.config.db_name or os.getenv("MILVUS_DB_NAME")
        self.config.metric_type = self.config.metric_type.upper()

        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise LangroidImportError("pymilvus", "milvus") from exc
        if (
            self._is_local_lite_uri(self.config.uri)
            and self._milvus_lite_version() is None
        ):
            raise ValueError(
                "Milvus Lite is unavailable on this platform; set MILVUS_URI "
                "to a Milvus server or Zilliz Cloud endpoint"
            )

        self._milvus_lite_3_0 = self._uses_milvus_lite_3_0(self.config.uri)

        client_kwargs: Dict[str, Any] = {
            "uri": self.config.uri,
            "timeout": self.config.timeout,
        }
        if self.config.token:
            client_kwargs["token"] = self.config.token
        if self.config.db_name:
            client_kwargs["db_name"] = self.config.db_name
        self.client = MilvusClient(**client_kwargs)

        if self.config.collection_name is not None:
            self.create_collection(
                self.config.collection_name,
                replace=self.config.replace_collection,
            )

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()

    def __enter__(self) -> "MilvusDB":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def clear_empty_collections(self) -> int:
        n_deletes = 0
        for collection_name in self.list_collections(empty=True):
            if self._row_count(collection_name) == 0:
                self.delete_collection(collection_name)
                n_deletes += 1
        return n_deletes

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        if not really:
            logger.warning("Not deleting all collections, set really=True to confirm")
            return 0
        collection_names = [
            name
            for name in self.list_collections(empty=True)
            if name.startswith(prefix)
        ]
        if len(collection_names) == 0:
            logger.warning(f"No collections found with prefix {prefix}")
            return 0
        for collection_name in collection_names:
            self.delete_collection(collection_name)
        logger.warning(f"Deleted {len(collection_names)} collections.")
        return len(collection_names)

    def list_collections(self, empty: bool = False) -> List[str]:
        collection_names = list(self.client.list_collections())
        if empty:
            return collection_names
        return [
            collection_name
            for collection_name in collection_names
            if self._row_count(collection_name) != 0
        ]

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        self.config.collection_name = collection_name
        if self.client.has_collection(collection_name=collection_name):
            if replace:
                self.client.drop_collection(collection_name=collection_name)
            else:
                self._validate_collection_schema(collection_name)
                self.client.load_collection(collection_name=collection_name)
                return

        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=self.config.id_field_max_length,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_dim,
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=self.config.text_field_max_length,
        )
        schema.add_field(field_name="metadata", datatype=DataType.JSON)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=self.config.metric_type,
        )
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self.config.consistency_level,
        )

    def set_collection(self, collection_name: str, replace: bool = False) -> None:
        """Set the active collection and load it when it already exists.

        Args:
            collection_name: Name of the collection.
            replace: Whether to replace an existing collection.
        """
        super().set_collection(collection_name, replace)
        if not replace and self.client.has_collection(collection_name=collection_name):
            self.create_collection(collection_name, replace=False)

    def add_documents(self, documents: Sequence[Document]) -> None:
        if len(documents) == 0:
            return
        super().maybe_add_ids(documents)
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot ingest docs")
        if not self.client.has_collection(collection_name=self.config.collection_name):
            self.create_collection(self.config.collection_name, replace=True)

        for doc in documents:
            self._validate_document(doc)
        embedding_vecs = self.embedding_fn([doc.content for doc in documents])
        if len(embedding_vecs) != len(documents):
            raise ValueError(
                f"Embedding count {len(embedding_vecs)} does not match "
                f"document count {len(documents)}"
            )
        batch_size = self.config.batch_size
        for i in range(0, len(documents), batch_size):
            rows = [
                self._document_to_record(doc, embedding)
                for doc, embedding in zip(
                    documents[i : i + batch_size],
                    embedding_vecs[i : i + batch_size],
                )
            ]
            self.client.upsert(collection_name=self.config.collection_name, data=rows)

    def get_all_documents(self, where: str = "") -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        if not self.client.has_collection(collection_name=self.config.collection_name):
            return []

        docs: List[Document] = []
        iterator = self.client.query_iterator(
            collection_name=self.config.collection_name,
            batch_size=self.config.batch_size,
            filter=self._where_to_filter(where),
            output_fields=_OUTPUT_FIELDS,
        )
        try:
            while True:
                records = iterator.next()
                if len(records) == 0:
                    break
                docs.extend(self._records_to_docs(records))
        finally:
            iterator.close()
        return docs

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        if not self.client.has_collection(collection_name=self.config.collection_name):
            return []
        if len(ids) == 0:
            return []
        records = self.client.get(
            collection_name=self.config.collection_name,
            ids=[str(doc_id) for doc_id in ids],
            output_fields=_OUTPUT_FIELDS,
        )
        id_to_record = {str(record["id"]): record for record in records}
        ordered_records = [
            id_to_record[str(doc_id)] for doc_id in ids if str(doc_id) in id_to_record
        ]
        return self._records_to_docs(ordered_records)

    def delete_collection(self, collection_name: str) -> None:
        if self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot search")
        if not self.client.has_collection(collection_name=self.config.collection_name):
            return []
        row_count = self._row_count(self.config.collection_name)
        if row_count == 0:
            return []
        limit = k if row_count < 0 else min(k, row_count)

        embedding = self.embedding_fn([text])[0]
        results = self.client.search(
            collection_name=self.config.collection_name,
            data=[embedding],
            anns_field="vector",
            filter=self._where_to_filter(where),
            limit=limit,
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": self.config.metric_type},
        )
        matches = results[0] if len(results) > 0 else []
        docs = self._records_to_docs([match["entity"] for match in matches])
        scores = [
            self._score_from_distance(float(match["distance"])) for match in matches
        ]
        doc_score_pairs = list(zip(docs, scores))
        if len(doc_score_pairs) == 0:
            logger.warning(f"No matches found for {text}")
            return []
        if settings.debug:
            logger.info(
                f"Found {len(doc_score_pairs)} matches, max score: {max(scores)}"
            )
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs

    def _row_count(self, collection_name: str) -> int:
        """Return the row count, or -1 when collection stats are unavailable."""
        try:
            stats = self.client.get_collection_stats(collection_name=collection_name)
            if "row_count" not in stats:
                return -1
            return int(stats["row_count"])
        except Exception:
            logger.warning(f"Error getting collection stats for {collection_name}")
            return -1

    def _validate_collection_schema(self, collection_name: str) -> None:
        from pymilvus import DataType

        description = self.client.describe_collection(collection_name=collection_name)
        fields = {field["name"]: field for field in description.get("fields", [])}
        missing_fields = _STATIC_FIELDS.difference(fields)
        if len(missing_fields) > 0:
            raise ValueError(
                f"Milvus collection {collection_name} is missing fields: "
                f"{sorted(missing_fields)}"
            )
        if fields["id"]["type"] != DataType.VARCHAR:
            raise ValueError(f"Milvus collection {collection_name} has invalid id type")
        if not fields["id"].get("is_primary", False):
            raise ValueError(
                f"Milvus collection {collection_name} must use id as primary key"
            )
        if fields["vector"]["type"] != DataType.FLOAT_VECTOR:
            raise ValueError(
                f"Milvus collection {collection_name} has invalid vector type"
            )
        dim = int(fields["vector"].get("params", {}).get("dim", 0))
        if dim != self.embedding_dim:
            raise ValueError(
                f"Milvus collection {collection_name} vector dim {dim} does not "
                f"match embedding dim {self.embedding_dim}"
            )
        if fields["content"]["type"] != DataType.VARCHAR:
            raise ValueError(
                f"Milvus collection {collection_name} has invalid content type"
            )
        if fields["metadata"]["type"] != DataType.JSON:
            raise ValueError(
                f"Milvus collection {collection_name} has invalid metadata type"
            )

    def _document_to_record(
        self, doc: Document, embedding: List[float]
    ) -> Dict[str, Any]:
        self._validate_document(doc)
        doc_id = str(doc.id())
        metadata = doc.metadata.model_dump()
        metadata["id"] = doc_id
        extra = {
            key: value
            for key, value in doc.model_dump().items()
            if key not in ("content", "metadata")
        }
        return {
            "id": doc_id,
            "vector": embedding,
            "content": doc.content,
            "metadata": metadata,
            "doc_extra": extra,
            **self._dynamic_metadata_fields(metadata),
        }

    def _validate_document(self, doc: Document) -> None:
        """Validate document field lengths for Milvus storage.

        Args:
            doc: Document to validate.
        """
        doc_id = str(doc.id())
        if len(doc_id.encode("utf-8")) > self.config.id_field_max_length:
            raise ValueError(
                f"Document id exceeds Milvus max length "
                f"{self.config.id_field_max_length}: {doc_id}"
            )
        if len(doc.content.encode("utf-8")) > self.config.text_field_max_length:
            raise ValueError(
                "Document content exceeds Milvus VARCHAR max length "
                f"{self.config.text_field_max_length}"
            )

    @staticmethod
    def _dynamic_metadata_fields(metadata: Dict[str, Any]) -> Dict[str, Any]:
        dynamic_fields: Dict[str, Any] = {}
        for key, value in metadata.items():
            if key == "doc_extra":
                continue
            if key in _STATIC_FIELDS or _FIELD_NAME_RE.match(key) is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                dynamic_fields[key] = value
        return dynamic_fields

    def _records_to_docs(self, records: Sequence[Dict[str, Any]]) -> List[Document]:
        docs = []
        for record in records:
            metadata = dict(record.get("metadata") or {})
            metadata["id"] = str(record.get("id", metadata.get("id", "")))
            extra = record.get("doc_extra") or {}
            if not isinstance(extra, dict):
                extra = {}
            docs.append(
                self.config.document_class(
                    content=record["content"],
                    metadata=self.config.metadata_class(**metadata),
                    **extra,
                )
            )
        return docs

    def _score_from_distance(self, distance: float) -> float:
        metric_type = self.config.metric_type.upper()
        if metric_type == "COSINE":
            return 1.0 - distance if self._milvus_lite_3_0 else distance
        if metric_type == "L2":
            l2_distance = (
                distance if self._milvus_lite_3_0 else math.sqrt(max(distance, 0.0))
            )
            return -l2_distance
        return distance

    @staticmethod
    def _uses_milvus_lite_3_0(uri: str) -> bool:
        if not MilvusDB._is_local_lite_uri(uri):
            return False
        milvus_lite_version = MilvusDB._milvus_lite_version()
        return milvus_lite_version in {"3.0", "3.0.0"}

    @staticmethod
    def _is_local_lite_uri(uri: str) -> bool:
        server_schemes = {"unix", "http", "https", "tcp", "grpc"}
        return urlparse(uri).scheme.lower() not in server_schemes

    @staticmethod
    def _milvus_lite_version() -> Optional[str]:
        try:
            from milvus_lite import __version__ as milvus_lite_version
        except ImportError:
            return None
        # Lite 3.0 reports COSINE as a distance and L2 as Euclidean distance.
        # Server, cloud, and newer Lite releases report COSINE similarity and
        # squared L2 distance. See https://github.com/milvus-io/milvus-lite/issues/343.
        return str(milvus_lite_version)

    @classmethod
    def _where_to_filter(cls, where: Optional[str]) -> str:
        if where is None or where.strip() == "":
            return ""
        where = where.strip()
        try:
            parsed = json.loads(where)
        except json.JSONDecodeError:
            return where
        if not isinstance(parsed, dict):
            raise ValueError("Milvus JSON filters must be objects")
        expressions = [
            cls._filter_expression(key, value) for key, value in parsed.items()
        ]
        return " and ".join(expressions)

    @classmethod
    def _filter_expression(cls, field_name: str, value: Any) -> str:
        field = f"metadata[{json.dumps(field_name, ensure_ascii=False)}]"
        if isinstance(value, dict):
            if set(value.keys()) == {"$eq"}:
                return f"{field} == {cls._format_filter_value(value['$eq'])}"
            if set(value.keys()) == {"$in"}:
                return cls._in_filter_expression(field_name, value["$in"])
            raise ValueError(f"Unsupported Milvus filter operator for {field_name}")
        if isinstance(value, list):
            return cls._in_filter_expression(field_name, value)
        return f"{field} == {cls._format_filter_value(value)}"

    @classmethod
    def _in_filter_expression(cls, field_name: str, values: Any) -> str:
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"Milvus filter for {field_name} requires a non-empty list"
            )
        field = f"metadata[{json.dumps(field_name, ensure_ascii=False)}]"
        comparisons = " or ".join(
            f"{field} == {cls._format_filter_value(value)}" for value in values
        )
        return f"({comparisons})"

    @staticmethod
    def _format_filter_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Milvus filters do not support NaN or infinity")
            return repr(value)
        raise ValueError(f"Unsupported Milvus filter value: {value!r}")
