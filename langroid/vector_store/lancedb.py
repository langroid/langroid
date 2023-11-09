import logging
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Type

import lancedb
from dotenv import load_dotenv
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel, create_model

from langroid.embedding_models.base import (
    EmbeddingModel,
    EmbeddingModelsConfig,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import Document, EmbeddingFunction
from langroid.utils.configuration import settings
from langroid.utils.pydantic_utils import (
    flatten_pydantic_instance,
    flatten_pydantic_model,
    nested_dict_from_flat,
)
from langroid.vector_store.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class LanceDBConfig(VectorStoreConfig):
    cloud: bool = False
    collection_name: str | None = None
    storage_path: str = ".lancedb/data"
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    distance: str = "cosine"
    document_class: Type[Document] = Document


class LanceDB(VectorStore):
    def __init__(self, config: LanceDBConfig):
        super().__init__(config)
        self.config: LanceDBConfig = config
        emb_model = EmbeddingModel.create(config.embedding)
        self.embedding_fn: EmbeddingFunction = emb_model.embedding_fn()
        self.embedding_dim = emb_model.embedding_dims
        self.host = config.host
        self.port = config.port
        self.schema = self._create_lance_schema(self.config.document_class)
        self.flat_schema = self._create_flat_lance_schema(self.config.document_class)
        load_dotenv()
        if self.config.cloud:
            logger.warning(
                "LanceDB Cloud is not available yet. Switching to local storage."
            )
            config.cloud = False
        else:
            try:
                self.client = lancedb.connect(
                    uri=config.storage_path,
                )
            except Exception as e:
                new_storage_path = config.storage_path + ".new"
                logger.warning(
                    f"""
                    Error connecting to local LanceDB at {config.storage_path}:
                    {e}
                    Switching to {new_storage_path}
                    """
                )
                self.client = lancedb.connect(
                    uri=new_storage_path,
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
            nr = self.client.open_table(name).head(1).shape[0]
            if nr == 0:
                n_deletes += 1
                self.client.drop_table(name)
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
            nr = self.client.open_table(name).head(1).shape[0]
            n_empty_deletes += nr == 0
            n_non_empty_deletes += nr > 0
            self.client.drop_table(name)
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
        colls = self.client.table_names()
        if len(colls) == 0:
            return []
        if empty:  # include empty tbls
            return colls  # type: ignore
        counts = [self.client.open_table(coll).head(1).shape[0] for coll in colls]
        return [coll for coll, count in zip(colls, counts) if count > 0]

    def _create_lance_schema(self, doc_cls: Type[Document]) -> Type[BaseModel]:
        """
        Create a subclass of LanceModel with fields:
         - id (str)
         - Vector field that has dims equal to
            the embedding dimension of the embedding model, and a data field of type
            DocClass.
         - payload of type `doc_cls`

        Args:
            doc_cls (Type[Document]): A Pydantic model which should be a subclass of
                Document, to be used as the type for the data field.

        Returns:
            Type[BaseModel]: A new Pydantic model subclassing from LanceModel.

        Raises:
            ValueError: If `n` is not a non-negative integer or if `DocClass` is not a
                subclass of Document.
        """
        if not issubclass(doc_cls, Document):
            raise ValueError("DocClass must be a subclass of Document")

        n = self.embedding_dim

        NewModel = create_model(
            "NewModel",
            __base__=LanceModel,
            id=(str, ...),
            vector=(Vector(n), ...),
            payload=(doc_cls, ...),
        )
        return NewModel  # type: ignore

    def _create_flat_lance_schema(self, doc_cls: Type[Document]) -> Type[BaseModel]:
        """
        Flat version of the lance_schema, as nested Pydantic schemas are not yet
        supported by LanceDB.
        """
        lance_model = self._create_lance_schema(doc_cls)
        FlatModel = flatten_pydantic_model(lance_model, base_model=LanceModel)
        return FlatModel

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Create a collection with the given name, optionally replacing an existing
            collection if `replace` is True.
        Args:
            collection_name (str): Name of the collection to create.
            replace (bool): Whether to replace an existing collection
                with the same name. Defaults to False.
        """
        self.config.collection_name = collection_name
        collections = self.list_collections()
        if collection_name in collections:
            coll = self.client.open_table(collection_name)
            if coll.head().shape[0] > 0:
                logger.warning(f"Non-empty Collection {collection_name} already exists")
                if not replace:
                    logger.warning("Not replacing collection")
                    return
                else:
                    logger.warning("Recreating fresh collection")
        tbl = self.client.create_table(
            collection_name, schema=self.flat_schema, mode="overwrite"
        )
        if settings.debug:
            level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)
            logger.info(tbl.schema)
            logger.setLevel(level)

    def add_documents(self, documents: Sequence[Document]) -> None:
        colls = self.list_collections(empty=True)
        if len(documents) == 0:
            return
        embedding_vecs = self.embedding_fn([doc.content for doc in documents])
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot ingest docs")
        if self.config.collection_name not in colls:
            self.create_collection(self.config.collection_name, replace=True)
        ids = [str(d.id()) for d in documents]
        # don't insert all at once, batch in chunks of b,
        # else we get an API error
        b = self.config.batch_size

        def make_batches() -> Generator[List[Dict[str, Any]], None, None]:
            for i in range(0, len(ids), b):
                yield [
                    flatten_pydantic_instance(
                        self.schema(
                            id=ids[i],
                            vector=embedding_vecs[i],
                            payload=doc,
                        )
                    )
                    for i, doc in enumerate(documents[i : i + b])
                ]

        tbl = self.client.open_table(self.config.collection_name)
        tbl.add(make_batches())

    def delete_collection(self, collection_name: str) -> None:
        self.client.drop_table(collection_name)

    def get_all_documents(self) -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        tbl = self.client.open_table(self.config.collection_name)
        records = tbl.search(None).to_arrow().to_pylist()
        docs = [
            self.config.document_class(
                **(nested_dict_from_flat(rec, sub_dict="payload"))
            )
            for rec in records
        ]
        return docs

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        _ids = [str(id) for id in ids]
        tbl = self.client.open_table(self.config.collection_name)
        records = [
            tbl.search().where(f"id == '{_id}'").to_arrow().to_pylist()[0]
            for _id in _ids
        ]
        doc_cls = self.config.document_class
        docs = [
            doc_cls(**(nested_dict_from_flat(rec, sub_dict="payload")))
            for rec in records
        ]
        return docs

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_fn([text])[0]
        tbl = self.client.open_table(self.config.collection_name)
        records = (
            tbl.search(embedding)
            .metric(self.config.distance)
            .where(where)
            .limit(k)
            .to_arrow()
            .to_pylist()
        )

        # note _distance is 1 - cosine
        scores = [1 - rec["_distance"] for rec in records]
        docs = [
            self.config.document_class(
                **(nested_dict_from_flat(rec, sub_dict="payload"))
            )
            for rec in records
        ]
        if len(docs) == 0:
            logger.warning(f"No matches found for {text}")
            return []
        if settings.debug:
            logger.info(f"Found {len(docs)} matches, max score: {max(scores)}")
        doc_score_pairs = list(zip(docs, scores))
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs
