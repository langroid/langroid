from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import pandas as pd
from dotenv import load_dotenv

from langroid.pydantic_v1 import BaseModel, ValidationError, create_model

if TYPE_CHECKING:
    from lancedb.query import LanceVectorQueryBuilder

from langroid.embedding_models.base import (
    EmbeddingModel,
    EmbeddingModelsConfig,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.exceptions import LangroidImportError
from langroid.mytypes import Document, EmbeddingFunction
from langroid.utils.configuration import settings
from langroid.utils.pydantic_utils import (
    dataframe_to_document_model,
    dataframe_to_documents,
)
from langroid.vector_store.base import VectorStore, VectorStoreConfig

try:
    import lancedb
    from lancedb.pydantic import LanceModel, Vector

    has_lancedb = True
except ImportError:
    has_lancedb = False

logger = logging.getLogger(__name__)


class LanceDBConfig(VectorStoreConfig):
    cloud: bool = False
    collection_name: str | None = "temp"
    storage_path: str = ".lancedb/data"
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    distance: str = "cosine"


class LanceDB(VectorStore):
    def __init__(self, config: LanceDBConfig = LanceDBConfig()):
        super().__init__(config)
        if not has_lancedb:
            raise LangroidImportError("lancedb", "lancedb")

        self.config: LanceDBConfig = config
        emb_model = EmbeddingModel.create(config.embedding)
        self.embedding_fn: EmbeddingFunction = emb_model.embedding_fn()
        self.embedding_dim = emb_model.embedding_dims
        self.host = config.host
        self.port = config.port
        self.is_from_dataframe = False  # were docs ingested from a dataframe?
        self.df_metadata_columns: List[str] = []  # metadata columns from dataframe

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
        colls = self.client.table_names(limit=None)
        if len(colls) == 0:
            return []
        if empty:  # include empty tbls
            return colls  # type: ignore
        counts = [self.client.open_table(coll).head(1).shape[0] for coll in colls]
        return [coll for coll, count in zip(colls, counts) if count > 0]

    def _create_lance_schema(self, doc_cls: Type[Document]) -> Type[BaseModel]:
        """
        NOTE: NOT USED, but leaving it here as it may be useful.

        Create a subclass of LanceModel with fields:
         - id (str)
         - Vector field that has dims equal to
            the embedding dimension of the embedding model, and a data field of type
            DocClass.
         - other fields from doc_cls

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

        if not has_lancedb:
            raise LangroidImportError("lancedb", "lancedb")

        n = self.embedding_dim

        # Prepare fields for the new model
        fields = {"id": (str, ...), "vector": (Vector(n), ...)}

        sorted_fields = dict(
            sorted(doc_cls.__fields__.items(), key=lambda item: item[0])
        )
        # Add both statically and dynamically defined fields from doc_cls
        for field_name, field in sorted_fields.items():
            fields[field_name] = (field.outer_type_, field.default)

        # Create the new model with dynamic fields
        NewModel = create_model(
            "NewModel", __base__=LanceModel, **fields
        )  # type: ignore
        return NewModel  # type: ignore

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        self.config.replace_collection = replace

    def add_documents(self, documents: Sequence[Document]) -> None:
        super().maybe_add_ids(documents)
        colls = self.list_collections(empty=True)
        if len(documents) == 0:
            return
        embedding_vecs = self.embedding_fn([doc.content for doc in documents])
        coll_name = self.config.collection_name
        if coll_name is None:
            raise ValueError("No collection name set, cannot ingest docs")
        # self._maybe_set_doc_class_schema(documents[0])
        table_exists = False
        if (
            coll_name in colls
            and self.client.open_table(coll_name).head(1).shape[0] > 0
        ):
            # collection exists and  is not empty:
            # if replace_collection is True, we'll overwrite the existing collection,
            # else we'll append to it.
            if self.config.replace_collection:
                self.client.drop_table(coll_name)
            else:
                table_exists = True

        ids = [str(d.id()) for d in documents]
        # don't insert all at once, batch in chunks of b,
        # else we get an API error
        b = self.config.batch_size

        def make_batches() -> Generator[List[Dict[str, Any]], None, None]:
            for i in range(0, len(ids), b):
                batch = [
                    dict(
                        id=ids[i + j],
                        vector=embedding_vecs[i + j],
                        **doc.dict(),
                    )
                    for j, doc in enumerate(documents[i : i + b])
                ]
                yield batch

        try:
            if table_exists:
                tbl = self.client.open_table(coll_name)
                tbl.add(make_batches())
            else:
                batch_gen = make_batches()
                batch = next(batch_gen)
                # use first batch to create table...
                tbl = self.client.create_table(
                    coll_name,
                    data=batch,
                    mode="create",
                )
                # ... and add the rest
                tbl.add(batch_gen)
        except Exception as e:
            logger.error(
                f"""
                Error adding documents to LanceDB: {e}
                POSSIBLE REMEDY: Delete the LancdDB storage directory
                {self.config.storage_path} and try again.
                """
            )

    def add_dataframe(
        self,
        df: pd.DataFrame,
        content: str = "content",
        metadata: List[str] = [],
    ) -> None:
        """
        Add a dataframe to the collection.
        Args:
            df (pd.DataFrame): A dataframe
            content (str): The name of the column in the dataframe that contains the
                text content to be embedded using the embedding model.
            metadata (List[str]): A list of column names in the dataframe that contain
                metadata to be stored in the database. Defaults to [].
        """
        self.is_from_dataframe = True
        actual_metadata = metadata.copy()
        self.df_metadata_columns = actual_metadata  # could be updated below
        # get content column
        content_values = df[content].values.tolist()
        embedding_vecs = self.embedding_fn(content_values)

        # add vector column
        df["vector"] = embedding_vecs
        if content != "content":
            # rename content column to "content", leave existing column intact
            df = df.rename(columns={content: "content"}, inplace=False)

        if "id" not in df.columns:
            docs = dataframe_to_documents(df, content="content", metadata=metadata)
            ids = [str(d.id()) for d in docs]
            df["id"] = ids

        if "id" not in actual_metadata:
            actual_metadata += ["id"]

        colls = self.list_collections(empty=True)
        coll_name = self.config.collection_name
        if (
            coll_name not in colls
            or self.client.open_table(coll_name).head(1).shape[0] == 0
        ):
            # collection either doesn't exist or is empty, so replace it
            # and set new schema from df
            self.client.create_table(
                self.config.collection_name,
                data=df,
                mode="overwrite",
            )
            doc_cls = dataframe_to_document_model(
                df,
                content=content,
                metadata=actual_metadata,
                exclude=["vector"],
            )
            self.config.document_class = doc_cls  # type: ignore
        else:
            # collection exists and is not empty, so append to it
            tbl = self.client.open_table(self.config.collection_name)
            tbl.add(df)

    def delete_collection(self, collection_name: str) -> None:
        self.client.drop_table(collection_name, ignore_missing=True)

    def _lance_result_to_docs(
        self, result: "LanceVectorQueryBuilder"
    ) -> List[Document]:
        if self.is_from_dataframe:
            df = result.to_pandas()
            return dataframe_to_documents(
                df,
                content="content",
                metadata=self.df_metadata_columns,
                doc_cls=self.config.document_class,
            )
        else:
            records = result.to_arrow().to_pylist()
            return self._records_to_docs(records)

    def _records_to_docs(self, records: List[Dict[str, Any]]) -> List[Document]:
        try:
            docs = [self.config.document_class(**rec) for rec in records]
        except ValidationError as e:
            raise ValueError(
                f"""
            Error validating LanceDB result: {e}
            HINT: This could happen when you're re-using an 
            existing LanceDB store with a different schema.
            Try deleting your local lancedb storage at `{self.config.storage_path}`
            re-ingesting your documents and/or replacing the collections.
            """
            )
        return docs

    def get_all_documents(self, where: str = "") -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        tbl = self.client.open_table(self.config.collection_name)
        pre_result = tbl.search(None).where(where or None).limit(None)
        return self._lance_result_to_docs(pre_result)

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        _ids = [str(id) for id in ids]
        tbl = self.client.open_table(self.config.collection_name)
        docs = []
        for _id in _ids:
            results = self._lance_result_to_docs(tbl.search().where(f"id == '{_id}'"))
            if len(results) > 0:
                docs.append(results[0])
        return docs

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_fn([text])[0]
        tbl = self.client.open_table(self.config.collection_name)
        result = (
            tbl.search(embedding)
            .metric(self.config.distance)
            .where(where, prefilter=True)
            .limit(k)
        )
        docs = self._lance_result_to_docs(result)
        # note _distance is 1 - cosine
        if self.is_from_dataframe:
            scores = [
                1 - rec["_distance"] for rec in result.to_pandas().to_dict("records")
            ]
        else:
            scores = [1 - rec["_distance"] for rec in result.to_arrow().to_pylist()]
        if len(docs) == 0:
            logger.warning(f"No matches found for {text}")
            return []
        if settings.debug:
            logger.info(f"Found {len(docs)} matches, max score: {max(scores)}")
        doc_score_pairs = list(zip(docs, scores))
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs
