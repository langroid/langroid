import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd

from langroid.embedding_models.base import EmbeddingModel, EmbeddingModelsConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document, EmbeddingFunction
from langroid.pydantic_v1 import BaseSettings
from langroid.utils.algorithms.graph import components, topological_sort
from langroid.utils.configuration import settings
from langroid.utils.object_registry import ObjectRegistry
from langroid.utils.output.printing import print_long_text
from langroid.utils.pandas_utils import stringify
from langroid.utils.pydantic_utils import flatten_dict

logger = logging.getLogger(__name__)


class VectorStoreConfig(BaseSettings):
    type: str = ""  # deprecated, keeping it for backward compatibility
    collection_name: str | None = "temp"
    replace_collection: bool = False  # replace collection if it already exists
    storage_path: str = ".qdrant/data"
    cloud: bool = False
    batch_size: int = 200
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig(
        model_type="openai",
    )
    embedding_model: Optional[EmbeddingModel] = None
    timeout: int = 60
    host: str = "127.0.0.1"
    port: int = 6333
    # used when parsing search results back as Document objects
    document_class: Type[Document] = Document
    metadata_class: Type[DocMetaData] = DocMetaData
    # compose_file: str = "langroid/vector_store/docker-compose-qdrant.yml"


class VectorStore(ABC):
    """
    Abstract base class for a vector store.
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        if config.embedding_model is None:
            self.embedding_model = EmbeddingModel.create(config.embedding)
        else:
            self.embedding_model = config.embedding_model
        self.embedding_fn: EmbeddingFunction = self.embedding_model.embedding_fn()

    @staticmethod
    def create(config: VectorStoreConfig) -> Optional["VectorStore"]:
        from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig
        from langroid.vector_store.lancedb import LanceDB, LanceDBConfig
        from langroid.vector_store.meilisearch import MeiliSearch, MeiliSearchConfig
        from langroid.vector_store.pineconedb import PineconeDB, PineconeDBConfig
        from langroid.vector_store.postgres import PostgresDB, PostgresDBConfig
        from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig
        from langroid.vector_store.weaviatedb import WeaviateDB, WeaviateDBConfig

        if isinstance(config, QdrantDBConfig):
            return QdrantDB(config)
        elif isinstance(config, ChromaDBConfig):
            return ChromaDB(config)
        elif isinstance(config, LanceDBConfig):
            return LanceDB(config)
        elif isinstance(config, MeiliSearchConfig):
            return MeiliSearch(config)
        elif isinstance(config, PostgresDBConfig):
            return PostgresDB(config)
        elif isinstance(config, WeaviateDBConfig):
            return WeaviateDB(config)
        elif isinstance(config, PineconeDBConfig):
            return PineconeDB(config)

        else:
            logger.warning(
                f"""
                Unknown vector store config: {config.__repr_name__()},
                so skipping vector store creation!
                If you intended to use a vector-store, please set a specific 
                vector-store in your script, typically in the `vecdb` field of a 
                `ChatAgentConfig`, otherwise set it to None.
                """
            )
            return None

    @property
    def embedding_dim(self) -> int:
        return len(self.embedding_fn(["test"])[0])

    @abstractmethod
    def clear_empty_collections(self) -> int:
        """Clear all empty collections in the vector store.
        Returns the number of collections deleted.
        """
        pass

    @abstractmethod
    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        """
        Clear all collections in the vector store.

        Args:
            really (bool, optional): Whether to really clear all collections.
                Defaults to False.
            prefix (str, optional): Prefix of collections to clear.
        Returns:
            int: Number of collections deleted.
        """
        pass

    @abstractmethod
    def list_collections(self, empty: bool = False) -> List[str]:
        """List all collections in the vector store
        (only non empty collections if empty=False).
        """
        pass

    def set_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Set the current collection to the given collection name.
        Args:
            collection_name (str): Name of the collection.
            replace (bool, optional): Whether to replace the collection if it
                already exists. Defaults to False.
        """

        self.config.collection_name = collection_name
        self.config.replace_collection = replace
        if replace:
            self.create_collection(collection_name, replace=True)

    @abstractmethod
    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """Create a collection with the given name.
        Args:
            collection_name (str): Name of the collection.
            replace (bool, optional): Whether to replace the
                collection if it already exists. Defaults to False.
        """
        pass

    @abstractmethod
    def add_documents(self, documents: Sequence[Document]) -> None:
        pass

    def compute_from_docs(self, docs: List[Document], calc: str) -> str:
        """Compute a result on a set of documents,
        using a dataframe calc string like `df.groupby('state')['income'].mean()`.
        """
        # convert each doc to a dict, using dotted paths for nested fields
        dicts = [flatten_dict(doc.dict(by_alias=True)) for doc in docs]
        df = pd.DataFrame(dicts)

        try:
            result = pd.eval(  # safer than eval but limited to single expression
                calc,
                engine="python",
                parser="pandas",
                local_dict={"df": df},
            )
        except Exception as e:
            # return error message so LLM can fix the calc string if needed
            err = f"""
            Error encountered in pandas eval: {str(e)}
            """
            if isinstance(e, KeyError) and "not in index" in str(e):
                # Pd.eval sometimes fails on a perfectly valid exprn like
                # df.loc[..., 'column'] with a KeyError.
                err += """
                Maybe try a different way, e.g. 
                instead of df.loc[..., 'column'], try df.loc[...]['column']
                """
            return err
        return stringify(result)

    def maybe_add_ids(self, documents: Sequence[Document]) -> None:
        """Add ids to metadata if absent, since some
        vecdbs don't like having blank ids."""
        for d in documents:
            if d.metadata.id in [None, ""]:
                d.metadata.id = ObjectRegistry.new_id()

    @abstractmethod
    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Find k most similar texts to the given text, in terms of vector distance metric
        (e.g., cosine similarity).

        Args:
            text (str): The text to find similar texts for.
            k (int, optional): Number of similar texts to retrieve. Defaults to 1.
            where (Optional[str], optional): Where clause to filter the search.

        Returns:
            List[Tuple[Document,float]]: List of (Document, score) tuples.

        """
        pass

    def add_context_window(
        self, docs_scores: List[Tuple[Document, float]], neighbors: int = 0
    ) -> List[Tuple[Document, float]]:
        """
        In each doc's metadata, there may be a window_ids field indicating
        the ids of the chunks around the current chunk.
        These window_ids may overlap, so we
        - coalesce each overlapping groups into a single window (maintaining ordering),
        - create a new document for each part, preserving metadata,

        We may have stored a longer set of window_ids than we need during chunking.
        Now, we just want `neighbors` on each side of the center of the window_ids list.

        Args:
            docs_scores (List[Tuple[Document, float]]): List of pairs of documents
                to add context windows to together with their match scores.
            neighbors (int, optional): Number of neighbors on "each side" of match to
                retrieve. Defaults to 0.
                "Each side" here means before and after the match,
                in the original text.

        Returns:
            List[Tuple[Document, float]]: List of (Document, score) tuples.
        """
        # We return a larger context around each match, i.e.
        # a window of `neighbors` on each side of the match.
        docs = [d for d, s in docs_scores]
        scores = [s for d, s in docs_scores]
        if neighbors == 0:
            return docs_scores
        doc_chunks = [d for d in docs if d.metadata.is_chunk]
        if len(doc_chunks) == 0:
            return docs_scores
        window_ids_list = []
        id2metadata = {}
        # id -> highest score of a doc it appears in
        id2max_score: Dict[int | str, float] = {}
        for i, d in enumerate(docs):
            window_ids = d.metadata.window_ids
            if len(window_ids) == 0:
                window_ids = [d.id()]
            id2metadata.update({id: d.metadata for id in window_ids})

            id2max_score.update(
                {id: max(id2max_score.get(id, 0), scores[i]) for id in window_ids}
            )
            n = len(window_ids)
            chunk_idx = window_ids.index(d.id())
            neighbor_ids = window_ids[
                max(0, chunk_idx - neighbors) : min(n, chunk_idx + neighbors + 1)
            ]
            window_ids_list += [neighbor_ids]

        # window_ids could be from different docs,
        # and they may overlap, so we coalesce overlapping groups into
        # separate windows.
        window_ids_list = self.remove_overlaps(window_ids_list)
        final_docs = []
        final_scores = []
        for w in window_ids_list:
            metadata = copy.deepcopy(id2metadata[w[0]])
            metadata.window_ids = w
            document = Document(
                content="".join([d.content for d in self.get_documents_by_ids(w)]),
                metadata=metadata,
            )
            # make a fresh id since content is in general different
            document.metadata.id = ObjectRegistry.new_id()
            final_docs += [document]
            final_scores += [max(id2max_score[id] for id in w)]
        return list(zip(final_docs, final_scores))

    @staticmethod
    def remove_overlaps(windows: List[List[str]]) -> List[List[str]]:
        """
        Given a collection of windows, where each window is a sequence of ids,
        identify groups of overlapping windows, and for each overlapping group,
        order the chunk-ids using topological sort so they appear in the original
        order in the text.

        Args:
            windows (List[int|str]): List of windows, where each window is a
                sequence of ids.

        Returns:
            List[int|str]: List of windows, where each window is a sequence of ids,
                and no two windows overlap.
        """
        ids = set(id for w in windows for id in w)
        # id -> {win -> # pos}
        id2win2pos: Dict[str, Dict[int, int]] = {id: {} for id in ids}

        for i, w in enumerate(windows):
            for j, id in enumerate(w):
                id2win2pos[id][i] = j

        n = len(windows)
        # relation between windows:
        order = np.zeros((n, n), dtype=np.int8)
        for i, w in enumerate(windows):
            for j, x in enumerate(windows):
                if i == j:
                    continue
                if len(set(w).intersection(x)) == 0:
                    continue
                id = list(set(w).intersection(x))[0]  # any common id
                if id2win2pos[id][i] > id2win2pos[id][j]:
                    order[i, j] = -1  # win i is before win j
                else:
                    order[i, j] = 1  # win i is after win j

        # find groups of windows that overlap, like connected components in a graph
        groups = components(np.abs(order))

        # order the chunk-ids in each group using topological sort
        new_windows = []
        for g in groups:
            # find total ordering among windows in group based on order matrix
            # (this is a topological sort)
            _g = np.array(g)
            order_matrix = order[_g][:, _g]
            ordered_window_indices = topological_sort(order_matrix)
            ordered_window_ids = [windows[i] for i in _g[ordered_window_indices]]
            flattened = [id for w in ordered_window_ids for id in w]
            flattened_deduped = list(dict.fromkeys(flattened))
            # Note we are not going to split these, and instead we'll return
            # larger windows from concatenating the connected groups.
            # This ensures context is retained for LLM q/a
            new_windows += [flattened_deduped]

        return new_windows

    @abstractmethod
    def get_all_documents(self, where: str = "") -> List[Document]:
        """
        Get all documents in the current collection, possibly filtered by `where`.
        """
        pass

    @abstractmethod
    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Get documents by their ids.
        Args:
            ids (List[str]): List of document ids.

        Returns:
            List[Document]: List of documents
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        pass

    def show_if_debug(self, doc_score_pairs: List[Tuple[Document, float]]) -> None:
        if settings.debug:
            for i, (d, s) in enumerate(doc_score_pairs):
                print_long_text("red", "italic red", f"\nMATCH-{i}\n", d.content)
