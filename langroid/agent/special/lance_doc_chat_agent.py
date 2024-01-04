"""
LanceDocChatAgent is a subclass of DocChatAgent that uses LanceDB as a vector store:
- Uses the DocChatAgentConfig.filter variable
    (a sql string) in the `where` clause to do filtered vector search.
- Overrides the get_similar_chunks_bm25() to use LanceDB FTS (Full Text Search).

For usage see:
 - `tests/main/test_lance_doc_chat_agent.py`.
 - example script `examples/docqa/lance_rag.py`.

"""
import json
import logging
from typing import List, Tuple

import pandas as pd

from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.special.lance_rag.lance_tools import QueryPlanTool
from langroid.mytypes import Document
from langroid.parsing.table_loader import describe_dataframe
from langroid.utils.constants import DONE, NO_ANSWER
from langroid.utils.pydantic_utils import (
    clean_schema,
    dataframe_to_documents,
)
from langroid.vector_store.lancedb import LanceDB

logger = logging.getLogger(__name__)


class LanceDocChatAgent(DocChatAgent):
    vecdb: LanceDB

    def __init__(self, cfg: DocChatAgentConfig):
        super().__init__(cfg)
        self.config: DocChatAgentConfig = cfg
        self.enable_message(QueryPlanTool, use=False, handle=True)

    def _get_clean_vecdb_schema(self) -> str:
        if self.from_dataframe:
            return self.df_description
        schema_dict = clean_schema(
            self.vecdb.schema,
            excludes=["id", "vector"],
        )
        return json.dumps(schema_dict, indent=4)

    def query_plan(self, msg: QueryPlanTool) -> str:
        """
        Handle the LLM's use of the FilterTool.
        Temporarily set the config filter and invoke the DocChatAgent.llm_response()
        """
        # create document-subset based on this filter
        try:
            self.setup_documents(filter=msg.filter or None)
        except Exception as e:
            logger.error(f"Error setting up documents: {e}")
            # say DONE with err msg so it goes back to LanceFilterAgent
            return f"{DONE} Possible Filter Error:\n {e}"
        # update the filter so it is used in the DocChatAgent
        self.config.filter = msg.filter or None
        if msg.dataframe_calc:
            # we just get relevant docs then do the calculation
            # TODO if calc causes err, it is captured in result,
            # and LLM can correct the calc based on the err,
            # and this will cause retrieval all over again,
            # which may be wasteful if only the calc part is wrong.
            # The calc step can later be done with a separate Agent/Tool.
            _, docs = self.get_relevant_extracts(msg.query)
            if len(docs) == 0:
                return DONE + " " + NO_ANSWER
            result = self.vecdb.compute_from_docs(docs, msg.dataframe_calc)
            return DONE + " " + result
        else:
            # pass on the query so LLM can handle it
            return msg.query

    def ingest_docs(self, docs: List[Document], split: bool = True) -> int:
        n = super().ingest_docs(docs, split)
        tbl = self.vecdb.client.open_table(self.vecdb.config.collection_name)
        # We assume "content" is available as top-level field
        if "content" in tbl.schema.names:
            tbl.create_fts_index("content")
        return n

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        content: str = "content",
        metadata: List[str] = [],
    ) -> int:
        self.from_dataframe = True
        if df.shape[0] == 0:
            raise ValueError(
                """
                LanceDocChatAgent.ingest_dataframe() received an empty dataframe.
                """
            )
        n = df.shape[0]
        df, metadata = DocChatAgent.document_compatible_dataframe(df, content, metadata)
        self.df_description = describe_dataframe(df, sample_size=3)
        self.vecdb.add_dataframe(df, content="content", metadata=metadata)

        tbl = self.vecdb.client.open_table(self.vecdb.config.collection_name)
        # We assume "content" is available as top-level field
        if "content" in tbl.schema.names:
            tbl.create_fts_index("content")
        # We still need to do the below so that
        # other types of searches in DocChatAgent
        # can work, as they require Document objects
        docs = dataframe_to_documents(df, content="content", metadata=metadata)
        self.setup_documents(docs)
        # mark each doc as already-chunked so we don't try to split them further
        # TODO later we may want to split large text-columns
        for d in docs:
            d.metadata.is_chunk = True
        return n  # type: ignore

    def get_similar_chunks_bm25(
        self, query: str, multiple: int
    ) -> List[Tuple[Document, float]]:
        """
        Override the DocChatAgent.get_similar_chunks_bm25()
        to use LanceDB FTS (Full Text Search).
        """
        # replace all newlines with spaces in query
        query_clean = query.replace("\n", " ")
        # force special search keywords to lower case
        # so it's not interpreted as search syntax
        query_clean = (
            query_clean.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
        )

        tbl = self.vecdb.client.open_table(self.vecdb.config.collection_name)
        result = (
            tbl.search(query_clean)
            .where(self.config.filter or None)
            .limit(self.config.parsing.n_similar_docs * multiple)
        )
        docs = self.vecdb._lance_result_to_docs(result)
        scores = [r["score"] for r in result.to_list()]
        return list(zip(docs, scores))
