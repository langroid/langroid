from typing import Any

import pandas as pd
from _typeshed import Incomplete

from langroid.agent.special.doc_chat_agent import (
    DocChatAgent as DocChatAgent,
)
from langroid.agent.special.doc_chat_agent import (
    DocChatAgentConfig as DocChatAgentConfig,
)
from langroid.agent.special.lance_tools import QueryPlanTool as QueryPlanTool
from langroid.mytypes import DocMetaData as DocMetaData
from langroid.mytypes import Document as Document
from langroid.parsing.table_loader import describe_dataframe as describe_dataframe
from langroid.utils.constants import DONE as DONE
from langroid.utils.constants import NO_ANSWER as NO_ANSWER
from langroid.utils.pydantic_utils import (
    clean_schema as clean_schema,
)
from langroid.utils.pydantic_utils import (
    dataframe_to_documents as dataframe_to_documents,
)
from langroid.vector_store.lancedb import LanceDB as LanceDB

logger: Incomplete

class LanceDocChatAgent(DocChatAgent):
    vecdb: LanceDB
    config: Incomplete
    def __init__(self, cfg: DocChatAgentConfig) -> None: ...
    def query_plan(self, msg: QueryPlanTool) -> str: ...
    def ingest_docs(
        self,
        docs: list[Document],
        split: bool = True,
        metadata: (
            list[dict[str, Any]] | dict[str, Any] | DocMetaData | list[DocMetaData]
        ) = [],
    ) -> int: ...
    from_dataframe: bool
    df_description: Incomplete
    def ingest_dataframe(
        self, df: pd.DataFrame, content: str = "content", metadata: list[str] = []
    ) -> int: ...
    def get_similar_chunks_bm25(
        self, query: str, multiple: int
    ) -> list[tuple[Document, float]]: ...
