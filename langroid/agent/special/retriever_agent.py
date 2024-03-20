"""
Deprecated: use DocChatAgent instead, with DocChatAgentConfig.retrieve_only=True,
and if you want to retrieve FULL relevant doc-contents rather than just extracts,
then set DocChatAgentConfig.extraction_granularity=-1

This is an agent to retrieve relevant extracts from a vector store,
where the LLM is used to filter for "true" relevance after retrieval from the
vector store.
This is essentially the same as DocChatAgent, except that instead of
generating final summary answer based on relevant extracts, it just returns
those extracts.
See test_retriever_agent.py for example usage.
"""

import logging
from typing import Sequence

from rich.console import Console

from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.mytypes import DocMetaData, Document

console = Console()
logger = logging.getLogger(__name__)

# for backwards compatibility:
RecordMetadata = DocMetaData
RecordDoc = Document
RetrieverAgentConfig = DocChatAgentConfig


class RetrieverAgent(DocChatAgent):
    """
    Agent for just retrieving chunks/docs/extracts matching a query
    """

    def __init__(self, config: DocChatAgentConfig):
        super().__init__(config)
        self.config: DocChatAgentConfig = config
        logger.warning(
            """
        `RetrieverAgent` is deprecated. Use `DocChatAgent` instead, with
        `DocChatAgentConfig.retrieve_only=True`, and if you want to retrieve
        FULL relevant doc-contents rather than just extracts, then set
        `DocChatAgentConfig.extraction_granularity=-1`
        """
        )

    def get_records(self) -> Sequence[Document]:
        raise NotImplementedError

    def ingest(self) -> None:
        records = self.get_records()
        if self.vecdb is None:
            raise ValueError("No vector store specified")
        self.vecdb.add_documents(records)
