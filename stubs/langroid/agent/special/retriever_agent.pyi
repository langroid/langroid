from typing import Sequence

from _typeshed import Incomplete

from langroid.agent.special.doc_chat_agent import (
    DocChatAgent as DocChatAgent,
)
from langroid.agent.special.doc_chat_agent import (
    DocChatAgentConfig as DocChatAgentConfig,
)
from langroid.mytypes import DocMetaData as DocMetaData
from langroid.mytypes import Document as Document

console: Incomplete
logger: Incomplete
RecordMetadata = DocMetaData
RecordDoc = Document
RetrieverAgentConfig = DocChatAgentConfig

class RetrieverAgent(DocChatAgent):
    config: Incomplete
    def __init__(self, config: DocChatAgentConfig) -> None: ...
    def get_records(self) -> Sequence[Document]: ...
    def ingest(self) -> None: ...
