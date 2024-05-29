from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .relevance_extractor_agent import (
        RelevanceExtractorAgent,
        RelevanceExtractorAgentConfig,
    )
    from .doc_chat_agent import DocChatAgent, DocChatAgentConfig
    from .retriever_agent import (
        RecordMetadata,
        RecordDoc,
        RetrieverAgentConfig,
        RetrieverAgent,
    )
    from .lance_doc_chat_agent import LanceDocChatAgent
    from .table_chat_agent import (
        dataframe_summary,
        TableChatAgent,
        TableChatAgentConfig,
        PandasEvalTool,
    )
    from . import sql
    from . import relevance_extractor_agent
    from . import doc_chat_agent
    from . import retriever_agent
    from . import lance_tools
    from . import lance_doc_chat_agent
    from . import lance_rag
    from . import table_chat_agent
else:
    RelevanceExtractorAgent = LazyLoad(
        "langroid.agent.special.relevance_extractor_agent.RelevanceExtractorAgent"
    )
    RelevanceExtractorAgentConfig = LazyLoad(
        "langroid.agent.special.relevance_extractor_agent.RelevanceExtractorAgentConfig"
    )

    table_chat_agent = LazyLoad("langroid.agent.special.table_chat_agent")
    dataframe_summary = LazyLoad(
        "langroid.agent.special.table_chat_agent.dataframe_summary"
    )
    TableChatAgent = LazyLoad("langroid.agent.special.table_chat_agent.TableChatAgent")
    TableChatAgentConfig = LazyLoad(
        "langroid.agent.special.table_chat_agent.TableChatAgentConfig"
    )
    PandasEvalTool = LazyLoad("langroid.agent.special.table_chat_agent.PandasEvalTool")

    sql = LazyLoad("langroid.agent.special.sql")
    relevance_extractor_agent = LazyLoad(
        "langroid.agent.special.relevance_extractor_agent"
    )

    RecordDoc = LazyLoad("langroid.agent.special.retriever_agent.RecordDoc")
    RecordMetadata = LazyLoad("langroid.agent.special.retriever_agent.RecordMetadata")
    doc_chat_agent = LazyLoad("langroid.agent.special.doc_chat_agent")
    retriever_agent = LazyLoad("langroid.agent.special.retriever_agent")
    DocChatAgent = LazyLoad("langroid.agent.special.doc_chat_agent.DocChatAgent")
    DocChatAgentConfig = LazyLoad(
        "langroid.agent.special.doc_chat_agent.DocChatAgentConfig"
    )
    RetrieverAgent = LazyLoad("langroid.agent.special.retriever_agent.RetrieverAgent")
    RetrieverAgentConfig = LazyLoad(
        "langroid.agent.special.retriever_agent.RetrieverAgentConfig"
    )

    lance_doc_chat_agent = LazyLoad("langroid.agent.special.lance_doc_chat_agent")
    lance_rag = LazyLoad("langroid.agent.special.lance_rag")
    lance_tools = LazyLoad("langroid.agent.special.lance_tools")
    LanceDocChatAgent = LazyLoad(
        "langroid.agent.special.lance_doc_chat_agent.LanceDocChatAgent"
    )


__all__ = [
    "RelevanceExtractorAgent",
    "RelevanceExtractorAgentConfig",
    "DocChatAgent",
    "DocChatAgentConfig",
    "RecordMetadata",
    "RecordDoc",
    "RetrieverAgentConfig",
    "RetrieverAgent",
    "dataframe_summary",
    "TableChatAgent",
    "TableChatAgentConfig",
    "PandasEvalTool",
    "sql",
    "relevance_extractor_agent",
    "doc_chat_agent",
    "retriever_agent",
    "table_chat_agent",
    "LanceDocChatAgent",
    "lance_tools",
    "lance_doc_chat_agent",
    "lance_rag",
]
