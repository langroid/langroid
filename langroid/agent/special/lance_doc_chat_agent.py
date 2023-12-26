"""
LanceDocChatAgent is a subclass of DocChatAgent that uses LanceDB as a vector store:
- Uses the DocChatAgentConfig.filter variable
    (a sql string) in the `where` clause to do filtered vector search.
- Overrides the get_similar_chunks_bm25() to use LanceDB FTS (Full Text Search).

The LanceRAGTaskCreator.new() method creates a 2-Agent system that uses this agent.
It takes a LanceDocChatAgent instance as argument, and:
- creates a LanceFilterAgent, which is given the LanceDB schema in LanceDocChatAgent,
  and based on this schema decides for a user query, whether a filter can help,
  and if so, what filter, along with a possibly rephrased query.
- sets up the LanceFilterAgent's task as the "main" task interacting with the user,
     and adds the LanceDocChatAgent's task as sub-task.

Langroid's built-in task loops will ensure that the LanceFilterAgent automatically
retries with a different filter if the RAGTask returns an empty answer.

For usage see:
 - `tests/main/test_lance_doc_chat_agent.py`.
 - example script `examples/docqa/lance_rag.py`.

"""
import json
import logging
from typing import List, Tuple

import pandas as pd

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPT
from langroid.mytypes import Document, Entity
from langroid.utils.constants import DONE, NO_ANSWER, PASS
from langroid.utils.pydantic_utils import (
    clean_schema,
    dataframe_to_documents,
)
from langroid.vector_store.lancedb import LanceDB

logger = logging.getLogger(__name__)


class FilterTool(ToolMessage):
    request = "add_filter"  # the agent method name that handles this tool
    purpose = """
    Given a query, determine if a <filter> condition is needed, and present
    the <filter> (which can be emptry string '' if no filter is needed), 
    and a possibly rephrased <query>.
    """
    filter: str
    query: str


class LanceFilterAgentConfig(ChatAgentConfig):
    name = "LanceFilter"
    vecdb_schema: str
    use_tools = False
    use_functions_api = True
    system_message = f"""
    You will receive a QUERY, to be answered based on some documents you DO NOT have 
    access to. However you know that these documents have this SCHEMA:
    {{doc_schema}}
    Note that in the schema, the "type" of each field is given, and a "descripton".
    
    The SCHEMA fields can be used as a FILTER on the documents. 

    Based on the QUERY and the SCHEMA, your ONLY task is to decide:
    - whether applying a FILTER to the QUERY would help to answer it.
    - whether the QUERY needs to be REPHRASED to be answerable given the FILTER.
    (for example, the rephrased QUERY should NOT refer to fields used in the FILTER) 
        
    The FILTER must be a SQL-like condition, e.g. 
    "year > 2000 AND genre = 'ScienceFiction'".
    To ensure you get useful results, you should make your FILTER 
    NOT TOO STRICT, e.g. look for approximate match using LIKE, etc.
        
    You must present the FILTER and (POSSIBLY rephrased QUERY)
    using the `add_filter` tool. Use dot notation to refer to nested fields, 
    e.g. "payload.metadata.year" or "metadata.author".
    
    If you think no FILTER would help, you can leave the `filter` field empty.
    
    If you receive an answer that is an empty-string or {NO_ANSWER}, 
    try a NEW FILTER, i.e. an empty or broader or better filter.
    
    When you receive a satisfactory answer,
    or if you're still getting NO_ANSWER after trying a few filters, 
    say {DONE} {PASS} and nothing else.
    
    If there is no query, ask the user what they want to know.
    """


class LanceFilterAgent(ChatAgent):
    def __init__(self, config: LanceFilterAgentConfig):
        super().__init__(config)
        self.config: LanceFilterAgentConfig = config
        # This agent should generate the FilterTool,
        # as well as handle it for validation
        self.enable_message(FilterTool, use=True, handle=True)
        is_openai_llm = (
            isinstance(self.llm, OpenAIGPT) and self.llm.is_openai_chat_model()
        )
        self.config.use_tools = not is_openai_llm
        self.config.use_functions_api = is_openai_llm
        self.system_message = self.config.system_message.format(
            doc_schema=self.config.vecdb_schema,
        )

    def add_filter(self, msg: FilterTool) -> str:
        """Valid, so pass it on to sub-task"""
        return PASS

    # def llm_response(
    #     self, message: Optional[str | ChatDocument] = None
    # ) -> Optional[ChatDocument]:
    #     result = super().llm_response(message)
    #     # IF LLM says "DONE", then use the content of the incoming message as
    #     # the content of the result.
    #     # This works because in the above system_message, we instructed the LLM
    #     # to simply say "DONE" when it receives an answer.
    #
    #     if (
    #         result is not None
    #         and message is not None
    #         and result.content in ["DONE", "DONE.", "DONE!", "DONE"]
    #     ):
    #         content = message if isinstance(message, str) else message.content
    #         result.content = "DONE " + content
    #     return result


class LanceDocChatAgent(DocChatAgent):
    vecdb: LanceDB

    def __init__(self, cfg: DocChatAgentConfig):
        super().__init__(cfg)
        self.config: DocChatAgentConfig = cfg
        self.enable_message(FilterTool, use=False, handle=True)

    def _get_clean_vecdb_schema(self) -> str:
        schema_dict = clean_schema(
            self.vecdb.schema,
            excludes=["id", "vector"],
        )
        return json.dumps(schema_dict, indent=4)

    def add_filter(self, msg: FilterTool) -> str:
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
        # pass on the query so LLM can handle it
        return msg.query

    def ingest_docs(self, docs: List[Document], split: bool = True) -> int:
        n = super().ingest_docs(docs, split)
        if self.vecdb.config.flatten:
            tbl = self.vecdb.client.open_table(self.vecdb.config.collection_name)
            payload_content = "payload__content"
            if payload_content in tbl.schema.names:
                tbl.create_fts_index(payload_content)
        return n

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        content: str = "content",
        metadata: List[str] = [],
    ) -> int:
        n = df.shape[0]
        df, metadata = DocChatAgent.document_compatible_dataframe(df, content, metadata)
        self.vecdb.add_dataframe(df, content="content", metadata=metadata)
        docs = dataframe_to_documents(df, content="content", metadata=metadata)
        self.setup_documents(docs)
        return n  # type: ignore

    def _get_similar_chunks_bm25_(
        self, query: str, multiple: int
    ) -> List[Tuple[Document, float]]:
        """
        Override the DocChatAgent.get_similar_chunks_bm25()
        to use LanceDB FTS (Full Text Search).
        """
        if not self.vecdb.config.flatten:
            # in this case we can't use FTS since we don't have
            # access to the payload fields.
            # TODO: get rid of this and the below checks
            # when LanceDB supports nested fields:
            # https://github.com/lancedb/lance/issues/1739
            # PR pending: https://github.com/lancedb/lancedb/pull/723
            return super().get_similar_chunks_bm25(query, multiple)
        tbl = self.vecdb.client.open_table(self.vecdb.config.collection_name)
        payload_content = "payload__content"
        if payload_content not in tbl.schema.names:
            return super().get_similar_chunks_bm25(query, multiple)
        columns = tbl.schema.names
        results = (
            tbl.search(query)
            .where(self.config.filter or None)
            .limit(self.config.parsing.n_similar_docs * multiple)
            .to_list()
        )
        scores = [r["score"] for r in results]
        non_scores = [{c: r[c] for c in columns} for r in results]
        docs = self.vecdb._records_to_docs(non_scores)
        return list(zip(docs, scores))


class LanceRAGTaskCreator:
    @staticmethod
    def new(agent: LanceDocChatAgent, interactive: bool = True) -> Task:
        """
        Add a LanceFilterAgent to the LanceDocChatAgent,
        set up the corresponding Tasks, connect them,
        and return the top-level filter_task.
        """
        filter_agent_cfg = LanceFilterAgentConfig(
            vecdb_schema=agent._get_clean_vecdb_schema(),
        )
        filter_agent = LanceFilterAgent(filter_agent_cfg)
        filter_task = Task(
            filter_agent,
            interactive=interactive,
            llm_delegate=True,
        )
        rag_task = Task(
            agent,
            name="LanceRAG",
            interactive=False,
            done_if_response=[Entity.LLM],
            done_if_no_response=[Entity.LLM],
        )
        filter_task.add_sub_task(rag_task)
        return filter_task
