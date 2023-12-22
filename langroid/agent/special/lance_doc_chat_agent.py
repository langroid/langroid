"""
Version of DocChatAgent that uses LanceDB as a vector store:
- adds a FilterTool to let the agent decide if the given query requires a filter,
    to be used in the LanceDB table search as a `where` clause.
- overrides the get_similar_chunks_bm25() to use LanceDB FTS (Full Text Search).

The LanceRAGTaskCreator.new() method creates a 2-Task system that uses this agent:
- FilterTask (LanceFilterAgent) to decide if a filter is needed, and if so, what filter,
    along with a possibly rephrased query.
- RAGTask (LanceDocChatAgent) to answer the query using the filter and the documents.

Langroid's built-in task loops will ensure that the LanceFilterAgent automatically
retries with a different filter if the RAGTask returns an empty answer.

For usage see `tests/main/test_lance_doc_chat_agent.py`.

"""
import json
import logging
from typing import List, Optional, Tuple

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPT
from langroid.mytypes import Document
from langroid.utils.constants import NO_ANSWER
from langroid.utils.pydantic_utils import clean_schema
from langroid.vector_store.lancedb import LanceDB, LanceDBConfig

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
    vecdb: LanceDBConfig = LanceDBConfig()
    use_tools = False
    use_functions_api = True
    system_message = f"""
    You will receive a QUERY, to be answered based on some documents you DO NOT have 
    access to. However you know that these documents have this PAYLOAD schema:
    {{payload_schema}}
    Note that in the schema, the "type" of each field is given, and a "descripton".
    
    The PAYLOAD fields can be used as a FILTER on the documents. 

    Based on the QUERY and the PAYLOAD SCHEMA, your ONLY task is to decide:
    - whether applying a FILTER to the QUERY would help to answer it.
    - whether the QUERY needs to be REPHRASED to be answerable given the FILTER.
    (for example, the rephrased QUERY should NOT refer to fields used in the FILTER) 
        
    The FILTER must be a SQL-like condition, e.g. 
    "year > 2000 AND genre = 'ScienceFiction'".
    To ensure you get useful results, you should make your FILTER 
    NOT TOO STRICT, e.g. look for approximate match using LIKE, and
    allow non-case-sensitive matching (e.g. convert to lower-case using SQL).
        
    You must present the FILTER and (POSSIBLY rephrased QUERY)
    using the `add_filter` tool. Use dot notation to refer to nested fields, 
    e.g. "payload.metadata.year" or "payload.content".
    
    If you think no FILTER would help, you can leave the `filter` field empty.
    
    If you receive an answer that is an empty-string or {NO_ANSWER}, 
    try again with an empty or broader or better filter.
    
    When you receive a satisfactory answer, say "DONE" and nothing else.
    """


class LanceFilterAgent(ChatAgent):
    vecdb: LanceDB

    def __init__(self, config: LanceFilterAgentConfig):
        super().__init__(config)
        # This agent should only generate the FilterTool, not handle it;
        # the LanceDocChatAgent will handle it.
        self.enable_message(FilterTool, use=True, handle=False)
        is_openai_llm = (
            isinstance(self.llm, OpenAIGPT) and self.llm.is_openai_chat_model()
        )
        self.config.use_tools = not is_openai_llm
        self.config.use_functions_api = is_openai_llm
        self.system_message = self.config.system_message.format(
            payload_schema=self._get_payload_schema()
        )

    def _get_payload_schema(self) -> str:
        schema_dict = clean_schema(
            self.vecdb.schema,
            excludes=["id", "vector"],
        )
        return json.dumps(schema_dict, indent=4)

    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        result = super().llm_response(message)
        # IF LLM says "DONE", then use the content of the incoming message as
        # the content of the result.
        # This works because in the above system_message, we instructed the LLM
        # to simply say "DONE" when it receives an answer.

        if (
            result is not None
            and message is not None
            and result.content in ["DONE", "DONE.", "DONE!", "DONE"]
        ):
            content = message if isinstance(message, str) else message.content
            result.content = "DONE " + content
        return result


class LanceDocChatAgent(DocChatAgent):
    vecdb: LanceDB

    def __init__(self, cfg: DocChatAgentConfig):
        super().__init__(cfg)
        self.config: DocChatAgentConfig = cfg
        self.enable_message(FilterTool, use=False, handle=True)

    def add_filter(self, msg: FilterTool) -> str:
        """
        Handle the LLM's use of the FilterTool.
        Temporarily set the config filter and invoke the DocChatAgent.llm_response()
        """
        # create document-subset based on this filter
        self.setup_documents(filter=msg.filter or None)
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
    def new(agent: LanceDocChatAgent) -> Task:
        """
        Add a LanceFilterAgent to the LanceDocChatAgent,
        set up the corresponding Tasks, connect them,
        and return the top-level filter_task.
        """
        filter_agent_cfg = LanceFilterAgentConfig(
            vecdb=agent.config.vecdb,
        )
        filter_agent = LanceFilterAgent(filter_agent_cfg)
        filter_task = Task(
            filter_agent,
            llm_delegate=True,
            single_round=False,
            interactive=False,
            allow_null_result=False,
        )
        rag_task = Task(
            agent,
            name="LanceRAG",
            llm_delegate=False,
            single_round=False,
            interactive=False,
            allow_null_result=False,
        )
        filter_task.add_sub_task(rag_task)
        return filter_task
