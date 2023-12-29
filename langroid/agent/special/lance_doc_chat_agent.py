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
from typing import List, Optional, Tuple

import pandas as pd

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.mytypes import Document, Entity
from langroid.parsing.table_loader import describe_dataframe
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
    use_tools = False
    use_functions_api = True
    vecdb_schema: str = ""
    system_message = f"""
    You will receive a QUERY, to be answered based on an EXTREMELY LARGE collection
    of documents you DO NOT have access to. However you know that these documents have 
    this SCHEMA:
    
    {{doc_schema}}
    
    The SCHEMA fields can be used as a FILTER on the documents. 

    Based on the QUERY and the SCHEMA, your ONLY task is to decide:
    - whether applying a FILTER would help to answer it.
    - whether the QUERY needs to be REPHRASED to be answerable given the FILTER.
    
    The (possibly rephrased) QUERY should be answerable by your assistant
    who DOES have access to the documents, and they will FIRST APPLY the FILTER
    before answering the QUERY. 

    KEEP THIS IN MIND: The FILTER narrows the set of matching documents,
    and the QUERY must make sense in the context of the FILTER.
    
    Example:
    ------- 
    ORIGINAL QUERY: Tell me about crime movies rated over 8 made in 2023.
    FILTER: genre = 'Crime' AND rating > 8 AND year = 2023
    REPHRASED QUERY: Tell me about the movies. 
        [NOTE how the REPHRASED QUERY does NOT mention crime, rating, or year,
        since those are already taken care of by the FILTER.]
    
    The FILTER must be a SQL-like condition, e.g. 
    "year > 2000 AND genre = 'ScienceFiction'".
    To ensure you get useful results, you should make your FILTER 
    NOT TOO STRICT, e.g. look for approximate match using LIKE, etc.
        
    You must present the FILTER and (POSSIBLY rephrased QUERY)
    using the `add_filter` tool. Use dot notation to refer to nested fields. 
        
    If you think no FILTER would help, you can leave the `filter` field empty.
    
    If you receive an answer that is an empty-string or {NO_ANSWER}, 
    try a NEW FILTER, i.e. an empty or broader or better filter.
    
    When you receive a satisfactory answer,
    or if you're still getting NO_ANSWER after trying a few filters, 
    say {DONE} {PASS} and nothing else.
    
    At the BEGINNING if there is no query, ASK the user what they want to know.
    """


class LanceFilterAgent(ChatAgent):
    def __init__(self, config: LanceFilterAgentConfig):
        super().__init__(config)
        self.config: LanceFilterAgentConfig = config
        # This agent should generate the FilterTool,
        # as well as handle it for validation
        self.enable_message(FilterTool, use=True, handle=True)
        if (self.config.vecdb_schema or None) is None:
            raise ValueError(
                """
                LanceFilterAgentConfig.vecdb_schema must be non-empty,
                otherwise LanceFilterAgent cannot be used.
                """
            )
        self.system_message = self.config.system_message.format(
            doc_schema=self.config.vecdb_schema,
        )

    def add_filter(self, msg: FilterTool) -> str:
        """Valid, so pass it on to sub-task"""
        return PASS

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """When this agent receives answer from RAGTask, the LLM
        may forget to say DONE PASS, and simply re-state answer,
        in that case this fallback method will say DONE PASS
        so the task ends rather than going to the RAGTask"""
        if isinstance(msg, ChatDocument) and msg.metadata.sender == Entity.LLM:
            return f"{DONE} {PASS}"
        return None

    def llm_response(
        self,
        query: None | str | ChatDocument = None,
    ) -> Optional[ChatDocument]:
        """Replace DONE with DONE PASS in case LLM says DONE without PASS"""
        response = super().llm_response(query)
        if response is None:
            return None
        if DONE in response.content and PASS not in response.content:
            response.content = response.content.replace(DONE, f"{DONE} {PASS}")
        return response

    async def llm_response_async(
        self,
        query: None | str | ChatDocument = None,
    ) -> Optional[ChatDocument]:
        """Replace DONE with DONE PASS in case LLM says DONE without PASS"""
        response = await super().llm_response_async(query)
        if response is None:
            return None
        if DONE in response.content and PASS not in response.content:
            response.content = response.content.replace(DONE, f"{DONE} {PASS}")
        return response


class LanceDocChatAgent(DocChatAgent):
    vecdb: LanceDB

    def __init__(self, cfg: DocChatAgentConfig):
        super().__init__(cfg)
        self.config: DocChatAgentConfig = cfg
        self.enable_message(FilterTool, use=False, handle=True)

    def _get_clean_vecdb_schema(self) -> str:
        if self.from_dataframe:
            return self.df_description
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

        tbl = self.vecdb.client.open_table(self.vecdb.config.collection_name)
        result = (
            tbl.search(query_clean)
            .where(self.config.filter or None)
            .limit(self.config.parsing.n_similar_docs * multiple)
        )
        docs = self.vecdb._lance_result_to_docs(result)
        scores = [r["score"] for r in result.to_list()]
        return list(zip(docs, scores))


class LanceRAGTaskCreator:
    @staticmethod
    def new(
        agent: LanceDocChatAgent,
        filter_agent_config: LanceFilterAgentConfig = LanceFilterAgentConfig(),
        interactive: bool = True,
    ) -> Task:
        """
        Add a LanceFilterAgent to the LanceDocChatAgent,
        set up the corresponding Tasks, connect them,
        and return the top-level filter_task.
        """
        filter_agent_config.vecdb_schema = agent._get_clean_vecdb_schema()

        filter_agent = LanceFilterAgent(filter_agent_config)
        filter_task = Task(
            filter_agent,
            interactive=interactive,
        )
        rag_task = Task(
            agent,
            name="LanceRAG",
            interactive=False,
            done_if_response=[Entity.LLM],  # done when non-null response from LLM
            done_if_no_response=[Entity.LLM],  # done when null response from LLM
        )
        filter_task.add_sub_task(rag_task)
        return filter_task
