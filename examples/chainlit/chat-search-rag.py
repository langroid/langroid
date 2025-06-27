"""
Single-agent question-answering system that has access to
Metaphor web search when needed,
and in case a web search is used, ingests contents into a vector-db,
and uses Retrieval Augmentation to answer the question.

This is a chainlit UI version of examples/docqa/chat-search.py

Run like this:

    chainlit run examples/chainlit/chat-search-rag.py


(See here for guide to using local LLMs with Langroid:)
https://langroid.github.io/langroid/tutorials/local-llm-setup/
"""

import logging
from textwrap import dedent
from typing import Any, List, Optional

import chainlit as cl
import typer

import langroid as lr
import langroid.language_models as lm
from langroid.agent.callbacks.chainlit import (
    add_instructions,
    make_llm_settings_widgets,
    setup_llm,
    update_llm,
)
from langroid.agent.chat_agent import ChatAgent, ChatDocument
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import ForwardTool
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from langroid.parsing.web_search import metaphor_search
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER

logger = logging.getLogger(__name__)

app = typer.Typer()


class RelevantExtractsTool(ToolMessage):
    request = "relevant_extracts"
    purpose = "Get docs/extracts relevant to the <query>, from prior search results"
    query: str

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(query="when was the Mistral LLM released?"),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        IMPORTANT: You must include an ACTUAL query in the `query` field,
        """


class RelevantSearchExtractsTool(ToolMessage):
    request = "relevant_search_extracts"
    purpose = (
        "Perform an internet search for up to <num_results> results "
        "relevant to the <query>"
    )

    query: str
    num_results: int = 3

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                query="when was the Mistral LLM released?",
                num_results=3,
            ),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        IMPORTANT: You must include an ACTUAL query in the `query` field,
        """


class SearchDocChatAgent(DocChatAgent):
    tried_vecdb: bool = False

    def llm_response_async(
        self,
        message: None | str | ChatDocument = None,
    ) -> Optional[ChatDocument]:
        return ChatAgent.llm_response_async(self, message)

    def handle_message_fallback(self, msg: str | ChatDocument) -> Any:
        if isinstance(msg, ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            # non-tool LLM msg => forward to User
            return ForwardTool(agent="User")

    def relevant_extracts(self, msg: RelevantExtractsTool) -> str:
        """Get docs/extracts relevant to the query, from vecdb"""
        self.tried_vecdb = True
        self.callbacks.show_start_response(entity="agent")
        query = msg.query
        logger.info(f"Trying to get relevant extracts for query: {query}")
        _, extracts = self.get_relevant_extracts(query)
        if len(extracts) == 0:
            return """
            No extracts found! You can try doing a web search with the
            `relevant_search_extracts` tool/function-call.
            """
        return "\n".join(str(e) for e in extracts)

    def relevant_search_extracts(self, msg: RelevantSearchExtractsTool) -> str:
        """Get docs/extracts relevant to the query, from a web search"""
        if not self.tried_vecdb and len(self.original_docs) > 0:
            return "Please try the `relevant_extracts` tool, before using this tool"
        query = msg.query
        num_results = msg.num_results
        self.callbacks.show_start_response(entity="agent")
        results = metaphor_search(query, num_results)
        links = [r.link for r in results]
        self.config.doc_paths = links
        self.ingest()
        _, extracts = self.get_relevant_extracts(query)
        if len(extracts) == 0:
            return """
            No release search results found! You can try 
            rephrasing your query to see if results improve, using the
            `relevant_search_extracts` tool/function-call.
            """
        return "\n".join(str(e) for e in extracts)


async def setup_agent_task():
    """Set up Agent and Task from session settings state."""

    # set up LLM and LLMConfig from settings state
    await setup_llm()
    llm_config = cl.user_session.get("llm_config")

    set_global(
        Settings(
            debug=False,
            cache=True,
        )
    )

    config = DocChatAgentConfig(
        name="Searcher",
        llm=llm_config,
        n_similar_chunks=3,
        n_relevant_chunks=3,
        system_message=f"""
        You are a savvy, tenacious, persistent researcher, who knows when to search the 
        internet for an answer.
        
        You will try your best to answer my questions,
        in this order of preference:
        1. If you can answer from your own knowledge, simply return the answer
        2. Otherwise, use the `relevant_extracts` tool/function to
            ask me for some relevant text, and I will send you.  
            Then answer based on the relevant text.
            If I say {NO_ANSWER}, it means I found no relevant docs, and you can try 
            the next step, using a web search.
        3. If you are still unable to answer, you can use the `relevant_search_extracts`
           tool/function-call to get some text from a web search. Answer the question
           based on these text pieces.
        4. If you still can't answer, simply say {NO_ANSWER} 
        5. Be tenacious and persistent, DO NOT GIVE UP. Try asking your questions
        differently to arrive at an answer.
        
        Remember to always FIRST try `relevant_extracts` to see if there are already 
        any relevant docs, before trying web-search with `relevant_search_extracts`.
        
        Be very concise in your responses, use no more than 1-2 sentences.
        When you answer based on provided documents, be sure to show me 
        the SOURCE(s) and EXTRACT(s), for example:
        
        SOURCE: https://www.wikihow.com/Be-a-Good-Assistant-Manager
        EXTRACT: Be a Good Assistant ... requires good leadership skills.
        
        For the EXTRACT, ONLY show up to first 3 words, and last 3 words.
        """,
        parsing=ParsingConfig(  # modify as needed
            splitter=Splitter.TOKENS,
            chunk_size=200,  # aim for this many tokens per chunk
            overlap=30,  # overlap between chunks
            max_chunks=10_000,
            n_neighbor_ids=5,  # store ids of window of k chunks around each chunk.
            # aim to have at least this many chars per chunk when
            # truncating due to punctuation
            min_chunk_chars=200,
            discard_chunk_chars=5,  # discard chunks with fewer than this many chars
            # NOTE: PDF parsing is extremely challenging, each library has its own
            # strengths and weaknesses. Try one that works for your use case.
            pdf=PdfParsingConfig(
                # alternatives: "unstructured", "docling", "fitz"
                library="pymupdf4llm",
            ),
        ),
    )

    agent = SearchDocChatAgent(config)
    agent.enable_message(RelevantExtractsTool)
    agent.enable_message(RelevantSearchExtractsTool)
    collection_name = "chainlit-chat-search-rag"

    agent.vecdb.set_collection(collection_name, replace=True)

    # set up task with interactive=False, so awaits user ONLY
    # when LLM sends  non-tool msg (see handle_message_fallback method).
    task = Task(agent, interactive=False)
    cl.user_session.set("agent", agent)
    cl.user_session.set("task", task)


@cl.on_settings_update
async def on_update(settings):
    await update_llm(settings)
    await setup_agent_task()


@cl.on_chat_start
async def chat() -> None:
    await add_instructions(
        title="Welcome to the Internet Search + RAG chatbot!",
        content=dedent(
            """
        Ask me anything, especially about recent events that I may not have been trained on.
        
        I have access to two Tools, which I will try to use in order of priority:
        - `relevant_extracts` to try to answer your question using Retrieval Augmented Generation
           from prior search results ingested into a vector-DB (from prior searches in this session),
           and failing this, I will use my second tool:
        - `relevant_search_extracts` to do a web search (Using Metaphor Search)
        and ingest the results into the vector-DB, and then use 
        Retrieval Augmentation Generation (RAG) to answer the question.
        """
        ),
    )

    await make_llm_settings_widgets(
        lm.OpenAIGPTConfig(
            timeout=180,
            chat_context_length=16_000,
            chat_model="",
            temperature=0.1,
        )
    )
    await setup_agent_task()


@cl.on_message
async def on_message(message: cl.Message):
    task = cl.user_session.get("task")
    lr.ChainlitTaskCallbacks(task)
    await task.run_async(message.content)
