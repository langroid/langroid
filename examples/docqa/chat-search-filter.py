"""
Variant of chat-search.py that uses a filter to identify different
set of ingested docs (obtained from web-search), so that cross-doc
questions can be answered.

This is a single-agent question-answering system that has access to a Web-Search
Tool when needed,
and in case a web search is used, ingests scraped link contents into a vector-db,
and uses Retrieval Augmentation to answer the question.

Run like this:

    python3 examples/docqa/chat-search-filter.py

Optional args:
    -nc : turn off caching (i.e. don't retrieve cached LLM responses)
    -d: debug mode, to show all intermediate results
    -f: use OpenAI functions api instead of tools
    -m <model_name>:  (e.g. -m ollama/mistral:7b-instruct-v0.2-q4_K_M)
    (defaults to GPT4-Turbo if blank)

(See here for guide to using local LLMs with Langroid:)
https://langroid.github.io/langroid/tutorials/local-llm-setup/
"""

import json
import re
from typing import Any, List

from fire import Fire
from rich import print
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
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
from pydantic import Field
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER


class RelevantExtractsTool(ToolMessage):
    request: str = Field(
        "relevant_extracts", description="MUST be included in EVERY use of this tool!"
    )
    purpose: str = "Get docs/extracts relevant to the <query> from prior searches"
    query: str = Field(..., description="The query to get relevant extracts for")
    filter_tag: str = Field(
        "",
        description="""
        Optional LOWER-CASE tag to filter to use for the search, 
        to restrict relevance extraction to a SPECIFIC PRIOR search result.
        IMPORTANT - DO NOT INTRODUCE A NEW TAG HERE!! You MUST use ONLY a
        tag you previously used in the `relevant_search_extracts` tool,
        to correctly identify a prior search result.
        """,
    )

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                query="when was the Mistral LLM released?",
                filter_tags=["mistral", "llm"],
            ),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        IMPORTANT: You must include an ACTUAL query in the `query` field,
        """


class RelevantSearchExtractsTool(ToolMessage):
    request: str = Field(
        "relevant_search_extracts",
        description="MUST be included in EVERY use of this tool!",
    )
    purpose: str = "Get docs/extracts relevant to the <query> from a web search"
    query: str = Field(..., description="The search query to get relevant extracts for")
    num_results: int = Field(3, description="The number of search results to use")
    tag: str = Field(
        "",
        description="""
        Optional LOWER-CASE tag to attach to the documents ingested from the search, 
        to UNIQUELY IDENTIFY the docs ingested from this search, for future reference
        when using the `relevant_extracts` tool.
        """,
    )

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                query="when was the Mistral LLM released?",
                num_results=3,
                tag="mistral",
            ),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        IMPORTANT: You must include an ACTUAL query in the `query` field,
        """


def tags_to_filter(tags: List[str]) -> str | None:
    """
    Given a list of tags, create a qdrant-db filter condition expressing:
    EVERY tag MUST appear in the metadata.tags field of the document.
    Args:
        tags: List of tags to filter by
    Returns:
        json string of the qdrant filter condition, or None
    """
    if len(tags) == 0:
        return None
    match_conditions = [
        {"key": "metadata.tags", "match": {"any": [tag]}} for tag in tags
    ]

    filter = {"must": match_conditions}
    return json.dumps(filter)


class SearchDocChatAgent(DocChatAgent):

    def init_state(self) -> None:
        super().init_state()
        self.original_docs = []
        self.tried_vecdb: bool = False

    def handle_message_fallback(self, msg: str | ChatDocument) -> Any:
        if isinstance(msg, ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            # no tool, so it must be meant for user
            return ForwardTool(agent="user")

    def llm_response(
        self,
        message: None | str | ChatDocument = None,
    ) -> ChatDocument | None:
        return ChatAgent.llm_response(self, message)

    def relevant_extracts(self, msg: RelevantExtractsTool) -> str:
        """Get docs/extracts relevant to the query, from vecdb"""
        self.tried_vecdb = True
        query = msg.query
        if msg.filter_tag != "":
            self.set_filter(tags_to_filter([msg.filter_tag]))
        _, extracts = self.get_relevant_extracts(query)
        if len(extracts) == 0:
            return """
            No extracts found! You can try doing a web search with the
            `relevant_search_extracts` tool/function-call.
            """
        return "\n".join(str(e) for e in extracts)

    def relevant_search_extracts(self, msg: RelevantSearchExtractsTool) -> str:
        """Get docs/extracts relevant to the query, from a web search"""
        # if not self.tried_vecdb and len(self.original_docs) > 0:
        #     return "Please try the `relevant_extracts` tool, before using this tool"
        self.tried_vecdb = False
        query = msg.query
        # if query contains a url, then no need to do web search --
        # just ingest the specific link in the query
        if "http" in query:
            # extract the URL from the query
            url = re.search(r"(?P<url>https?://[^\s]+)", query).group("url")
            links = [url]
            # remove the url from the query
            query = re.sub(r"http\S+", "", query)
        else:
            results = metaphor_search(query, msg.num_results)
            links = [r.link for r in results]
        self.ingest_doc_paths(links, metadata={"tags": [msg.tag]})
        if msg.tag != "":
            self.set_filter(tags_to_filter([msg.tag]))
        _, extracts = self.get_relevant_extracts(query)
        return "\n".join(str(e) for e in extracts)


def main(
    debug: bool = False,
    nocache: bool = False,
    model: str = "",
    fn_api: bool = True,
) -> None:

    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )

    print(
        """
        [blue]Welcome to the Internet Search chatbot!
        I will try to answer your questions, relying on (full content of links from) 
        Duckduckgo (DDG) Search when needed.
        
        Enter x or q to quit, or ? for evidence
        """
    )

    system_msg = Prompt.ask(
        """
    [blue] Tell me who I am (give me a role) by completing this sentence: 
    You are...
    [or hit enter for default]
    [blue] Human
    """,
        default="a helpful assistant.",
    )
    system_msg = re.sub("you are", "", system_msg, flags=re.IGNORECASE)

    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        # or, other possibilities for example:
        # "litellm/bedrock/anthropic.claude-instant-v1"
        # "ollama/llama2"
        # "local/localhost:8000/v1"
        # "local/localhost:8000"
        chat_context_length=2048,  # adjust based on model
    )

    config = DocChatAgentConfig(
        use_functions_api=fn_api,
        use_tools=not fn_api,
        llm=llm_config,
        extraction_granularity=3,
        # for relevance extraction
        # relevance_extractor_config=None,  # set to None to disable relevance extraction
        # set it to > 0 to retrieve a window of k chunks on either side of a match
        n_neighbor_chunks=2,
        n_similar_chunks=5,
        n_relevant_chunks=5,
        parsing=ParsingConfig(  # modify as needed
            splitter=Splitter.TOKENS,
            chunk_size=200,  # aim for this many tokens per chunk
            overlap=50,  # overlap between chunks
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
                library="fitz",
            ),
        ),
        system_message=f"""
        {system_msg} You will try your best to answer my questions,
        in this order of preference:
        1. If you can answer from your own knowledge, simply return the answer
        2. Otherwise:
         2.1 If the question contains a URL, then use the `relevant_search_extracts`
             tool/function with the `query` field set to 
             this EXACT QUESTION INTACT! (DO NOT REPHRASE IT),
             and set the appropriate `tag` to UNIQUELY identify 
             docs from this search, to be able to refer to docs from 
             this search in FUTURE uses of the `relevant_extracts` tool.
         2.1 Otherwise, 
             if you have previously used the `relevant_search_extracts` 
                tool/fn-call
             to do a web search, you can ask for some relevant text from those search
             results, using the `relevant_extracts` tool/function-call, 
             and you MUST ONLY use a PREVIOUSLY used tag to correctly identify
             the prior search results to narrow down the search,
             and you will receive relevant extracts, if any.
             If you receive {NO_ANSWER}, it means no relevant extracts exist,
             and you can try the next step 2.2, using a web search.
             
         2.2 otherwise, i.e. you have NOT YET done a web search, you can use
             the `relevant_search_extracts` tool/function-call to search the web,
             MAKING SURE YOU SET a UNIQUE TAG (LOWER CASE, short word or 
             phrase) in the `tag` field, to UNIQUELY identify the docs from 
             this search, to be able to refer to them in a future use of 
             `relevant_extracts` tool.
             You will then receive relevant extracts from these search results, 
             if any. 
        3. If you are still unable to answer, you can use the `relevant_search_extracts`
           tool/function-call to get some text from a web search. Once you receive the
           text, you can use it to answer my question.
        4. If you still can't answer, simply say {NO_ANSWER} 
        
        Remember these simple rules:
         (a) if a question contains a URL, simply use the `relevant_search_extracts`
                tool/function-call with the `query` field set to this EXACT QUESTION
         (b) else if you have ALREADY done a web-search 
         (using the `relevant_search_extracts` tool),
         you should FIRST try `relevant_extracts` to see if there are
         any relevant passages from PREVIOUS SEARCHES, before doing a new search.
         
         YOU CAN USE TOOLS MULTIPLE TIMES before composing your answer.
         For example, when asked to compare two things, you can use the
         `relevant_extracts` tool multiple times to get relevant extracts
         from different PRIOR search results, and THEN compose your answer!
        
        Be very concise in your responses, use no more than 1-2 sentences.
        When you answer based on provided documents, be sure to show me 
        the SOURCE(s) and EXTRACT(s), for example:
        
        SOURCE: https://www.wikihow.com/Be-a-Good-Assistant-Manager
        EXTRACT: Be a Good Assistant ... requires good leadership skills.
        
        For the EXTRACT, ONLY show up to first 3 words, and last 3 words.
        """,
    )

    agent = SearchDocChatAgent(config)
    agent.enable_message(RelevantExtractsTool)
    agent.enable_message(RelevantSearchExtractsTool)
    collection_name = Prompt.ask(
        "Name a collection to use",
        default="docqa-chat-search",
    )
    replace = (
        Prompt.ask(
            "Would you like to replace (i.e. erase) this collection?",
            choices=["y", "n"],
            default="n",
        )
        == "y"
    )

    print(f"[red]Using {collection_name}")

    agent.vecdb.set_collection(collection_name, replace=replace)

    task = Task(agent, interactive=False)
    task.run("Can you help me answer some questions, possibly using web search?")


if __name__ == "__main__":
    Fire(main)
