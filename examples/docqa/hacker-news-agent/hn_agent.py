from langroid.embedding_models.models import GeminiEmbeddingsConfig
import os
from langroid.exceptions import LangroidImportError

import asyncio
from pathlib import Path
import logging
import re
import json
from typing import Any, List, Optional, Dict, Union
from langroid.utils.configuration import Settings, set_global
from fire import Fire
from rich import print
from rich.prompt import Prompt
import typer
import langroid as lr
import langroid.language_models as lm
from langroid.agent.chat_agent import ChatAgent, ChatDocument
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.mytypes import DocMetaData, Document
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import ForwardTool
from langroid.parsing.url_loader import (
    Crawl4aiConfig,
    ExaCrawlerConfig,
    FirecrawlConfig,
    TrafilaturaConfig,
)
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER
from hn_parsing_utils import (
    ThreadMetadata,
    ThreadDoc,
    parse_comment_thread,
    process_hn_json,
)
from hn_crawler import hn_crawl_session

logger = logging.getLogger(__name__)


class WebSearchResult:
    """Simple data class for web search results"""

    def __init__(self, title: str, link: Optional[str]):
        self.title = title
        self.link = link


def exa_search(
    query: str,
    num_results: int = 5,
    start_crawl_date: Optional[str] = None,  # YYYY-MM-DD
    end_crawl_date: Optional[str] = None,
) -> List[WebSearchResult]:
    """
    Custom Exa search with support for domain filters and crawl dates.
    """

    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY environment variable is not set.")

    try:
        from exa_py import Exa
    except ImportError:
        raise LangroidImportError("exa-py", "exa")

    client = Exa(api_key=api_key)

    try:
        response = client.search(
            query=query,
            num_results=num_results,
            include_domains=["news.ycombinator.com"],
            start_crawl_date=start_crawl_date,
            end_crawl_date=end_crawl_date,
        )
        raw_results = response.results

        return [
            WebSearchResult(
                title=result.title or "",
                link=result.url,
            )
            for result in raw_results
            if result.url
        ]
    except Exception as e:
        return [
            WebSearchResult(
                title=f"Error: {str(e)}",
                link=None,
            )
        ]


class RelevantExtractsTool(ToolMessage):
    request: str = "relevant_extracts"
    purpose: str = "Get docs/extracts relevant to the <query>"
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
    request: str = "relevant_search_extracts"
    purpose: str = "Get docs/extracts relevant to the <query> from a web search"
    query: str
    num_results: int = 3
    start_crawl_date: Optional[str] = None
    end_crawl_date: Optional[str] = None

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                query="when was the Mistral LLM released?",
                num_results=3,
            ),
            cls(
                query="latest developments in AI safety",
                num_results=5,
            ),
            cls(
                query="OpenAI GPT-4 discussions",
                num_results=4,
                start_crawl_date="2024-01-01",
                end_crawl_date="2024-12-31",
            ),
            cls(
                query="cryptocurrency trends",
                num_results=3,
                start_crawl_date="2024-06-01",
            ),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        Use this tool to search Hacker News for new information not in the vector database.
        IMPORTANT: You must include an ACTUAL query in the `query` field.
        This will search HN using Exa, crawl the results, ingest them, and return relevant extracts.
        
        Parameters:
        - query: The search query (required)
        - num_results: Number of search results to process (optional, default: 3)
        - start_crawl_date: Earliest crawl date in YYYY-MM-DD format (optional)
        - end_crawl_date: Latest crawl date in YYYY-MM-DD format (optional)
        
        The date parameters help filter results to specific time periods when available.
        """


class HNAgent(DocChatAgent):
    def __init__(self, cfg: DocChatAgentConfig):
        super().__init__(cfg)
        self.config: DocChatAgentConfig = cfg

    def ingest_hn(
        self,
        data_or_path: Union[str, Dict[str, Any]],
        split_strategy: str = "per_thread",
        text_split: bool = True,
    ) -> int:
        """
        Ingest HN threads from either a JSON file path or raw JSON data.

        Args:
            data_or_path: Either a file path to the JSON file or the raw JSON data as a dict.
            split_strategy: How to split the content ("per_thread", "per_comment", "whole_post").
            text_split: Whether to apply additional text chunking within documents.

        Returns:
            Number of documents ingested.
        """
        try:
            if isinstance(data_or_path, str):
                with open(data_or_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
            else:
                json_data = data_or_path

            thread_docs = process_hn_json(json_data, split_strategy)

            # Convert to Document objects
            docs = [
                Document(content=doc.content, metadata=doc.metadata)
                for doc in thread_docs
            ]

            # Ingest the documents
            n = super().ingest_docs(docs, split=text_split)

            print(f"Ingested {n} documents using '{split_strategy}' strategy")
            return n

        except Exception as e:
            logger.error(f"Error ingesting HN data: {e}")
            return 0

    def relevant_extracts(self, msg: RelevantExtractsTool) -> str:
        """Get docs/extracts relevant to the query, from vecdb"""
        self.tried_vecdb = True
        query = msg.query
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
        self.tried_vecdb = False
        query = msg.query
        num_results = msg.num_results
        start_crawl_date = getattr(msg, "start_crawl_date", None)
        end_crawl_date = getattr(msg, "end_crawl_date", None)
        logger.warning("Trying exa search...")
        results = exa_search(query, num_results, start_crawl_date, end_crawl_date)
        links = [r.link for r in results]

        session_dir = asyncio.run(hn_crawl_session(links))

        # Loop through all post JSON files in the session directory
        total_docs = 0
        for json_file in Path(session_dir).glob("post_*.json"):
            print(f"Ingesting {json_file.name}...")
            num_docs = self.ingest_hn(str(json_file))
            print(f"✓ Ingested {num_docs} from {json_file.name}")
            total_docs += num_docs
        logger.warning(f"Ingested {len(links)} links into vecdb")
        print(
            f"\n✅ Successfully ingested total {total_docs} documents from session {session_dir}"
        )
        _, extracts = self.get_relevant_extracts(query)
        return "\n".join(str(e) for e in extracts)

    def llm_response(
        self,
        message: None | str | ChatDocument = None,
    ) -> ChatDocument | None:
        # override llm_response of DocChatAgent to allow use of the tools.
        return ChatAgent.llm_response(self, message)

    def handle_message_fallback(self, msg: str | ChatDocument) -> Any:
        if isinstance(msg, ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            return ForwardTool(agent="user")


app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            cache_type="fakeredis",
        )
    )

    # Configs
    embed_cfg = GeminiEmbeddingsConfig()
    vecdb_cfg = lr.vector_store.QdrantDBConfig(
        docker=True,
        collection_name="hn_db",
        embedding=embed_cfg,
    )

    cfg = DocChatAgentConfig(
        vecdb=vecdb_cfg,
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gemini/gemini-2.0-flash"),
        use_tools=True,
        system_message="""
        You are hackernews summary creation agent. You try to generate detailed summaries based on the following preference.  
        
        1.  Ask me for what is my relevant_text , and I will send you. Use the 
            `relevant_extracts` tool/function-call for this purpose. Once you receive 
            the text, you can use it to answer my question. 
            If I say {NO_ANSWER}, it means I found no relevant docs, and you can try 
            the next step, using a web search.
        2. If you are still unable to answer, you can use the `relevant_search_extracts`
           tool/function-call to get some text from a web search. Once you receive the
           text, you can use it to answer my question.
        3. If you still can't answer, simply say {NO_ANSWER} 
        
        Remember to always FIRST try `relevant_extracts` to see if there are already 
        any relevant docs, before trying web-search with `relevant_search_extracts`.
        use all of these extracts and form very informed report for the relevant_test.
        
        """,
    )

    agent = HNAgent(cfg)
    # urls = [
    #     "https://news.ycombinator.com/item?id=42157556",
    #     "https://news.ycombinator.com/item?id=44594475",
    #     "https://news.ycombinator.com/item?id=44492290",
    # ]
    #
    # Create a task for the agent

    agent.enable_message(
        [
            RelevantExtractsTool,
            RelevantSearchExtractsTool,
        ]
    )

    collection_name = Prompt.ask(
        "Name a collection to use",
        default="hn_db",
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

    # Start interactive chat
    print("HN Thread Chat Agent ready! Ask questions about the ingested threads.")
    task.run()


if __name__ == "__main__":
    app()
