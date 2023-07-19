"""
Agent that supports asking queries about a set of documents, using
retrieval-augmented queries.
Functionality includes:
- summarizing a document, with a custom instruction; see `summarize_docs`
- asking a question about a document; see `answer_from_docs`
"""
import logging
from contextlib import ExitStack
from typing import List, Optional, no_type_check

from rich import print
from rich.console import Console

from langroid.agent.base import Agent
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.language_models.base import StreamingIfAllowed
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.mytypes import DocMetaData, Document, Entity
from langroid.parsing.parser import ParsingConfig, Splitter
from langroid.parsing.repo_loader import RepoLoader
from langroid.parsing.url_loader import URLLoader
from langroid.parsing.urls import get_urls_and_paths
from langroid.prompts.prompts_config import PromptsConfig
from langroid.prompts.templates import SUMMARY_ANSWER_PROMPT_GPT4
from langroid.utils.configuration import settings
from langroid.utils.constants import NO_ANSWER
from langroid.utils.output.printing import show_if_debug
from langroid.vector_store.base import VectorStoreConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig

logger = logging.getLogger(__name__)

console = Console()

DEFAULT_DOC_CHAT_INSTRUCTIONS = """
Your task is to answer questions about various documents.
You will be given various passages from these documents, and asked to answer questions 
about them, or summarize them into coherent answers.
"""

DEFAULT_DOC_CHAT_SYSTEM_MESSAGE = """
You are a helpful assistant, helping me understand a collection of documents.
"""


class DocChatAgentConfig(ChatAgentConfig):
    """
    Attributes:
        max_context_tokens (int): threshold to use for various steps, e.g.
            if we are able to fit the current stage of doc processing into
            this many tokens, we skip additional compression steps, and
            use the current docs as-is in the context
        conversation_mode (bool): if True, we will accumulate message history,
            and pass entire history to LLM at each round.
            If False, each request to LLM will consist only of the
            initial task messages plus the current query.
    """

    system_message: str = DEFAULT_DOC_CHAT_SYSTEM_MESSAGE
    user_message: str = DEFAULT_DOC_CHAT_INSTRUCTIONS
    summarize_prompt: str = SUMMARY_ANSWER_PROMPT_GPT4
    max_context_tokens: int = 1000
    conversation_mode: bool = True
    cache: bool = True  # cache results
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    doc_paths: List[str] = []
    default_paths: List[str] = [
        "https://news.ycombinator.com/item?id=35629033",
        "https://www.newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-jpeg-of-the-web",
        "https://www.wired.com/1995/04/maes/",
        "https://cthiriet.com/articles/scaling-laws",
        "https://www.jasonwei.net/blog/emergence",
        "https://www.quantamagazine.org/the-unpredictable-abilities-emerging-from-large-ai-models-20230316/",
        "https://ai.googleblog.com/2022/11/characterizing-emergent-phenomena-in.html",
    ]
    parsing: ParsingConfig = ParsingConfig(  # modify as needed
        splitter=Splitter.TOKENS,
        chunk_size=100,  # aim for this many tokens per chunk
        max_chunks=10_000,
        # aim to have at least this many chars per chunk when
        # truncating due to punctuation
        min_chunk_chars=350,
        discard_chunk_chars=5,  # discard chunks with fewer than this many chars
        n_similar_docs=4,
    )
    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name=None,
        storage_path=".qdrant/data/",
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    )
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        completion_model=OpenAIChatModel.GPT4,
    )
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


class DocChatAgent(ChatAgent):
    """
    Agent for chatting with a collection of documents.
    """

    def __init__(
        self,
        config: DocChatAgentConfig,
    ):
        super().__init__(config)
        self.config: DocChatAgentConfig = config
        self.original_docs: None | List[Document] = None
        self.original_docs_length = 0
        self.response: None | Document = None
        if len(config.doc_paths) > 0:
            self.ingest()

    def ingest(self) -> None:
        """
        Chunk + embed + store docs specified by self.config.doc_paths

        Returns:
            dict with keys:
                n_splits: number of splits
                urls: list of urls
                paths: list of file paths
        """
        if len(self.config.doc_paths) == 0:
            return
        urls, paths = get_urls_and_paths(self.config.doc_paths)
        docs: List[Document] = []
        if len(urls) > 0:
            loader = URLLoader(urls=urls)
            docs = loader.load()
        if len(paths) > 0:
            for p in paths:
                path_docs = RepoLoader.get_documents(p)
                docs.extend(path_docs)
        n_docs = len(docs)
        n_splits = self.ingest_docs(docs)
        if n_docs == 0:
            return
        n_urls = len(urls)
        n_paths = len(paths)
        print(
            f"""
        [green]I have processed the following {n_urls} URLs 
        and {n_paths} paths into {n_splits} parts:
        """.strip()
        )
        print("\n".join(urls))
        print("\n".join(paths))

    def ingest_docs(self, docs: List[Document]) -> int:
        """
        Chunk docs into pieces, map each chunk to vec-embedding, store in vec-db
        """
        self.original_docs = docs
        if self.parser is None:
            raise ValueError("Parser not set")
        docs = self.parser.split(docs)
        if self.vecdb is None:
            raise ValueError("VecDB not set")
        self.vecdb.add_documents(docs)
        self.original_docs_length = self.doc_length(docs)
        return len(docs)

    def doc_length(self, docs: List[Document]) -> int:
        """
        Calc token-length of a list of docs
        Args:
            docs: list of Document objects
        Returns:
            int: number of tokens
        """
        if self.parser is None:
            raise ValueError("Parser not set")
        return self.parser.num_tokens(self.doc_string(docs))

    @no_type_check
    def llm_response(
        self,
        query: None | str | ChatDocument = None,
    ) -> Optional[ChatDocument]:
        if not self.llm_can_respond(query):
            return None
        query_str: str | None
        if isinstance(query, ChatDocument):
            query_str = query.content
        else:
            query_str = query
        if query_str is None or query_str.startswith("!"):
            # direct query to LLM
            query_str = query_str[1:] if query_str is not None else None
            if self.llm is None:
                raise ValueError("LLM not set")
            with StreamingIfAllowed(self.llm):
                response = super().llm_response(query_str)
            if query_str is not None:
                self.update_dialog(query_str, response.content)
            return response
        if query_str == "":
            return None
        elif query_str == "?" and self.response is not None:
            return self.justify_response()
        elif (query_str.startswith(("summar", "?")) and self.response is None) or (
            query_str == "??"
        ):
            return self.summarize_docs()
        else:
            response = self.answer_from_docs(query_str)
            return ChatDocument(
                content=response.content,
                metadata=ChatDocMetaData(
                    source=response.metadata.source,
                    sender=Entity.LLM,
                ),
            )

    @staticmethod
    def doc_string(docs: List[Document]) -> str:
        """
        Generate a string representation of a list of docs.
        Args:
            docs: list of Document objects
        Returns:
            str: string representation
        """
        contents = [f"Extract: {d.content}" for d in docs]
        sources = [d.metadata.source for d in docs]
        sources = [f"Source: {s}" if s is not None else "" for s in sources]
        return "\n".join(
            [
                f"""
                {content}
                {source}
                """
                for (content, source) in zip(contents, sources)
            ]
        )

    def get_summary_answer(self, question: str, passages: List[Document]) -> Document:
        """
        Given a question and a list of (possibly) doc snippets,
        generate an answer if possible
        Args:
            question: question to answer
            passages: list of `Document` objects each containing a possibly relevant
                snippet, and metadata
        Returns:
            a `Document` object containing the answer,
            and metadata containing source citations

        """

        passages_str = self.doc_string(passages)
        # Substitute Q and P into the templatized prompt

        final_prompt = self.config.summarize_prompt.format(
            question=f"Question:{question}", extracts=passages_str
        )
        show_if_debug(final_prompt, "SUMMARIZE_PROMPT= ")

        # Generate the final verbatim extract based on the final prompt.
        # Note this will send entire message history, plus this final_prompt
        # to the LLM, and self.message_history will be updated to include
        # 2 new LLMMessage objects:
        # one for `final_prompt`, and one for the LLM response

        # TODO need to "forget" last two messages in message_history
        # if we are not in conversation mode

        if self.config.conversation_mode:
            # respond with temporary context
            answer_doc = super()._llm_response_temp_context(question, final_prompt)
        else:
            answer_doc = super().llm_response_forget(final_prompt)

        final_answer = answer_doc.content.strip()
        show_if_debug(final_answer, "SUMMARIZE_RESPONSE= ")
        parts = final_answer.split("SOURCE:", maxsplit=1)
        if len(parts) > 1:
            content = parts[0].strip()
            sources = parts[1].strip()
        else:
            content = final_answer
            sources = ""
        return Document(
            content=content,
            metadata=DocMetaData(
                source="SOURCE: " + sources,
                sender=Entity.LLM,
                cached=getattr(answer_doc.metadata, "cached", False),
            ),
        )

    @no_type_check
    def answer_from_docs(self, query: str) -> Document:
        """Answer query based on docs in vecdb, and conv history"""
        response = Document(
            content=NO_ANSWER,
            metadata=DocMetaData(
                source="None",
            ),
        )
        if len(self.dialog) > 0 and not self.config.conversation_mode:
            # In conversation mode, we let self.message_history accumulate
            # and do not need to convert to standalone query
            # (We rely on the LLM to interpret the new query in the context of
            # the message history so far)
            with console.status("[cyan]Converting to stand-alone query...[/cyan]"):
                with StreamingIfAllowed(self.llm, False):
                    query = self.llm.followup_to_standalone(self.dialog, query)
            print(f"[orange2]New query: {query}")

        passages = self.original_docs

        # if original docs too long, no need to look for relevant parts.
        if (
            passages is None
            or self.original_docs_length > self.config.max_context_tokens
        ):
            with console.status("[cyan]Searching VecDB for relevant doc passages..."):
                docs_and_scores = self.vecdb.similar_texts_with_scores(
                    query,
                    k=self.config.parsing.n_similar_docs,
                )
            if len(docs_and_scores) == 0:
                return response
            passages = [
                Document(content=d.content, metadata=d.metadata)
                for (d, _) in docs_and_scores
            ]

        # if passages not too long, no need to extract relevant verbatim text
        extracts = passages
        if self.doc_length(passages) > self.config.max_context_tokens:
            with console.status("[cyan]LLM Extracting verbatim passages..."):
                with StreamingIfAllowed(self.llm, False):
                    extracts = self.llm.get_verbatim_extracts(query, passages)
        with ExitStack() as stack:
            # conditionally use Streaming or rich console context
            cm = (
                StreamingIfAllowed(self.llm)
                if settings.stream
                else (console.status("LLM Generating final answer..."))
            )
            stack.enter_context(cm)
            response = self.get_summary_answer(query, extracts)

        self.update_dialog(query, response.content)
        self.response = response  # save last response
        return response

    def summarize_docs(
        self,
        instruction: str = "Give a concise summary of the following text:",
    ) -> None | ChatDocument:
        """Summarize all docs"""
        if self.original_docs is None:
            logger.warning(
                """
                No docs to summarize! Perhaps you are re-using a previously
                defined collection? 
                In that case, we don't have access to the original docs.
                To create a summary, use a new collection, and specify a list of docs. 
                """
            )
            return None
        full_text = "\n\n".join([d.content for d in self.original_docs])
        if self.parser is None:
            raise ValueError("No parser defined")
        tot_tokens = self.parser.num_tokens(full_text)
        model = (
            self.config.llm.chat_model
            if self.config.llm.use_chat_for_completion
            else self.config.llm.completion_model
        )
        MAX_INPUT_TOKENS = (
            self.config.llm.context_length[model]
            - self.config.llm.max_output_tokens
            - 100
        )
        if tot_tokens > MAX_INPUT_TOKENS:
            # truncate
            full_text = self.parser.tokenizer.decode(
                self.parser.tokenizer.encode(full_text)[:MAX_INPUT_TOKENS]
            )
            logger.warning(
                f"Summarizing after truncating text to {MAX_INPUT_TOKENS} tokens"
            )
        prompt = f"""
        {instruction}
        {full_text}
        """.strip()
        with StreamingIfAllowed(self.llm):  # type: ignore
            summary = Agent.llm_response(self, prompt)
            return summary  # type: ignore

    def justify_response(self) -> None:
        """Show evidence for last response"""
        if self.response is None:
            print("[magenta]No response yet")
            return
        source = self.response.metadata.source
        if len(source) > 0:
            print("[magenta]" + source)
        else:
            print("[magenta]No source found")
