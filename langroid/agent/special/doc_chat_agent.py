"""
Agent that supports asking queries about a set of documents, using
retrieval-augmented generation (RAG).

Functionality includes:
- summarizing a document, with a custom instruction; see `summarize_docs`
- asking a question about a document; see `answer_from_docs`

Note: to use the sentence-transformer embeddings, you must install
langroid with the [hf-embeddings] extra, e.g.:

pip install "langroid[hf-embeddings]"

"""
import logging
from contextlib import ExitStack
from typing import List, Optional, Tuple, no_type_check

from rich import print
from rich.console import Console

from langroid.agent.base import Agent
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.language_models.base import StreamingIfAllowed
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.mytypes import DocMetaData, Document, Entity
from langroid.parsing.parser import Parser, ParsingConfig, PdfParsingConfig, Splitter
from langroid.parsing.repo_loader import RepoLoader
from langroid.parsing.search import (
    find_closest_matches_with_bm25,
    find_fuzzy_matches_in_docs,
    preprocess_text,
)
from langroid.parsing.url_loader import URLLoader
from langroid.parsing.urls import get_urls_and_paths
from langroid.parsing.utils import batched
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
    # In assistant mode, DocChatAgent receives questions from another Agent,
    # and those will already be in stand-alone form, so in this mode
    # there is no need to convert them to stand-alone form.
    assistant_mode: bool = False
    # Use LLM to generate hypothetical answer A to the query Q,
    # and use the embed(A) to find similar chunks in vecdb.
    # Referred to as HyDE in the paper:
    # https://arxiv.org/pdf/2212.10496.pdf
    # It is False by default; its benefits depends on the context.
    hypothetical_answer: bool = False
    n_query_rephrases: int = 0
    use_fuzzy_match: bool = True
    use_bm25_search: bool = True
    cross_encoder_reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embed_batch_size: int = 500  # get embedding of at most this many at a time
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
        chunk_size=1000,  # aim for this many tokens per chunk
        overlap=100,  # overlap between chunks
        max_chunks=10_000,
        # aim to have at least this many chars per chunk when
        # truncating due to punctuation
        min_chunk_chars=200,
        discard_chunk_chars=5,  # discard chunks with fewer than this many chars
        n_similar_docs=3,
        pdf=PdfParsingConfig(
            # NOTE: PDF parsing is extremely challenging, and each library
            # has its own strengths and weaknesses.
            # Try one that works for your use case.
            # or "haystack", "unstructured", "pdfplumber", "fitz", "pypdf"
            library="pdfplumber",
        ),
    )
    from langroid.embedding_models.models import SentenceTransformerEmbeddingsConfig

    hf_embed_config = SentenceTransformerEmbeddingsConfig(
        model_type="sentence-transformer",
        model_name="BAAI/bge-large-en-v1.5",
    )

    oai_embed_config = OpenAIEmbeddingsConfig(
        model_type="openai",
        model_name="text-embedding-ada-002",
        dims=1536,
    )

    vecdb: VectorStoreConfig = QdrantDBConfig(
        collection_name=None,
        storage_path=".qdrant/data/",
        embedding=hf_embed_config,
    )
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        completion_model=OpenAIChatModel.GPT4,
        timeout=40,
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
        self.chunked_docs: None | List[Document] = None
        self.chunked_docs_clean: None | List[Document] = None
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
            # we must be using a previously defined collection
            # But let's get all the chunked docs so we can
            # do keyword and other non-vector searches
            if self.vecdb is None:
                raise ValueError("VecDB not set")
            self.chunked_docs = self.vecdb.get_all_documents()
            self.chunked_docs_clean = [
                Document(content=preprocess_text(d.content), metadata=d.metadata)
                for d in self.chunked_docs
            ]
            return
        urls, paths = get_urls_and_paths(self.config.doc_paths)
        docs: List[Document] = []
        parser = Parser(self.config.parsing)
        if len(urls) > 0:
            loader = URLLoader(urls=urls, parser=parser)
            docs = loader.load()
        if len(paths) > 0:
            for p in paths:
                path_docs = RepoLoader.get_documents(p, parser=parser)
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
        self.chunked_docs = docs
        self.chunked_docs_clean = [
            Document(content=preprocess_text(d.content), metadata=d.metadata)
            for d in self.chunked_docs
        ]
        if self.vecdb is None:
            raise ValueError("VecDB not set")
        # add embeddings in batches, to stay under limit of embeddings API
        batches = list(batched(docs, self.config.embed_batch_size))
        for batch in batches:
            self.vecdb.add_documents(batch)
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

    def llm_hypothetical_answer(self, query: str) -> str:
        if self.llm is None:
            raise ValueError("LLM not set")
        with console.status("[cyan]LLM generating hypothetical answer..."):
            with StreamingIfAllowed(self.llm, False):
                # TODO: provide an easy way to
                # Adjust this prompt depending on context.
                answer = self.llm_response_forget(
                    f"""
                    Give an ideal answer to the following query, 
                    in up to 3 sentences. Do not explain yourself, 
                    and do not apologize, just show 
                    a good possible answer, even if you do not have any information.
                    Preface your answer with "HYPOTHETICAL ANSWER: "
                    
                    QUERY: {query}
                    """
                ).content
        return answer

    def llm_rephrase_query(self, query: str) -> List[str]:
        if self.llm is None:
            raise ValueError("LLM not set")
        with console.status("[cyan]LLM generating rephrases of query..."):
            with StreamingIfAllowed(self.llm, False):
                rephrases = self.llm_response_forget(
                    f"""
                        Rephrase the following query in {self.config.n_query_rephrases}
                        different equivalent ways, separate them with 2 newlines.
                        QUERY: {query}
                        """
                ).content.split("\n\n")
        return rephrases

    def get_similar_chunks_bm25(
        self, query: str, multiple: int
    ) -> List[Tuple[Document, float]]:
        # find similar docs using bm25 similarity:
        # these may sometimes be more likely to contain a relevant verbatim extract
        with console.status("[cyan]Searching for similar chunks using bm25..."):
            if self.chunked_docs is None:
                raise ValueError("No chunked docs")
            if self.chunked_docs_clean is None:
                raise ValueError("No cleaned chunked docs")
            docs_scores = find_closest_matches_with_bm25(
                self.chunked_docs,
                self.chunked_docs_clean,  # already pre-processed!
                query,
                k=self.config.parsing.n_similar_docs * multiple,
            )
        return docs_scores

    def get_fuzzy_matches(self, query: str, multiple: int) -> List[Document]:
        # find similar docs using fuzzy matching:
        # these may sometimes be more likely to contain a relevant verbatim extract
        with console.status("[cyan]Finding fuzzy matches in chunks..."):
            if self.chunked_docs is None:
                raise ValueError("No chunked docs")
            fuzzy_match_docs = find_fuzzy_matches_in_docs(
                query,
                self.chunked_docs,
                k=self.config.parsing.n_similar_docs * multiple,
                words_before=1000,
                words_after=1000,
            )
        return fuzzy_match_docs

    def rerank_with_cross_encoder(
        self, query: str, passages: List[Document]
    ) -> List[Document]:
        with console.status("[cyan]Re-ranking retrieved chunks using cross-encoder..."):
            if self.chunked_docs is None:
                raise ValueError("No chunked docs")
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    """
                    To use cross-encoder re-ranking, you must install
                    langroid with the [hf-embeddings] extra, e.g.:
                    pip install "langroid[hf-embeddings]"
                    """
                )

            model = CrossEncoder(self.config.cross_encoder_reranking_model)
            scores = model.predict([(query, p.content) for p in passages])
            # get top k scoring passages
            sorted_pairs = sorted(
                zip(scores, passages),
                key=lambda x: x[0],
                reverse=True,
            )
            passages = [
                d for _, d in sorted_pairs[: self.config.parsing.n_similar_docs]
            ]
        return passages

    @no_type_check
    def get_relevant_extracts(self, query: str) -> Tuple[str, List[Document]]:
        """
        Get list of extracts from doc-chunks relevant to answering a query.
        These are the stages (some optional based on config):
        - use LLM to convert query to stand-alone query
        - optionally rephrase query to use below
        - optionally generate hypothetical answer (HyDE) to use below.
        - get relevant doc-chunks, via:
            - vector-embedding distance, from vecdb
            - bm25-ranking (keyword similarity)
            - fuzzy matching (keyword similarity)
        - re-ranking of doc-chunks using cross-encoder, pick top k
        - use LLM to get relevant extracts from doc-chunks

        Args:
            query (str): query to search for

        Returns:
            query (str): stand-alone version of input query
            List[Document]: list of relevant extracts

        """
        if len(self.dialog) > 0 and not self.config.assistant_mode:
            # Regardless of whether we are in conversation mode or not,
            # for relevant doc/chunk extraction, we must convert the query
            # to a standalone query to get more relevant results.
            with console.status("[cyan]Converting to stand-alone query...[/cyan]"):
                with StreamingIfAllowed(self.llm, False):
                    query = self.llm.followup_to_standalone(self.dialog, query)
            print(f"[orange2]New query: {query}")

        # if we are using cross-encoder reranking, we can retrieve more docs
        # during retrieval, and leave it to the cross-encoder re-ranking
        # to whittle down to self.config.parsing.n_similar_docs
        retrieval_multiple = 1 if self.config.cross_encoder_reranking_model == "" else 3
        queries = [query]
        if self.config.hypothetical_answer:
            answer = self.llm_hypothetical_answer(query)
            queries = [query, answer]

        if self.config.n_query_rephrases > 0:
            rephrases = self.llm_rephrase_query(query)
            queries += rephrases

        with console.status("[cyan]Searching VecDB for relevant doc passages..."):
            docs_and_scores = []
            for q in queries:
                docs_and_scores += self.vecdb.similar_texts_with_scores(
                    q,
                    k=self.config.parsing.n_similar_docs * retrieval_multiple,
                )
        # keep only docs with unique d.id()
        id2doc_score = {d.id(): (d, s) for d, s in docs_and_scores}
        docs_and_scores = list(id2doc_score.values())

        passages = [
            Document(content=d.content, metadata=d.metadata)
            for (d, _) in docs_and_scores
        ]

        if self.config.use_bm25_search:
            docs_scores = self.get_similar_chunks_bm25(query, retrieval_multiple)
            passages += [d for (d, _) in docs_scores]

        if self.config.use_fuzzy_match:
            fuzzy_match_docs = self.get_fuzzy_matches(query, retrieval_multiple)
            passages += fuzzy_match_docs

        # keep unique passages
        id2passage = {p.id(): p for p in passages}
        passages = list(id2passage.values())

        # now passages can potentially have a lot of doc chunks,
        # so we re-rank them using a cross-encoder scoring model
        # https://www.sbert.net/examples/applications/retrieve_rerank
        if self.config.cross_encoder_reranking_model != "":
            passages = self.rerank_with_cross_encoder(query, passages)

        if len(passages) == 0:
            return query, []

        with console.status("[cyan]LLM Extracting verbatim passages..."):
            with StreamingIfAllowed(self.llm, False):
                # these are async calls, one per passage; turn off streaming
                extracts = self.llm.get_verbatim_extracts(query, passages)
                extracts = [e for e in extracts if e.content != NO_ANSWER]

        return query, extracts

    @no_type_check
    def answer_from_docs(self, query: str) -> Document:
        """
        Answer query based on relevant docs from the VecDB

        Args:
            query (str): query to answer

        Returns:
            Document: answer
        """
        response = Document(
            content=NO_ANSWER,
            metadata=DocMetaData(
                source="None",
            ),
        )
        # query may be updated to a stand-alone version
        query, extracts = self.get_relevant_extracts(query)
        if len(extracts) == 0:
            return response
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
        if self.llm is None:
            raise ValueError("LLM not set")
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
        MAX_INPUT_TOKENS = (
            self.llm.completion_context_length()
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
        with StreamingIfAllowed(self.llm):
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
