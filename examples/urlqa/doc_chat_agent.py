from llmagent.agent.base import AgentConfig
from llmagent.agent.chat_agent import ChatAgent
from llmagent.language_models.base import StreamingIfAllowed, LLMMessage, Role
from llmagent.prompts.templates import SUMMARY_ANSWER_PROMPT_GPT4
from llmagent.utils.output.printing import show_if_debug
from llmagent.utils.configuration import settings
from contextlib import ExitStack
from llmagent.mytypes import Document
from typing import List, Union
from rich import print
from rich.console import Console

console = Console()


class DocChatAgentConfig(AgentConfig):
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

    max_context_tokens: int = 500
    conversation_mode: bool = True


DOC_CHAT_INSTRUCTIONS = """
Your task is to answer questions about various documents.
You will be given various passages from these documents, and asked to answer questions 
about them, or summarize them into coherent answers.
"""


class DocChatAgent(ChatAgent):
    """
    Agent for chatting with a collection of documents.
    """

    def __init__(
        self,
        config: DocChatAgentConfig,
        instructions: str = DOC_CHAT_INSTRUCTIONS,
    ):
        task_messages = [
            LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant"),
            LLMMessage(role=Role.USER, content=instructions),
        ]
        super().__init__(config, task_messages)
        self.original_docs: List[Document] = None
        self.original_docs_length = 0

    def ingest_docs(self, docs: List[Document]) -> int:
        """
        Chunk docs into pieces, map each chunk to vec-embedding, store in vec-db
        """
        self.original_docs = docs
        docs = self.parser.split(docs)
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
        return self.parser.num_tokens(self.doc_string(docs))

    def respond(self, query: str) -> Union[Document, None]:
        if query.startswith("!"):
            # direct query to LLM
            query = query[1:]
            with StreamingIfAllowed(self.llm):
                response = super().respond(query)
            self.update_dialog(query, response.content)
            return response
        if query == "":
            return None
        elif query == "?" and self.response is not None:
            return self.justify_response()
        elif (query.startswith(("summar", "?")) and self.response is None) or (
            query == "??"
        ):
            return self.summarize_docs()
        else:
            return self.answer_from_docs(query)

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
        sources = [d.metadata.get("source") for d in docs]
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

        passages = self.doc_string(passages)
        # Substitute Q and P into the templatized prompt

        final_prompt = SUMMARY_ANSWER_PROMPT_GPT4.format(
            question=f"Question:{question}", extracts=passages
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
            answer_doc = super().respond(final_prompt)
        else:
            answer_doc = super().respond_forget(final_prompt)

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
            metadata={
                "source": "SOURCE: " + sources,
                "cached": answer_doc.metadata.get("cached", False),
            },
        )

    def answer_from_docs(self, query: str) -> Document:
        """Answer query based on docs in vecdb, and conv history"""
        if len(self.dialog) > 0 and not self.config.conversation_mode:
            # In conversation mode, we let self.message_history accumulate
            # and do not need to convert to standalone query
            # (We rely on the LLM to interpret the new query in the context of
            # the message history so far)
            with console.status("[cyan]Converting to stand-alone query...[/cyan]"):
                with StreamingIfAllowed(self.llm, False):
                    query = self.llm.followup_to_standalone(self.dialog, query)
            print(f"[orange2]New query: {query}")

        max_score = 0
        passages = self.original_docs
        # if original docs not too long, no need to look for relevant parts.
        if self.original_docs_length > self.config.max_context_tokens:
            with console.status("[cyan]Searching VecDB for relevant doc passages..."):
                docs_and_scores = self.vecdb.similar_texts_with_scores(
                    query,
                    k=self.config.parsing.n_similar_docs,
                )
            passages: List[Document] = [
                Document(content=d.content, metadata=d.metadata)
                for (d, _) in docs_and_scores
            ]
            max_score = max([s[1] for s in docs_and_scores])

        # if passages not too long, no need to extract relevant verbatim text
        verbatim_texts = passages
        if self.doc_length(passages) > self.config.max_context_tokens:
            with console.status("[cyan]LLM Extracting verbatim passages..."):
                with StreamingIfAllowed(self.llm, False):
                    verbatim_texts: List[Document] = self.llm.get_verbatim_extracts(
                        query, passages
                    )
        with ExitStack() as stack:
            # conditionally use Streaming or rich console context
            cm = (
                StreamingIfAllowed(self.llm)
                if settings.stream
                else (console.status("LLM Generating final answer..."))
            )
            stack.enter_context(cm)
            response = self.get_summary_answer(query, verbatim_texts)
            print(f"[green]relevance = {max_score}")
        # no need to print answer since get_summary_answer -> respond already did
        # if not self.llm.get_stream() or response.metadata["cached"]:
        #     # we would have already printed the response ONLY if
        #     # streaming was enabled AND the response was not cached
        #     print("[green]" + response.content)
        #     print("[magenta]" + response.metadata["source"])
        self.update_dialog(query, response.content)
        self.response = response  # save last response
        return response

    def summarize_docs(self) -> None:
        """Summarize all docs"""
        full_text = "\n\n".join([d.content for d in self.original_docs])
        tot_tokens = self.parser.num_tokens(full_text)
        if tot_tokens < 10000:
            # todo make this a config param
            prompt = f"""
            Give a concise summary of the following text:
            {full_text}
            """.strip()
            with StreamingIfAllowed(self.llm):
                super().respond(prompt)  # raw LLM call
        else:
            print("[red] No summarization for more than 1000 tokens, sorry!")

    def justify_response(self) -> None:
        """Show evidence for last response"""
        source = self.response.metadata["source"]
        if len(source) > 0:
            print("[magenta]" + source)
        else:
            print("[magenta]No source found")
