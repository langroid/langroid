from llmagent.agent.base import Agent, AgentConfig
from llmagent.mytypes import Document
from typing import List, Union
from halo import Halo
from rich import print


class DocChatAgent(Agent):
    """
    Agent for chatting with a collection of documents.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.original_docs: List[Document] = None

    def ingest_docs(self, docs: List[Document]) -> int:
        """
        Chunk docs into pieces, map each chunk to vec-embedding, store in vec-db
        """
        self.original_docs = docs
        docs = self.parser.split(docs)
        self.vecdb.add_documents(docs)
        return len(docs)

    def respond(self, query: str) -> Union[Document, None]:
        if query.startswith("!"):
            # direct query to LLM
            query = query[1:]
            response = super().respond(query)
            self.update_history(query, response.content)
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

    def answer_from_docs(self, query: str) -> Document:
        """Answer query based on docs in vecdb, and conv history"""
        if len(self.chat_history) > 0:
            with Halo(text="Converting to stand-alone query...", spinner="dots"):
                query = self.llm.followup_to_standalone(self.chat_history, query)
            print(f"[orange2]New query: {query}")

        with Halo(text="Searching VecDB for relevant doc passages...", spinner="dots"):
            docs_and_scores = self.vecdb.similar_texts_with_scores(query, k=4)
        passages: List[Document] = [
            Document(content=d.content, metadata=d.metadata)
            for (d, _) in docs_and_scores
        ]
        max_score = max([s[1] for s in docs_and_scores])
        with Halo(text="LLM Extracting verbatim passages...", spinner="dots"):
            verbatim_texts: List[Document] = self.llm.get_verbatim_extracts(
                query, passages
            )
        with Halo(text="LLM Generating final answer...", spinner="dots"):
            response = self.llm.get_summary_answer(query, verbatim_texts)
        print("[green]relevance = ", max_score)
        print("[green]" + response.content)
        # if len(source) > 0:
        #     print("[cyan]" + source)
        self.update_history(query, response.content)
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
