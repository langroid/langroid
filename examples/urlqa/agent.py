from llmagent.agent.base import Agent, Message
from llmagent.agent.config import AgentConfig
from llmagent.mytypes import Document
from typing import List
from halo import Halo
from rich import print

class DocChatAgent(Agent):

    def ingest_docs(self, docs: List[Document]):
        """
        Chunk docs into pieces, map each chunk to vec-embedding, store in vec-db
        """
        docs = self.parser.split(docs)
        self.vecdb.add_documents(docs)

    def respond(self, query:str):
        if query == "":
            return 
        if len(self.chat_history) > 0:
            with Halo(text="Converting to stand-alone query...",  spinner="dots"):
                query = self.llm.followup_to_standalone(llm, self.chat_history, query)
            print(f"[orange2]New query: {query}")

        with Halo(text="Searching VecDB for relevant doc passages...",
                  spinner="dots"):
            docs_and_scores = self.vecdb.similar_texts_with_scores(query, k=4)
        passages: List[Document] = [
            Document(content=d.content, metadata=d.metadata)
            for (d, _) in docs_and_scores
        ]
        max_score = max([s[1] for s in docs_and_scores])
        with Halo(text="LLM Extracting verbatim passages...",  spinner="dots"):
            verbatim_texts: List[Document] = self.llm.get_verbatim_extracts(
                query, passages
            )
        with Halo(text="LLM Generating final answer...", spinner="dots"):
            response = self.llm.get_summary_answer(query, verbatim_texts)
        print("[green]relevance = ", max_score)
        print("[green]" + response.content)
        source = response.metadata["source"]
        if len(source) > 0:
            print("[orange]" + source)
        self.update_history(query, response.content)
        self.response = response # save last response

    def summarize_docs(self):
        """Summarize all docs"""
        print("[red] Summaries not ready, coming soon!")

    def justify_response(self):
        """Show evidence for last response"""
        source = self.response.metadata["source"]
        if len(source) > 0:
            print("[orange]" + source)
        else:
            print("[orange]No source found")





