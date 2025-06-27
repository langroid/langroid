"""
Single Agent for Doc-aware chat with user.

- user asks question
- LLM decides whether to:
    - ask user for follow-up/clarifying information, or
    - retrieve relevant passages from documents, or
    - provide a final answer, if it has enough information from user and documents.

To reduce response latency, in the DocChatAgentConfig,
you can set the `relevance_extractor_config=None`,
to turn off the relevance_extraction step, which uses the LLM
to extract verbatim relevant portions of retrieved chunks.

Run like this:

python3 examples/docqa/doc-aware-chat.py
"""

import os
from typing import Any, Optional

from fire import Fire
from rich import print
from rich.prompt import Prompt

import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.chat_agent import ChatAgent
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.agent.task import Task
from langroid.agent.tools.orchestration import ForwardTool
from langroid.agent.tools.retrieval_tool import RetrievalTool
from langroid.mytypes import Entity
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from langroid.utils.configuration import Settings, set_global

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DocAwareChatAgent(DocChatAgent):
    def __init__(self, config: DocChatAgentConfig):
        super().__init__(config)
        self.enable_message(RetrievalTool)

    def retrieval_tool(self, msg: RetrievalTool) -> str:
        results = super().retrieval_tool(msg)
        return f"""
        
        RELEVANT PASSAGES:
        =====        
        {results}        
        ====
        
        
        BASED on these RELEVANT PASSAGES, DECIDE:
        - If this is sufficient to provide the user a final answer specific to 
            their situation, do so.
        - Otherwise, 
            - ASK the user for more information to get a better understanding
              of their situation or context, OR
            - use this tool again to get more relevant passages.
        """

    def llm_response(
        self,
        message: None | str | ChatDocument = None,
    ) -> Optional[ChatDocument]:
        # override DocChatAgent's default llm_response
        return ChatAgent.llm_response(self, message)

    def handle_message_fallback(self, msg: str | ChatDocument) -> Any:
        # we are here if there is no tool in the msg
        if isinstance(msg, ChatDocument) and msg.metadata.sender == Entity.LLM:
            # Any non-tool message must be meant for user, so forward it to user
            return ForwardTool(agent="User")


def main(
    debug: bool = False,
    nocache: bool = False,
    model: str = lm.OpenAIChatModel.GPT4o,
) -> None:
    llm_config = lm.OpenAIGPTConfig(chat_model=model)
    config = DocChatAgentConfig(
        llm=llm_config,
        n_query_rephrases=0,
        hypothetical_answer=False,
        relevance_extractor_config=None,
        # this turns off standalone-query reformulation; set to False to enable it.
        assistant_mode=True,
        n_neighbor_chunks=2,
        n_similar_chunks=5,
        n_relevant_chunks=5,
        parsing=ParsingConfig(  # modify as needed
            splitter=Splitter.TOKENS,
            chunk_size=100,  # aim for this many tokens per chunk
            n_neighbor_ids=5,
            overlap=20,  # overlap between chunks
            max_chunks=10_000,
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
    )

    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )

    doc_agent = DocAwareChatAgent(config)
    print("[blue]Welcome to the document chatbot!")
    url = Prompt.ask("[blue]Enter the URL of a document")
    doc_agent.ingest_doc_paths([url])

    # For a more flexible/elaborate user doc-ingest dialog, use this:
    # doc_agent.user_docs_ingest_dialog()

    doc_task = Task(
        doc_agent,
        interactive=False,
        name="DocAgent",
        system_message=f"""
        You are a DOCUMENT-AWARE-GUIDE, but you do NOT have direct access to documents.
        Instead you can use the `retrieval_tool` to get passages from the documents
        that are relevant to a certain query or search phrase or topic.
        DO NOT ATTEMPT TO ANSWER THE USER'S QUESTION WITHOUT RETRIEVING RELEVANT
        PASSAGES FROM THE DOCUMENTS. DO NOT use your own existing knowledge!!
        Everything you tell the user MUST be based on the documents.
        
        The user will ask you a question that you will NOT be able to answer
        immediately, because you are MISSING some information about:
            - the user or their context or situation, etc
            - the documents relevant to the question
        
        At each turn you must decide among these possible ACTIONS:
        - use the `{RetrievalTool.name()}` to get more relevant passages from the 
            documents, OR
        - ANSWER the user if you think you have enough information 
            from the user AND the documents, to answer the question.
            
        You can use the `{RetrievalTool.name()}` multiple times to get more 
        relevant passages, if you think the previous ones were not sufficient.
        
        REMEMBER - your goal is to be VERY HELPFUL to the user; this means
        you should NOT OVERWHELM them by throwing them a lot of information and
        ask them to figure things out. Instead, you must GUIDE them 
        by asking SIMPLE QUESTIONS, ONE at at time, and finally provide them
        a clear, DIRECTLY RELEVANT answer that is specific to their situation. 
        """,
    )

    print("[cyan]Enter x or q to quit, or ? for evidence")

    doc_task.run("Can you help me with some questions?")


if __name__ == "__main__":
    Fire(main)
