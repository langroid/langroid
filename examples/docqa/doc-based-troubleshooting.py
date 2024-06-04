"""
2-agent doc-aware conversation,
different from standard question -> answer RAG flow.

User indicates some type of problem,
TroubleShooter Agent engages in conversation with User,
guiding them toward a solution.
At each step, Troubleshooter Agent can either address:
- DocAgent (who has access to docs/manuals) for info, or
- User, to ask follow-up questions about the problem

python3 examples/docqa/doc-based-troubleshooting.py

"""

from typing import Optional

from rich import print
from rich.prompt import Prompt
import os

from langroid import ChatDocument
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
import langroid as lr
import langroid.language_models as lm
from langroid.mytypes import Entity
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.utils.configuration import set_global, Settings
from langroid.utils.constants import DONE, NO_ANSWER
from fire import Fire

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DocAgent(DocChatAgent):
    def llm_response(
        self,
        query: None | str | ChatDocument = None,
    ) -> Optional[ChatDocument]:
        # Augment the response
        results = super().llm_response(query).content
        return self.create_llm_response(
            f"""
            Summary answer FROM DocAgent:
            
            ===
            {results}
            ===
            
            Look at the results above. These might be too much for the user to read.
            DECIDE whether you want to:
            - Ask the User a SINGLE follow-up question (could be MultipleChoice,
                where they need to select a numbered choice)
                to get more info about their situation or context, OR
            - Ask the DocAgent for more information, if you think you need more info.
            - Provide the User a FINAL answer, if you think you have enough information 
               from the User AND the Documents
               
            IMPORTANT: Do NOT simply give the User a list of options -- 
                you must HELP the user by asking them FOLLOWUP questions
                about their situation and GUIDE them to a SPECIFIC, 
                DIRECTLY RELEVANT answer. 
                You CAN give the user a MULTIPLE CHOICE question, telling them 
                to pick a number (or choice-letter) from the list.
                
            REMEMBER - NEVER ask the DocAgent or User MULTIPLE questions at a time,
                always ask ONE question at a time;
                 if asking the USER, it CAN be a MULTIPLE CHOICE question.
            """
        )


def main(
    debug: bool = False,
    nocache: bool = False,
    model: str = lm.OpenAIChatModel.GPT4o,
) -> None:
    vecdb_config = lr.vector_store.QdrantDBConfig(
        storage_path=".qdrant/doc-aware/",
        replace_collection=False,
        cloud=False,
    )

    llm_config = lm.OpenAIGPTConfig(chat_model=model)
    config = DocChatAgentConfig(
        llm=llm_config,
        vecdb=vecdb_config,
        n_query_rephrases=0,
        hypothetical_answer=False,
        assistant_mode=True,
        n_neighbor_chunks=2,
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
            n_similar_docs=5,
            # NOTE: PDF parsing is extremely challenging, each library has its own
            # strengths and weaknesses. Try one that works for your use case.
            pdf=PdfParsingConfig(
                # alternatives: "unstructured", "pdfplumber", "fitz"
                library="pdfplumber",
            ),
        ),
    )

    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )

    doc_agent = DocAgent(config)
    print("[blue]Welcome to the document chatbot!")
    doc_agent.user_docs_ingest_dialog()
    print("[cyan]Enter x or q to quit, or ? for evidence")
    doc_task = Task(
        doc_agent,
        interactive=False,
        name="DocAgent",
        done_if_no_response=[Entity.LLM],  # done if null response from LLM
        done_if_response=[Entity.LLM],  # done if non-null response from LLM
    )

    guide_agent = ChatAgent(
        ChatAgentConfig(
            name="GuideAgent",
            llm=llm_config,
            vecdb=None,
        )
    )
    # MyRecipientTool = RecipientTool.create(
    #     recipients=["DocAgent", "User"], default="User"
    # )
    # guide_agent.enable_message(MyRecipientTool)
    guide_task = Task(
        guide_agent,
        interactive=False,
        system_message=f"""
        You are a TROUBLESHOOTER, who wants to help a User with their PROBLEM.
        
        Your task is to GUIDE them STEP BY STEP toward a specific
        resolution that is DIRECTLY RELEVANT to their specific problem.
        
        IMPORTANT: Your guidance/help should ONLY be based on certain DOCUMENTS
          and NOT on your existing knowledge. NEVER answer based on your own knowledge,
          ALWAYS refer to the documents.
          However you do NOT have direct access to the docs, but you have an assistant
          named DocAgent, who DOES have access to the documents.
          
        Since you could be talking to TWO people, in order to CLARIFY who you are
        addressing, you MUST ALWAYS EXPLICITLY ADDRESS either the 
        "User" or the "DocAgent" using @User or @DocAgent, respectively.
        
        You must THINK like this at each step after receiving a question from the User:
        
        (I NEVER WANT TO Overwhelm DocAgent or User with TOO MANY QUESTIONS,
        so I will ALWAYS ask ONE question at a time)
        
        - I must first find out more about this topic from DocAgent, 
            let me address DocAgent to get more information.
        - I got some info from DocAgent, let me now ask the User a follow-up question
            to get ONE SPECIFIC piece of information about their situation.
        - I need to get MORE info from DocAgent, let me ask DocAgent for more info.
        - DocAgent said {NO_ANSWER}!!, Let me try asking a different way.
        - I have a bit more info, now let me ask the User a further follow-up question,
            to get ONE SPECIFIC piece of information about their situation.
        - I need more info from user, let me ask the User a follow-up question,
            to get ANOTHER SPECIFIC piece of information about their situation.
        ...[and so on]...
        - Now I have ALL the info I need from BOTH the User and DocAgent,
            so I can provide the User a DIRECTLY RELEVANT answer,
            so I will say {DONE}, followed by the answer.   
            
        IMPORTANT: When giving the User a list of choices, always show them
            a NUMBERED list of choices.     
            
        I REPEAT -- NEVER use your OWN KNOWLEDGE. ALWAYS RELY ON the Documents
        from DocAgent.     
        """,
    )
    guide_task.add_sub_task(doc_task)

    while True:
        query = Prompt.ask("[blue]How can I help?")
        if query in ["x", "q"]:
            break
        guide_task.run(query)


if __name__ == "__main__":
    Fire(main)
