"""
3-agent student assistant system:

- Triage agent: routes questions to the appropriate agent
- Course Agent: answers questions about courses
- Finance Agent: answers questions about finances

Illustrates use of AgentDoneTool, ForwardTool

Run like this (if --model is omitted, it defaults to the GPT-4o model):

python3 examples/basic/multi-agent-triage.py --model groq/llama-3.1-70b-versatile


"""

import os
from typing import Optional

from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.tools.orchestration import (
    AgentDoneTool,
    ForwardTool,
    SendTool,
)
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig
from langroid.parsing.urls import find_urls
from langroid.vector_store.qdrantdb import QdrantDBConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

forward_tool_name = ForwardTool.default_value("request")


class FinanceAnswerTool(lr.ToolMessage):
    request: str = "finance_answer_tool"
    purpose: str = "Present the <answer> to a question about finances"

    answer: str

    def handle(self) -> SendTool:
        return SendTool(to="User", content=self.answer)


class CoursesAnswerTool(lr.ToolMessage):
    request: str = "courses_answer_tool"
    purpose: str = "Present the <answer> to a question about courses"

    answer: str

    def handle(self) -> SendTool:
        return SendTool(to="User", content=self.answer)


def main(model: str = ""):
    class TriageAgent(lr.ChatAgent):
        def init_state(self) -> None:
            # self.expecting_course_answer = False
            # self.expecting_finance_answer = False
            super().init_state()
            self.llm_responded = False

        def user_response(
            self,
            msg: Optional[str | ChatDocument] = None,
        ) -> Optional[ChatDocument]:
            self.llm_responded = False
            return super().user_response(msg)

        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            self.llm_responded = True
            return super().llm_response(message)

        def handle_message_fallback(
            self, msg: str | ChatDocument
        ) -> str | ChatDocument | lr.ToolMessage | None:
            """Handle any non-tool msg"""
            if self.llm_responded:
                self.llm_responded = False
                # LLM generated non-tool msg => send to user
                content = msg.content if isinstance(msg, ChatDocument) else msg
                return SendTool(to="User", content=content)

    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        max_output_tokens=200,
        chat_context_length=16_000,
    )

    triage_agent = TriageAgent(
        lr.ChatAgentConfig(
            name="Triage",
            llm=llm_config,
            system_message=f"""
            You are a helpful assistant to students at a university. 
            
            Students may ask about the following TYPES of questions and you must handle 
            each TYPE as specified below:
            
            - (a) COURSES:
                - use the TOOL: `{forward_tool_name}` to forward the 
                    question to the "Courses" agent
            - (b) FINANCES (student loans, scholarships, tuition, dining plans, etc)
                - use the TOOL: `{forward_tool_name}` to forward the
                    question to the "Finance" agent
            - (c) OTHER questions not specific to the university:
                - attempt to answer these based on your own knowledge, 
                  otherwise admit you don't know.
            
            Start by greeting the user and asking them what they need help with.
            """,
        )
    )
    triage_agent.enable_message(ForwardTool)
    triage_agent.enable_message(
        [FinanceAnswerTool, CoursesAnswerTool],
        use=False,
        handle=True,
    )

    triage_task = lr.Task(triage_agent, interactive=False)

    parsing_config = ParsingConfig(  # modify as needed
        chunk_size=200,  # aim for this many tokens per chunk
        overlap=50,  # overlap between chunks
        max_chunks=10_000,
        # aim to have at least this many chars per chunk when
        # truncating due to punctuation
        min_chunk_chars=50,
        discard_chunk_chars=5,  # discard chunks with fewer than this many chars
        n_neighbor_ids=5,  # num chunk IDs to store on either side of each chunk
        pdf=PdfParsingConfig(
            # NOTE: PDF parsing is extremely challenging, and each library
            # has its own strengths and weaknesses.
            # Try one that works for your use case.
            # See here for available alternatives:
            # https://github.com/langroid/langroid/blob/main/langroid/parsing/parser.py
            library="pymupdf4llm",
        ),
    )

    class CoursesAgent(lr.agent.special.DocChatAgent):
        def llm_response(
            self,
            message: None | str | ChatDocument = None,
        ) -> Optional[ChatDocument]:
            answer = super().llm_response(message)
            if answer is None:
                return None
            return self.create_llm_response(
                tool_messages=[
                    AgentDoneTool(tools=[CoursesAnswerTool(answer=answer.content)])
                ]
            )

    course_url = "https://csd.cmu.edu/cs-and-related-undergraduate-courses"

    courses_agent = CoursesAgent(
        config=lr.agent.special.DocChatAgentConfig(
            name="Courses",
            llm=llm_config,
            doc_paths=[course_url],  # contents will be ingested into vecdb
            vecdb=QdrantDBConfig(
                collection_name="courses",
                replace_collection=True,
                storage_path=".qdrantdb/data/",
            ),
            parsing=parsing_config,
            n_neighbor_chunks=3,
            n_similar_chunks=5,
            n_relevant_chunks=5,
        )
    )

    courses_task = lr.Task(courses_agent, interactive=False, single_round=True)

    finance_url = "https://www.cmu.edu/sfs/tuition/index.html"
    all_finance_urls = find_urls(finance_url, max_links=20, max_depth=3)

    class FinanceAgent(lr.agent.special.DocChatAgent):
        def llm_response(
            self,
            message: None | str | ChatDocument = None,
        ) -> Optional[ChatDocument]:
            answer = super().llm_response(message)
            if answer is None:
                return None
            return self.create_llm_response(
                tool_messages=[
                    AgentDoneTool(tools=[FinanceAnswerTool(answer=answer.content)])
                ]
            )

    finance_agent = FinanceAgent(
        config=lr.agent.special.DocChatAgentConfig(
            name="Finance",
            llm=llm_config,
            doc_paths=all_finance_urls,  # contents will be ingested into vecdb
            vecdb=QdrantDBConfig(
                collection_name="finances",
                replace_collection=True,
                storage_path=".qdrantdb/data/",
            ),
            parsing=parsing_config,
            n_neighbor_chunks=3,
            n_similar_chunks=5,
            n_relevant_chunks=5,
        )
    )

    finance_task = lr.Task(finance_agent, interactive=False, single_round=True)

    triage_task.add_sub_task([courses_task, finance_task])

    triage_task.run()


if __name__ == "__main__":
    Fire(main)
