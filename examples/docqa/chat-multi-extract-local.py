"""
Extract structured info from a commercial lease document,
using multiple agents, powered by a weaker/local LLM, combining tools/functions and RAG.

TASK:
Given a lease document, generate the lease terms, organized into
 a nested JSON structure defined by the Pydantic class Lease

Solution with Langroid Agents and tools:
1. QuestionGeneratorAgent: Lease JSON Spec -> list of questions to ask
2. InterrogatorAgent: For each question, generate 2 variants of the question,
   so we use total 3 variants per question, joined together, to increase
   the likelihood of getting an answer from the DocAgent (RAG).
3. DocAgent (has access to the lease) -> answer one question using RAG
3. LeasePresenterAgent: List of (question, answer) pairs ->
        organized into specified Lease JSON structure

This works with a local mistral-instruct-v0.2 model. High level instructions:

1. Spin up an OpenAI-compatible API for the model using oobabooga:
(See https://github.com/oobabooga/text-generation-webui for details)
python server.py --api --model mistral-7b-instruct-v0.2.Q8_0.gguf --verbose

This should load the model and start the API server at http://127.0.0.1:5000/v1
(It won't explicitly say "/v1" but you have to include it in the URL below)

2. Run this script

python3 examples/docqa/chat-multi-extract-local.py
        -m local/127.0.0.1:5000/v1//mistral-instruct-v0.2

Notice we append the "//mistral-instruct-v0.2" at the end to ensure
the chats are correctly formatted using the corresponding chat template.


Other optional script args:
-m <model_name_with_formatter_after//>, e.g. -m local/localhost:8000//mistral-instruct-v0.2
-nc to disable cache retrieval
-d to enable debug mode: see prompts, agent msgs etc.
"""

import typer
from rich import print
from pydantic import BaseModel
from typing import List, Optional
import json
import os

import langroid.language_models as lm
from langroid.agent import ChatDocument
from langroid.mytypes import Entity
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.parsing.parser import ParsingConfig
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.constants import NO_ANSWER, DONE
from langroid.utils.pydantic_utils import flatten_pydantic_model

app = typer.Typer()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LeasePeriod(BaseModel):
    start_date: str
    end_date: str


class LeaseFinancials(BaseModel):
    monthly_rent: str
    deposit: str


class Lease(BaseModel):
    """
    Various lease terms.
    Nested fields to make this more interesting/realistic
    """

    period: LeasePeriod
    financials: LeaseFinancials
    address: str


class QuestionsTool(ToolMessage):
    request: str = "questions_tool"
    purpose: str = """
    To present a list of <questions> to ask, to fill a desired JSON structure.
    """
    questions: List[str]


class QuestionGeneratorAgent(ChatAgent):
    questions_list: List[str] = []

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        if isinstance(msg, ChatDocument) and msg.metadata.sender == Entity.LLM:
            return """
            You forgot to present the information in JSON format 
            according to the `questions` tool specification,
            or you may have used a wrong tool name or field name.
            Try again.
            """
        return None

    def questions_tool(self, msg: QuestionsTool) -> str:
        # flatten the class and get field names
        flat_mdl = flatten_pydantic_model(Lease)
        fields = flat_mdl.__fields__.keys()
        # field names can be like "period__start_date", so we only want the last part
        fields = [f.split("__")[-1] for f in fields]
        if len(msg.questions) < len(fields):
            return f"""
            ERROR: Expected {len(fields)} questions, but only got {len(msg.questions)}.
            See what you may have missed and try again.
            Hint: the required fields are {fields}
            """
        elif len(msg.questions) > len(fields):
            return f"""
            ERROR: Expected {len(fields)} questions, but got {len(msg.questions)}.
            You generated an extra question. Try again.
            Hint: the required fields are {fields}
            """
        else:
            self.questions_list = msg.questions
            return DONE + json.dumps(msg.questions)


class MyDocChatAgent(DocChatAgent):
    def llm_response(
        self,
        query: None | str | ChatDocument = None,
    ) -> Optional[ChatDocument]:
        """
        Override the default LLM response to return the full document,
        to forget the last round in conversation, so we don't clutter
        the chat history with all previous questions
        (Assume questions don't depend on past ones, as is the case here,
        since we are extracting separate pieces of info from docs)
        """
        n_msgs = len(self.message_history)
        response = super().llm_response(query)
        # If there is a response, then we will have two additional
        # messages in the message history, i.e. the user message and the
        # assistant response. We want to (carefully) remove these two messages.
        self.message_history.pop() if len(self.message_history) > n_msgs else None
        self.message_history.pop() if len(self.message_history) > n_msgs else None
        return response


class LeasePresenterAgent(ChatAgent):
    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """Handle scenario where Agent failed to present the Lease JSON"""
        if isinstance(msg, ChatDocument) and msg.metadata.sender == Entity.LLM:
            return """
            You either forgot to present the information in the JSON format
            required in `lease_info` JSON specification,
            or you may have used the wrong name of the tool or fields.
            Try again.
            """
        return None


class LeaseMessage(ToolMessage):
    """Tool/function to use to present details about a commercial lease"""

    request: str = "lease_info"
    purpose: str = "To present the <terms> of a Commercial lease."
    terms: Lease

    def handle(self) -> str:
        print(
            f"""
        DONE! Successfully extracted Lease Info:
        {self.terms}
        """
        )
        return DONE + " " + json.dumps(self.terms.dict())

    @classmethod
    def examples(cls) -> List["LeaseMessage"]:
        return [
            cls(
                terms=Lease(
                    period=LeasePeriod(start_date="2021-01-01", end_date="2021-12-31"),
                    financials=LeaseFinancials(monthly_rent="$1000", deposit="$1000"),
                    address="123 Main St, San Francisco, CA 94105",
                ),
                result="",
            ),
            cls(
                terms=Lease(
                    period=LeasePeriod(start_date="2021-04-01", end_date="2022-04-28"),
                    financials=LeaseFinancials(monthly_rent="$2000", deposit="$2000"),
                    address="456 Main St, San Francisco, CA 94111",
                ),
                result="",
            ),
        ]


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
    llm_cfg = OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4_TURBO,
        chat_context_length=32000,  # adjust based on model
        timeout=120,
    )

    ### (1) QUESTION GENERATOR
    question_generator_agent = QuestionGeneratorAgent(
        ChatAgentConfig(
            llm=llm_cfg,
            vecdb=None,
            system_message="""
            See the `lease_info` JSON structure below. 
            Your ONLY task is to generate 
            QUESTIONS corresponding to each field in the `lease_info` JSON,
            and present these to me using the `questions_tool` in JSON format.
            Pay attention to the format and fields in the `questions_tool` JSON.
            """,
        )
    )
    question_generator_agent.enable_message(LeaseMessage)
    question_generator_agent.enable_message(QuestionsTool)
    question_generator_task = Task(
        question_generator_agent,
        name="QuestionGeneratorAgent",
        interactive=False,
    )

    ### (2) RAG AGENT: try to answer a given question based on documents
    doc_agent = MyDocChatAgent(
        DocChatAgentConfig(
            llm=llm_cfg,
            assistant_mode=True,
            n_neighbor_chunks=2,
            parsing=ParsingConfig(
                chunk_size=50,
                overlap=10,
                n_similar_docs=3,
                n_neighbor_ids=4,
            ),
            cross_encoder_reranking_model="",
        )
    )
    doc_agent.vecdb.set_collection("docqa-chat-multi-extract", replace=True)
    doc_agent.ingest_doc_paths(["examples/docqa/lease.txt"])
    print("[blue]Welcome to the real-estate info-extractor!")
    doc_task = Task(
        doc_agent,
        name="DocAgent",
        done_if_no_response=[Entity.LLM],  # done if null response from LLM
        done_if_response=[Entity.LLM],  # done if non-null response from LLM
        system_message="""You are an expert on Commercial Leases. 
        You will receive a question about a Commercial 
        Lease contract, and your job is to answer concisely in at most 2 sentences.
        """,
    )

    ### (3) Interrogator: persists in getting an answer for a SINGLE question
    #       from the RAG agent
    interrogator = ChatAgent(
        ChatAgentConfig(
            llm=llm_cfg,
            vecdb=None,
            system_message="""
            You are an expert on Commercial leases and their terms. 
            User will send you a QUESTION about such a lease.
            Your ONLY job is to reply with TWO VARIATIONS of the QUESTION,
            and say NOTHING ELSE.
            """,
        )
    )
    interrogator_task = Task(
        interrogator,
        name="Interrogator",
        restart=True,  # clear agent msg history
        interactive=False,
        single_round=True,
    )

    ### (4) LEASE PRESENTER: Given full list of question-answer pairs,
    #       organize them into the Lease JSON structure
    lease_presenter = LeasePresenterAgent(
        ChatAgentConfig(
            llm=llm_cfg,
            vecdb=None,
        )
    )
    lease_presenter.enable_message(LeaseMessage)

    lease_presenter_task = Task(
        lease_presenter,
        name="LeasePresenter",
        interactive=False,  # set to True to slow it down (hit enter to progress)
        system_message=f"""
        The user will give you a list of Questions and Answers 
        about a commercial lease.
        
        Organize this information into the `lease_info` JSON structure specified below,
        and present it to me. 
        For fields where the answer is NOT KNOWN, fill in "UNKNOWN" as the value.
        """,
    )

    ### (5) Use the agents/tasks

    # Lease info JSON -> Questions
    question_generator_task.run()
    questions = question_generator_agent.questions_list
    print(f"found {len(questions)} questions! Now generating answers...")

    # Questions -> Answers using RAG
    answers = []
    for q in questions:
        # use 3 variants of the question at the same time,
        # to increase likelihood of getting an answer
        q_variants = interrogator_task.run(q).content
        result = doc_task.run(q + "\n" + q_variants)
        answer = result.content or NO_ANSWER
        answers.append(answer)
    print(f"got {len(answers)} answers!")

    q2a = dict(zip(questions, answers))
    print(f"q2a: {q2a}")
    questions_answers = "\n\n".join(
        f"Question: {q}:\nAnswer: {a}" for q, a in q2a.items()
    )
    # Questions + Answers -> organized into nested Lease Info JSON
    lease_presenter_task.run(questions_answers)


if __name__ == "__main__":
    app()
