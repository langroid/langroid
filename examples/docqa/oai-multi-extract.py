"""
Two-agent chat with Retrieval-augmented LLM + function-call/tool.
ExtractorAgent (has no access to docs) is tasked with extracting structured
information from a commercial lease document, and must present the terms in
a specific nested JSON format.
DocAgent (has access to the lease) helps answer questions about the lease.
Repeat: WriterAgent --Question--> DocAgent --> Answer

Example:
python3 examples/docqa/chat_multi_extract.py

Use -f option to use OpenAI function calling API instead of Langroid tool.
"""

import json
import os

import typer
from rich import print

import langroid as lr
from langroid.agent.openai_assistant import (
    AssistantTool,
    OpenAIAssistant,
    OpenAIAssistantConfig,
)
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.pydantic_v1 import BaseModel
from langroid.utils.constants import DONE, NO_ANSWER
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()
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


class LeaseMessage(ToolMessage):
    """Tool/function to use to present details about a commercial lease"""

    request: str = "lease_info"
    purpose: str = "Collect information about a Commercial Lease."
    terms: Lease

    def handle(self):
        """Handle this tool-message when the LLM emits it.
        Under the hood, this method is transplated into the OpenAIAssistant class
        as a method with name `lease_info`.
        """
        print(f"DONE! Successfully extracted Lease Info:" f"{self.terms}")
        return DONE + " " + json.dumps(self.terms.dict())


@app.command()
def chat() -> None:
    retriever_cfg = OpenAIAssistantConfig(
        name="LeaseRetriever",
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4o),
        system_message="Answer questions based on the documents provided.",
    )

    retriever_agent = OpenAIAssistant(retriever_cfg)
    retriever_agent.add_assistant_tools([AssistantTool(type="retrieval")])
    retriever_agent.add_assistant_files(["examples/docqa/lease.txt"])

    retriever_task = Task(
        retriever_agent,
        interactive=False,
        done_if_response=[lr.Entity.LLM],
        done_if_no_response=[lr.Entity.LLM],
    )

    extractor_cfg = OpenAIAssistantConfig(
        name="LeaseExtractor",
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4o),
        system_message=f"""
        You have to collect information about a Commercial Lease from a 
        lease contract which you don't have access to. You need to ask
        questions ONE BY ONE to get this information. 
        Once you have all the REQUIRED fields, 
        you have to present it to me using the `lease_info` 
        function/tool (fill in {NO_ANSWER} for slots that you are unable to fill).
        """,
    )
    extractor_agent = OpenAIAssistant(extractor_cfg)
    extractor_agent.enable_message(LeaseMessage, include_defaults=False)

    extractor_task = Task(extractor_agent, interactive=False)
    extractor_task.add_sub_task(retriever_task)
    extractor_task.run()


if __name__ == "__main__":
    app()
