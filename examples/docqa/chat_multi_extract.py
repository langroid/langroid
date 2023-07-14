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
import typer
from rich import print
from pydantic import BaseModel, BaseSettings
from typing import List
import json

from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging
from langroid.utils.constants import NO_ANSWER

app = typer.Typer()

setup_colored_logging()


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
    request: str = "lease_info"
    purpose: str = """
        Collect information about a Commercial Lease.
        """
    terms: Lease
    result: str = ""

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


class LeaseExtractorAgent(ChatAgent):
    def __init__(self, config: ChatAgentConfig):
        super().__init__(config)

    def lease_info(self, message: LeaseMessage) -> str:
        print(
            f"""
        DONE! Successfully extracted Lease Info:
        {message.terms}
        """
        )
        return json.dumps(message.terms.dict())


class CLIOptions(BaseSettings):
    debug: bool = False
    fn_api: bool = False
    cache: bool = True


def chat(opts: CLIOptions) -> None:
    doc_agent = DocChatAgent(DocChatAgentConfig())
    doc_agent.vecdb.set_collection("docqa-chat-multi-extract", replace=True)
    print("[blue]Welcome to the real-estate info-extractor!")
    doc_agent.config.doc_paths = [
        "examples/docqa/lease.txt",
    ]
    doc_agent.ingest()
    doc_task = Task(
        doc_agent,
        name="DocAgent",
        llm_delegate=False,
        single_round=True,
        system_message="""You are an expert on Commercial Leases. 
        You will receive various questions about a Commercial 
        Lease contract, and your job is to answer them concisely in at most 2 sentences.
        """,
    )

    lease_extractor_agent = LeaseExtractorAgent(
        ChatAgentConfig(
            llm=OpenAIGPTConfig(),
            use_functions_api=opts.fn_api,
            use_tools=not opts.fn_api,
        )
    )
    lease_extractor_agent.enable_message(
        LeaseMessage,
        use=True,
        handle=True,
        force=False,
    )
    lease_task = Task(
        lease_extractor_agent,
        name="LeaseExtractorAgent",
        llm_delegate=True,
        single_round=False,
        system_message=f"""
        You have to collect some information about a Commercial Lease, but you do not 
        have access to the lease itself. 
        You can ask me questions about the lease, ONE AT A TIME, I will answer each 
        question. You only need to collect info corresponding to the fields in this 
        example:
        {LeaseMessage.usage_example()}
        If I am unable to answer your question initially, try asking me 
        differently. If I am still unable to answer after 3 tries, fill in 
        {NO_ANSWER} for that field.
        When you have collected this info, present it to me using the 
        'lease_info' function/tool.
        """,
    )
    lease_task.add_sub_task(doc_task)
    lease_task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    fn_api: bool = typer.Option(False, "--fn_api", "-f", help="use functions api"),
) -> None:
    cli_opts = CLIOptions(
        fn_api=fn_api,
    )
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )
    chat(cli_opts)


if __name__ == "__main__":
    app()
