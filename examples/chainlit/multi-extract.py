"""
Two-agent chat with Retrieval-augmented LLM + function-call/tool.
ExtractorAgent (has no access to docs) is tasked with extracting structured
information from a commercial lease document, and must present the terms in
a specific nested JSON format.
This agent generates questions corresponding to each field in the JSON format,
and the RAG-enabled DocAgent (has access to the lease) answers the  questions.

This is a Chainlit version of examples/docqa/chat_multi_extract.py.

Example:
chainlit run examples/chainlit/multi-extract.py

This uses GPT4-turbo by default, but works very well with the `dolphin-mixtral`
local LLM, which you can specify in the llm_config below
using `chat_model = "ollama/dolphin-mixtral:latest"`,
provided you've already spun it up with ollama:
```
ollama run dolphin-mixtral
```

See here for more on setting up LLMs to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/

The challenging parts in this script are agent-to-agent delegation, and the extractor
agent planning out a sequence of questions to ask the doc agent, and finally presenting
the collected information in a structured format to the user using a Tool/Function-call.
The `dolphin-mixtral` model seems to handle this pretty well, however weaker models
may not be able to handle this.

"""

from rich import print
from pydantic import BaseModel
from typing import List
import json
import os

import langroid as lr
import chainlit as cl
import langroid.language_models as lm
from langroid.mytypes import Entity
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.parsing.parser import ParsingConfig
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.constants import NO_ANSWER

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
        return "DONE \n" + json.dumps(message.terms.dict(), indent=4)


@cl.on_chat_start
async def main(
    debug: bool = False,
    model: str = "",  # or "ollama/dolphin-mixtral:latest"
    nocache: bool = False,
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )
    llm_cfg = OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4_TURBO,
        chat_context_length=16_000,  # adjust based on model
        temperature=0,
        timeout=45,
    )
    doc_agent = DocChatAgent(
        DocChatAgentConfig(
            llm=llm_cfg,
            parsing=ParsingConfig(
                chunk_size=300,
                overlap=50,
                n_similar_docs=3,
            ),
            cross_encoder_reranking_model="",
        )
    )
    doc_agent.vecdb.set_collection("docqa-chat-multi-extract", replace=True)
    print("[blue]Welcome to the real-estate info-extractor!")
    doc_agent.config.doc_paths = [
        "examples/docqa/lease.txt",
    ]
    doc_agent.ingest()
    doc_task = Task(
        doc_agent,
        name="DocAgent",
        done_if_no_response=[Entity.LLM],  # done if null response from LLM
        done_if_response=[Entity.LLM],  # done if non-null response from LLM
        system_message="""You are an expert on Commercial Leases. 
        You will receive various questions about a Commercial 
        Lease contract, along with some excerpts from the Lease.
        Your job is to answer them concisely in at most 2 sentences.
        """,
    )

    lease_extractor_agent = LeaseExtractorAgent(
        ChatAgentConfig(
            llm=llm_cfg,
            vecdb=None,
        )
    )
    lease_extractor_agent.enable_message(LeaseMessage)

    lease_task = Task(
        lease_extractor_agent,
        name="LeaseExtractor",
        interactive=False,  # set to True to slow it down (hit enter to progress)
        system_message=f"""
        You have to collect some SPECIFIC STRUCTURED information 
        about a Commercial Lease, as specified in the `lease_info` function/tool. 
        But you do not have access to the lease itself. 
        You can ask me questions about the lease, ONE AT A TIME, I will answer each 
        question. You only need to collect info to fill the fields in the 
        `field_info` function/tool. 
        If I am unable to answer your question initially, try asking me 
        differently. If I am still unable to answer after 3 tries, fill in 
        {NO_ANSWER} for that field.
        When you have collected this info, present it to me using the 
        'lease_info' function/tool.
        DO NOT USE THIS Function/tool UNTIL YOU HAVE ASKED QUESTIONS 
        TO FILL IN ALL THE FIELDS.
        
        Start by asking me for the start date of the lease.
        """,
    )
    lease_task.add_sub_task(doc_task)
    # The below line is essentially the ONLY change to make
    # to the original script on which this is based.
    lr.ChainlitTaskCallbacks(lease_task)
    # DocChatAgent does not have an async llm_response method,
    # so we must use task.run() instead of task.run_async(),
    # but fortunately we can wrap it in a cl.make_async() call
    await cl.make_async(lease_task.run)()
