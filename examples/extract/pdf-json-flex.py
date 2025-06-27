"""
Extract an arbitrary json structure from a pdf via markdown.

1. use Langroid's PDF Parser with `marker` library to
   extract content from (pdf) report in markdown format
2. use Langroid Agent equipped with a structured output tool to extract structured data

Run like this: (drop the -m arg to default to GPT4o)

uv run examples/pdf-json-flex.py -f examples/extract/um-financial-report.pdf \
    -m gemini/gemini-2.0-pro-exp-02-05

NOTES:
- this script uses the `marker` library for parsing PDF content,
and to get that to work with langroid, install langroid with the `marker-pdf` extra,
e.g.
uv pip install "langroid[marker-pdf]"
pip install "langroid[marker-pdf]"

- The structured extracted is very simple, consisting of 3 fields: item, year, and value.
  You may need to adapt it to your needs.
"""

import logging
import os
from typing import List

from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import ResultTool
from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig
from langroid.pydantic_v1 import BaseModel

logger = logging.getLogger(__name__)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
Agent to extract structured data from a markdown formatted table.
Typically, this markdown formatted content would come from using a parser
that extracted markdown from a pdf report, e.g. using a Langroid PDF Parser.
"""


class JsonData(BaseModel):
    """Data model for arbitrary nested JSON-like structures.

    This model allows for storing any valid JSON data format, including nested objects,
    arrays, primitives, etc.

    """

    class Config:
        extra = "allow"  # Allow any extra fields


class FinalResult(ResultTool):
    data: List[JsonData]


class JsonExtractTool(lr.ToolMessage):
    request: str = "json_extract_tool"
    purpose: str = "To present the <json_data> extracted from a document."

    json_data: List[JsonData]

    def handle(self) -> FinalResult:
        return FinalResult(data=self.json_data)


class JsonExtractorConfig(lr.ChatAgentConfig):
    # placeholder
    name: str = "JsonExtractor"


class JsonExtractor(lr.ChatAgent):
    def __init__(self, config: JsonExtractorConfig):
        super().__init__(config)
        self.config: JsonExtractorConfig = config
        self.enable_message(JsonExtractTool)


def display_json_data(data: List[JsonData]) -> None:
    """Display structured JSON data using Rich's JSON pretty printer.

    Args:
        data: List of JsonData objects to display
    """
    from rich.console import Console
    from rich.json import JSON
    from rich.panel import Panel

    console = Console()

    if not data:
        console.print("[bold red]No data found[/bold red]")
        return

    for i, item in enumerate(data):
        # Convert JsonData to dict, filtering out internal attributes
        item_dict = {k: v for k, v in item.__dict__.items() if not k.startswith("__")}
        # Create a panel for each data item with pretty-printed JSON inside
        json_str = JSON.from_data(item_dict)
        console.print(Panel(json_str, title=f"Item {i+1}", border_style="cyan"))

        # Add some spacing between items
        if i < len(data) - 1:
            console.print("")


def make_json_extractor_task(
    llm_config: lm.OpenAIGPTConfig = lm.OpenAIGPTConfig(
        chat_model=lm.OpenAIChatModel.GPT4o,
    )
):
    agent = JsonExtractor(
        JsonExtractorConfig(
            llm=llm_config,
            handle_llm_no_tool=f"You FORGOT to use the TOOL `{JsonExtractTool.name()}`",
            system_message=f"""
            You are an expert at creating (possibly nested) JSON structures
            from markdown documents.
            
            When you receive a markdown-formatted document, your job is to
            extract the data from the document and present it in a structured
            form using the TOOL `{JsonExtractTool.name()}`.
            """,
        )
    )
    # create task specialized to return FinalResult value
    task = lr.Task(agent, interactive=False, single_round=False)[FinalResult]
    return task


def main(
    filename: str,
    model: str = "",
) -> None:
    #    from langroid.parsing.parser import LLMPdfParserConfig
    parsing_config = ParsingConfig(
        pdf=PdfParsingConfig(
            library="marker",  # see alternative below
            # library="llm-pdf-parser",
            # llm_parser_config=LLMPdfParserConfig(
            #     model_name="gpt-4.1", #"gemini/gemini-2.5-pro-exp-03-25",
            #     split_on_page=False,
            #     max_tokens=7000,
            #     timeout=300,
            # )
        )
    )
    pdf_parser = DocumentParser.create(filename, config=parsing_config)
    content = pdf_parser.get_doc().content
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
    )
    extractor_task = make_json_extractor_task(llm_config)
    result: FinalResult = extractor_task.run(content)
    if result is None:
        logger.warning("No JSON content found.")
        return
    else:
        data = result.data
        logger.warning(f"Found {len(data)} items.")
        display_json_data(data)


if __name__ == "__main__":
    Fire(main)
