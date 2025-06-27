"""
Extract financial items from a financial report document, in two stages:

1. use Langroid's PDF Parser with `marker` library to
   extract content from (pdf) report in markdown format
2. use Langroid Agent equipped with a structured output tool to extract structured data

Run like this: (drop the -m arg to default to GPT4o)

uv run examples/pdf-json.py -f examples/extract/um-financial-report.pdf \
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
from rich.console import Console
from rich.table import Table

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import ResultTool
from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import LLMPdfParserConfig, ParsingConfig, PdfParsingConfig
from langroid.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
Agent to extract structured data from a markdown formatted table.
Typically, this markdown formatted content would come from using a parser
that extracted markdown from a pdf report, e.g. using a Langroid PDF Parser.
"""


class FinancialData(BaseModel):
    item: str = Field(..., description="Name of the specific financial item")
    year: int = Field(..., description="year of the data item")
    value: str = Field(..., description="value of the item, empty if not applicable")


class FinalResult(ResultTool):
    data: List[FinancialData]


class FinReportTool(lr.ToolMessage):
    request: str = "fin_report_tool"
    purpose: str = """
    To present the <financial_info> 
    extracted from a financial report, in a structured format.
    """

    data: List[FinancialData]

    def handle(self) -> FinalResult:
        return FinalResult(data=self.data)


class ReportExtractorConfig(lr.ChatAgentConfig):
    # placeholder
    name: str = "ReportExtractor"


class ReportReader(lr.ChatAgent):
    def __init__(self, config: ReportExtractorConfig):
        super().__init__(config)
        self.config: ReportExtractorConfig = config
        self.enable_message(FinReportTool)


def make_report_extractor_task(
    llm_config: lm.OpenAIGPTConfig = lm.OpenAIGPTConfig(
        chat_model=lm.OpenAIChatModel.GPT4o,
    )
):
    agent = ReportReader(
        ReportExtractorConfig(
            llm=llm_config,
            handle_llm_no_tool=f"You FORGOT to use the TOOL `{FinReportTool.name()}`",
            system_message=f"""
            You are an expert at financial reports containing various values
            over multiple years, and especially, extracting the 
            financial item, year and value.
            
            When you receive a markdown-formatted financial report, your job is to
            extract the financial data from the report and present it in a structured
            form using the TOOL `{FinReportTool.name()}`.
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
    parsing_config = ParsingConfig(
        pdf=PdfParsingConfig(
            library="llm-pdf-parser",
            llm_parser_config=LLMPdfParserConfig(
                model_name="gemini/gemini-2.0-flash",
                split_on_page=True,
                max_tokens=7000,
                requests_per_minute=5,
            ),
        )
    )
    pdf_parser = DocumentParser.create(filename, config=parsing_config)
    content = pdf_parser.get_doc().content
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
    )
    reader_task = make_report_extractor_task(llm_config)
    result: FinalResult = reader_task.run(content)
    if result is None:
        logger.warning("No Financial items found.")
        return
    else:
        data = result.data
        logger.warning(f"Found {len(data)} financial items.")

        # Print structured data in a nice table format
        console = Console()
        table = Table(title="Financial Results")

        # Add fixed columns based on PatientData model
        table.add_column("Item", style="cyan")
        table.add_column("Year", style="cyan")
        table.add_column("Value", style="cyan")
        # Add rows from PatientData objects
        for pd in data:
            table.add_row(
                pd.item,
                str(pd.year),
                str(pd.value),
            )

        console.print(table)


if __name__ == "__main__":
    Fire(main)
