"""
Variant of pdf-json.py, but uses a Multi-modal LM directly to extract info
without the need for any parsing, i.e. instead of:
     pdf -> markdown -> structured output,
we directly use the multi-modal LM to do:
    pdf -> structured output.
With a sufficiently good multi-modal LM, this can have many advantages:
- faster as it avoids parsing to markdown
- higher-fidelity extraction since markdown rendering is inherently lossy,
  and may lose important layout and other information on the
  relationships among elements.

Instead, directly extracting the info using a multi-modal LM is like
asking the model to directly extract what it "sees".

---

Extract financial items from a financial report document, directly
using a multi-modal LM without intermediate parsing steps.

Run like this: (drop the -m arg to default to GPT4o)

uv run examples/pdf-json-no-parse.py -f examples/extract/um-financial-report.pdf \
    -m gemini/gemini-2.0-pro-exp-03-25

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
from langroid.parsing.file_attachment import FileAttachment
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
            
            When you receive a financial report, your job is to
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
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
    )
    reader_task = make_report_extractor_task(llm_config)
    # If needed, split the PDF into pages, and do the below extraction page by page:
    # from langroid.parsing.pdf_utils import pdf_split_pages
    # pages, tmp_dir = pdf_split_pages(filename)
    # (pages is a list of temp file names -- use each page individually as
    # FileAttachment.from_path(page))
    input = reader_task.agent.create_user_response(
        content=f"""Extract the financial data from the attached file,
        and present the results using the TOOL `{FinReportTool.name()}`.        
        """,
        files=[FileAttachment.from_path(filename)],
    )
    result: FinalResult = reader_task.run(input)
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
