"""
Agent that supports asking queries about a tabular dataset, internally
represented as a Pandas dataframe. The `TableChatAgent` is configured with a
dataset, which can be a Pandas df, file or URL. The delimiter/separator
is auto-detected. In response to a user query, the Agent's LLM generates Pandas
code to answer the query. The code is passed via the `run_code` tool/function-call,
which is handled by the Agent's `run_code` method. This method executes/evaluates
the code and returns the result as a string.
"""
import io
import logging
import sys

import pandas as pd
from rich.console import Console

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.table_loader import read_tabular_data
from langroid.prompts.prompts_config import PromptsConfig
from langroid.vector_store.base import VectorStoreConfig

logger = logging.getLogger(__name__)

console = Console()

DEFAULT_TABLE_CHAT_SYSTEM_MESSAGE = """
You are a savvy data scientist, with expertise in analyzing tabular dataset,
using Python and the Pandas library for dataframe manipulation.
Since you do not have access to the dataframe 'df', you
will need to use the `run_code` tool/function-call to answer the question.
The columns in the dataframe are:
{columns}
Do not assume any columns other than those shown.
"""


class TableChatAgentConfig(ChatAgentConfig):
    system_message: str = DEFAULT_TABLE_CHAT_SYSTEM_MESSAGE
    user_message: None | str = None
    max_context_tokens: int = 1000
    cache: bool = True  # cache results
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    data: str | pd.DataFrame  # data file, URL, or DataFrame
    separator: None | str = None  # separator for data file
    vecdb: None | VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        completion_model=OpenAIChatModel.GPT4,
    )
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


class RunCodeTool(ToolMessage):
    request: str = "run_code"
    purpose: str = """
            To run <code> on the dataframe 'df' and 
            return the results to answer a question.
            """
    code: str


class TableChatAgent(ChatAgent):
    """
    Agent for chatting with a collection of documents.
    """

    def __init__(self, config: TableChatAgentConfig):
        super().__init__(config)
        self.config: TableChatAgentConfig = config
        if isinstance(config.data, pd.DataFrame):
            self.df = config.data
        else:
            self.df = read_tabular_data(config.data, config.separator)

        logger.info(
            f"""TableChatAgent initialized with dataframe of shape {self.df.shape}
            and columns: 
            {self.df.columns}
            """
        )
        self.config.system_message = self.config.system_message.format(
            columns=", ".join(self.df.columns)
        )
        # enable the agent to use and handle the RunCodeTool
        self.enable_message(RunCodeTool)

    def run_code(self, msg: RunCodeTool) -> str:
        """
        Handle a RunCodeTool message by running the code and returning the result.
        Args:
            msg (RunCodeTool): The tool-message to handle.

        Returns:
            str: The result of running the code along with any print output.
        """
        code = msg.code
        # Create a dictionary that maps 'df' to the actual DataFrame
        local_vars = {"df": self.df}

        # Create a string-based I/O stream
        code_out = io.StringIO()

        # Temporarily redirect standard output to our string-based I/O stream
        sys.stdout = code_out

        # Split the code into lines
        lines = code.strip().split("\n")

        # Run all lines as statements except for the last one
        for line in lines[:-1]:
            exec(line, {}, local_vars)

        # Evaluate the last line and get the result
        eval_result = eval(lines[-1], {}, local_vars) or ""

        # Always restore the original standard output
        sys.stdout = sys.__stdout__

        # If df has been modified in-place, save the changes back to self.df
        self.df = local_vars["df"]

        # Get the resulting string from the I/O stream
        print_result = code_out.getvalue() or ""
        sep = "\n" if print_result else ""
        # Combine the print and eval results
        result = f"{print_result}{sep}{eval_result}"

        # Return the result
        return result
