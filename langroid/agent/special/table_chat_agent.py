"""
Agent that supports asking queries about a tabular dataset, internally
represented as a Pandas dataframe. The `TableChatAgent` is configured with a
dataset, which can be a Pandas df, file or URL. The delimiter/separator
is auto-detected. In response to a user query, the Agent's LLM generates a Pandas
expression (involving a dataframe `df`) to answer the query.
The expression is passed via the `pandas_eval` tool/function-call,
which is handled by the Agent's `pandas_eval` method. This method evaluates
the expression and returns the result as a string.
"""

import io
import logging
import sys
from typing import List, Optional, Tuple, no_type_check

import numpy as np
import pandas as pd
from rich.console import Console

import langroid as lr
from langroid.agent import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.table_loader import read_tabular_data
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.constants import DONE, PASS
from langroid.vector_store.base import VectorStoreConfig

logger = logging.getLogger(__name__)

console = Console()

DEFAULT_TABLE_CHAT_SYSTEM_MESSAGE = f"""
You are a savvy data scientist, with expertise in analyzing tabular datasets,
using Python and the Pandas library for dataframe manipulation.
Since you do not have access to the dataframe 'df', you
will need to use the `pandas_eval` tool/function-call to answer my questions.
Here is a summary of the dataframe:
{{summary}}
Do not assume any columns other than those shown.
In the expression you submit to the `pandas_eval` tool/function, 
you are allowed to use the variable 'df' to refer to the dataframe.

Sometimes you may not be able to answer the question in a single call to `pandas_eval`,
so you can use a series of calls to `pandas_eval` to build up the answer. 
For example you may first want to know something about the possible values in a column.

If you receive a null or other unexpected result, see if you have made an assumption
in your code, and try another way, or use `pandas_eval` to explore the dataframe 
before submitting your final code. 

Once you have the answer to the question, possibly after a few steps,
say {DONE} and PRESENT THE ANSWER TO ME; do not just say {DONE}.
If you receive an error message, 
try using the `pandas_eval` tool/function again with the corrected code. 

VERY IMPORTANT: When using the `pandas_eval` tool/function, DO NOT EXPLAIN ANYTHING,
   SIMPLY USE THE TOOL, with the CODE.
Start by asking me what I want to know about the data.
"""


@no_type_check
def dataframe_summary(df: pd.DataFrame) -> str:
    """
    Generate a structured summary for a pandas DataFrame containing numerical
    and categorical values.

    Args:
        df (pd.DataFrame): The input DataFrame to summarize.

    Returns:
        str: A nicely structured and formatted summary string.
    """

    # Column names display
    col_names_str = (
        "COLUMN NAMES:\n" + " ".join([f"'{col}'" for col in df.columns]) + "\n\n"
    )

    # Numerical data summary
    num_summary = df.describe().map(lambda x: "{:.2f}".format(x))
    num_str = "Numerical Column Summary:\n" + num_summary.to_string() + "\n\n"

    # Categorical data summary
    cat_columns = df.select_dtypes(include=[np.object_]).columns
    cat_summary_list = []

    for col in cat_columns:
        unique_values = df[col].unique()
        if len(unique_values) < 10:
            cat_summary_list.append(f"'{col}': {', '.join(map(str, unique_values))}")
        else:
            cat_summary_list.append(f"'{col}': {df[col].nunique()} unique values")

    cat_str = "Categorical Column Summary:\n" + "\n".join(cat_summary_list) + "\n\n"

    # Missing values summary
    nan_summary = df.isnull().sum().rename("missing_values").to_frame()
    nan_str = "Missing Values Column Summary:\n" + nan_summary.to_string() + "\n"

    # Combine the summaries into one structured string
    summary_str = col_names_str + num_str + cat_str + nan_str

    return summary_str


class TableChatAgentConfig(ChatAgentConfig):
    system_message: str = DEFAULT_TABLE_CHAT_SYSTEM_MESSAGE
    user_message: None | str = None
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


class PandasEvalTool(ToolMessage):
    """Tool/function to evaluate a pandas expression involving a dataframe `df`"""

    request: str = "pandas_eval"
    purpose: str = """
            To eval a pandas <expression> on the dataframe 'df' and 
            return the results to answer a question.
            IMPORTANT: the <expression> field should be a valid pandas expression.
            """
    expression: str

    @classmethod
    def examples(cls) -> List["ToolMessage" | Tuple[str, "ToolMessage"]]:
        return [
            cls(expression="df.head()"),
            cls(expression="df[(df['gender'] == 'Male')]['income'].mean()"),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
            Use the `pandas_eval` tool/function to evaluate a pandas expression
            involving the dataframe 'df' to answer the user's question.
            """


class TableChatAgent(ChatAgent):
    """
    Agent for chatting with a collection of documents.
    """

    sent_expression: bool = False

    def __init__(self, config: TableChatAgentConfig):
        if isinstance(config.data, pd.DataFrame):
            df = config.data
        else:
            df = read_tabular_data(config.data, config.separator)

        df.columns = df.columns.str.strip().str.replace(" +", "_", regex=True)

        self.df = df
        summary = dataframe_summary(df)
        config.system_message = config.system_message.format(summary=summary)

        super().__init__(config)
        self.config: TableChatAgentConfig = config

        logger.info(
            f"""TableChatAgent initialized with dataframe of shape {self.df.shape}
            and columns: 
            {self.df.columns}
            """
        )
        # enable the agent to use and handle the PandasEvalTool
        self.enable_message(PandasEvalTool)

    def user_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        response = super().user_response(msg)
        if response is not None and response.content != "":
            self.sent_expression = False
        return response

    def pandas_eval(self, msg: PandasEvalTool) -> str:
        """
        Handle a PandasEvalTool message by evaluating the `expression` field
            and returning the result.
        Args:
            msg (PandasEvalTool): The tool-message to handle.

        Returns:
            str: The result of running the code along with any print output.
        """
        self.sent_expression = True
        exprn = msg.expression
        local_vars = {"df": self.df}
        # Create a string-based I/O stream
        code_out = io.StringIO()

        # Temporarily redirect standard output to our string-based I/O stream
        sys.stdout = code_out

        # Evaluate the last line and get the result
        try:
            eval_result = pd.eval(exprn, local_dict=local_vars)
        except Exception as e:
            eval_result = f"ERROR: {type(e)}: {e}"

        if eval_result is None:
            eval_result = ""

        # Always restore the original standard output
        sys.stdout = sys.__stdout__

        # If df has been modified in-place, save the changes back to self.df
        self.df = local_vars["df"]

        # Get the resulting string from the I/O stream
        print_result = code_out.getvalue() or ""
        sep = "\n" if print_result else ""
        # Combine the print and eval results
        result = f"{print_result}{sep}{eval_result}"
        if result == "":
            result = "No result"
        # Return the result
        return result

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """Handle various LLM deviations"""
        if isinstance(msg, ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            if msg.content.strip() == DONE and self.sent_expression:
                # LLM sent an expression (i.e. used the `pandas_eval` tool)
                # but upon receiving the results, simply said DONE without
                # narrating the result as instructed.
                return """
                    You forgot to PRESENT the answer to the user's query
                    based on the results from `pandas_eval` tool.
                """
            if self.sent_expression:
                # LLM forgot to say DONE
                self.sent_expression = False
                return DONE + " " + PASS
            else:
                # LLM forgot to use the `pandas_eval` tool
                return """
                    You forgot to use the `pandas_eval` tool/function 
                    to find the answer.
                    Try again using the `pandas_eval` tool/function.
                    """
        return None
