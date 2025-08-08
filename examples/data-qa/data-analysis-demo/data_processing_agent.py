import langroid as lr
from langroid.agent.special.table_chat_agent import (
    TableChatAgent,
    TableChatAgentConfig,
    PandasEvalTool,
)
from langroid.agent import ChatDocument
from langroid.utils.constants import DONE
from utils import SharedData
from messages import DEFAULT_DATA_PREPROCESSING_SYSTEM_MESSAGE

import pandas as pd
import io
import sys


class DataProcessingConfig(TableChatAgentConfig):
    """
    Configuration for the DataProcessingAgent.
    """

    system_message: str = DEFAULT_DATA_PREPROCESSING_SYSTEM_MESSAGE
    data: pd.DataFrame | str | SharedData


class PandasProcessTool(PandasEvalTool):
    """
    Tool for processing Pandas DataFrames.
    Allows the agent to evaluate Pandas expressions on the DataFrame.
    """

    request: str = "pandas_process"
    purpose: str = (
        "Evaluate a pandas <expression> on the dataframe 'df' to analyze or modify the data, "
        "then return the results to answer a question. "
        "IMPORTANT: the <expression> field should be a valid pandas expression."
    )
    expression: str

    @classmethod
    def examples(cls):
        return [
            cls(expression="df.head()"),
            cls(expression="df.assign(airline=df['airline'].str.replace('*', ''))"),
        ]

    @classmethod
    def instructions(cls) -> str:
        return (
            "Use the `pandas_process` tool/function to evaluate a pandas expression "
            "involving the dataframe 'df' to answer the user's question or process the data."
        )


class DataProcessingAgent(TableChatAgent):
    """
    Agent for processing and cleaning data in a Pandas DataFrame.
    """

    def __init__(self, config: DataProcessingConfig):
        if isinstance(config.data, SharedData):
            self.shared_data = config.data
            config.data = self.shared_data.df
        super().__init__(config)
        if not hasattr(self, "shared_data"):
            self.shared_data = SharedData(config.data)

    @property
    def df(self):
        return self.shared_data.df  # Always latest version

    @df.setter
    def df(self, value):
        self.shared_data.df = value

    def pandas_process(self, msg: PandasProcessTool) -> str:
        """
        Evaluate a Pandas expression on the DataFrame and return the result.
        """
        self.sent_expression = True
        exprn = msg.expression
        vars = {"df": self.df}
        code_out = io.StringIO()
        sys.stdout = code_out

        try:
            if not self.config.full_eval:
                exprn = super().sanitize_command(exprn)
            code = compile(exprn, "<calc>", "eval")
            eval_result = eval(code, vars, {})
        except Exception as e:
            eval_result = f"ERROR: {type(e).__name__}: {e}"

        if eval_result is None:
            eval_result = ""

        sys.stdout = sys.__stdout__

        # If df has been modified in-place, save the changes back to self.df
        if isinstance(eval_result, pd.DataFrame) and "assign" in exprn:
            vars["df"] = eval_result

        self.df = vars["df"]
        self.shared_data.df = self.df

        print_result = code_out.getvalue() or ""
        sep = "\n" if print_result else ""
        result = f"{print_result}{sep}{eval_result}"
        if result == "":
            result = "No result"
        return result

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """
        Handle LLM deviations and guide the agent to use the correct tool or provide the expected output.
        """
        if isinstance(msg, ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            if msg.content.strip() == DONE and self.sent_expression:
                return (
                    "You forgot to PRESENT the answer to the user's query "
                    "based on the results from `pandas_process` tool."
                )
            if self.sent_expression:
                self.sent_expression = False
                return None
            else:
                return (
                    "You forgot to use the `pandas_process` tool/function "
                    "to find the answer. "
                    "Try again using the `pandas_process` tool/function."
                )
        return None
