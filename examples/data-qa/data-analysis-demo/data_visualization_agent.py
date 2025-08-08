import langroid as lr
from langroid.agent.special.table_chat_agent import TableChatAgent, TableChatAgentConfig
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from langroid.agent import ChatDocument
from messages import DEFAULT_DATA_VISUALIZATION_SYSTEM_MESSAGE
import pandas as pd
from utils import SharedData
import io
import sys
from typing import Any, Dict, Optional, Union


class DataVisualizationConfig(TableChatAgentConfig):
    """
    Configuration for the DataVisualizationAgent.
    """

    system_message: str = DEFAULT_DATA_VISUALIZATION_SYSTEM_MESSAGE
    data: pd.DataFrame | str | SharedData


class ResultsHelperTool(lr.ToolMessage):
    """
    Tool message for accessing the results DataFrame.
    This tool allows the agent to access the info about results DataFrame for model predictions.
    """

    request: str = "results_helper"
    purpose: str = (
        "Access the information about the results DataFrame for model predictions"
    )

    @classmethod
    def examples(cls) -> list["ResultsHelperTool"]:
        return [
            cls(),
        ]

    @classmethod
    def instructions(cls):
        return """
        Use this tool to access information about the results DataFrame 'results_df'.
        You can access the DataFrame itself, the task type, metric name, and score.
        """


class DataVisualizationTool(lr.ToolMessage):
    """
    Tool message for visualizing data from a Pandas DataFrame using Matplotlib.
    """

    request: str = "data_eval"
    purpose: str = "Visualize data from a Pandas DataFrame using Matplotlib"
    expression: str

    @classmethod
    def examples(cls) -> list["DataVisualizationTool"]:
        return [
            cls(expression="df.plot(kind='bar', x='airline', y='incidents_85_99')"),
            cls(expression="df['incidents_00_14'].plot(kind='line')"),
            cls(
                expression="df.plot(kind='scatter', x='fatalities_85_99', y='fatalities_00_14')"
            ),
            cls(
                expression="results_df.plot(x='Actual', y='Predicted', kind='scatter')"
            ),
            cls(expression="results_df['Predicted'].value_counts().plot(kind='bar')"),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        Use this tool to visualize data from the DataFrame `df`, or the results DataFrame `results_df`.
        You can use any valid Pandas plotting method, such as `plot`, `hist`, `scatter`, etc.
        Ensure that your expression returns a plot object.
        Remember to use `df.assign(...)` for any modifications to the DataFrame.
        The `results_df` DataFrame contains model results and can be used for plotting model predictions, and will contain 'Actual' and 'Predicted' columns after regression or classification tasks.
        """


class DataVisualizationAgent(TableChatAgent):
    """
    Agent for visualizing data from a Pandas DataFrame using Matplotlib.
    """

    def __init__(self, config: DataVisualizationConfig):
        if isinstance(config.data, SharedData):
            self.shared_data = config.data
            config.data = self.shared_data.df
        super().__init__(config)
        if not hasattr(self, "shared_data"):
            self.shared_data = SharedData(config.data)

    @property
    def df(self) -> pd.DataFrame | str:
        return self.shared_data.df  # Always latest version

    @df.setter
    def df(self, value):
        self.shared_data.df = value

    def data_eval(self, msg: DataVisualizationTool) -> str:
        """
        Evaluate the provided expression using the DataFrame `df` or `results_df`.
        This method is called when the user requests data visualization.
        """
        self.sent_expression: bool = True
        exprn: str = msg.expression
        vars: Dict[str, Any] = {
            "df": self.df,
            "results_df": self.shared_data.df_results,
        }

        code_out = io.StringIO()
        sys.stdout = code_out

        fig: Optional[Figure] = None
        try:
            plt.clf()  # Clear the current figure to avoid overlap
            plt.close()
            code = compile(exprn, "<string>", "eval")
            eval_result = eval(code, vars, {})
            fig = plt.gcf()  # Get the current figure
            plt.show(block=False)
            plt.pause(0.1)  # Allow the plot to render without blocking
        except Exception as e:
            eval_result = f"ERROR: {type(e).__name__}: {e}"

        if eval_result is None:
            eval_result = ""

        sys.stdout = sys.__stdout__  # Restore original stdout

        if fig:
            return "Plot generated successfully."
        elif isinstance(eval_result, str) and eval_result.startswith("ERROR"):
            return eval_result
        elif eval_result is None:
            return "Expression evaluated (no output)."
        else:
            return str(eval_result)

    def results_helper(self, msg: ResultsHelperTool) -> str:
        """
        Access the results DataFrame and related information.
        This method is called when the user requests access to the results DataFrame.
        """
        self.sent_expression: bool = True
        if hasattr(self.shared_data, "df_results"):
            return f"Results DataFrame:\n{self.shared_data.df_results.head()}\nTask Type: {self.shared_data.task_type}\nMetric Name: {self.shared_data.metric_name}\nScore: {self.shared_data.score}"
        else:
            return "No results DataFrame available."

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """
        Handle LLM deviations and guide the agent to use the correct tool or provide the expected output.
        """
        if isinstance(msg, ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            if msg.content.strip() == "done" and self.sent_expression:
                return (
                    "You forgot to PRESENT the answer to the user's query "
                    "based on the results from `data_eval` tool."
                )
            if self.sent_expression:
                self.sent_expression = False
                return None
            else:
                return (
                    "You forgot to use the `data_eval` or the `results_helper` tool/function "
                    "to find the answer. "
                    "Try again using the `data_eval` or the `results_helper` tool/function."
                )
        return None
