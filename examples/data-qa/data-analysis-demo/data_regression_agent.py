import langroid as lr
from langroid.agent.special.table_chat_agent import TableChatAgent, TableChatAgentConfig
from messages import DEFAULT_DATA_REGRESSION_SYSTEM_MESSAGE
from utils import SharedData

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd


class DataRegressionConfig(TableChatAgentConfig):
    """
    Configuration for the DataRegressionAgent.
    """

    system_message: str = DEFAULT_DATA_REGRESSION_SYSTEM_MESSAGE
    data: pd.DataFrame | str | SharedData


class DataRegressionTool(lr.ToolMessage):
    """
    Tool message for fitting and interpreting a regression or classification model.
    """

    request: str = "data_regression"
    purpose: str = "Fit and interpret a regression model on the DataFrame"
    model_type: str = "LinearRegression"  # Default model type
    features: list
    target: str

    @classmethod
    def examples(cls) -> list["DataRegressionTool"]:
        return [
            cls(
                model_type="LinearRegression",
                features=["incidents_85_99", "fatalities_85_99"],
                target="fatalities_00_14",
            ),
            cls(
                model_type="LogisticRegression",
                features=["incidents_00_14", "fatalities_00_14"],
                target="safety_rating",
            ),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        Use this tool to fit and interpret a regression or classification model on the DataFrame `df`.
        Specify the model type, features, and target variable.
        Model types: LinearRegression, LogisticRegression, DecisionTreeRegressor, DecisionTreeClassifier.
        Features: list of column names to use as predictors.
        Target: column name to predict.
        """


class DataRegressionAgent(TableChatAgent):
    """
    Agent for fitting and interpreting regression or classification models on a Pandas DataFrame.
    """

    def __init__(self, config: DataRegressionConfig):
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

    def data_regression(self, msg: DataRegressionTool) -> str:
        """
        Fit and interpret a regression or classification model on the DataFrame `df`.
        """
        self.sent_expression = True
        model_type = msg.model_type
        features = msg.features
        target = msg.target

        model_map = {
            "LinearRegression": (LinearRegression, "regression"),
            "LogisticRegression": (LogisticRegression, "classification"),
            "DecisionTreeRegressor": (DecisionTreeRegressor, "regression"),
            "DecisionTreeClassifier": (DecisionTreeClassifier, "classification"),
        }

        if model_type not in model_map:
            supported = ", ".join(model_map.keys())
            return f"ERROR: Unsupported model type '{model_type}'. Supported types: {supported}."

        model_class, task_type = model_map[model_type]

        try:
            missing = [col for col in features + [target] if col not in self.df.columns]
            if missing:
                return f"ERROR: Missing columns in DataFrame: {', '.join(missing)}"

            X = self.df[features]
            y = self.df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task_type == "regression":
                score = r2_score(y_test, y_pred)
                metric_name = "R^2 Score"
            else:  # classification
                score = accuracy_score(y_test, y_pred)
                metric_name = "Accuracy Score"

            coefs = getattr(model, "coef_", None)
            coef_str = f"Coefficients: {coefs}\n" if coefs is not None else ""

            results_df = X_test.copy()
            results_df["Actual"] = y_test.loc[X_test.index].values
            results_df["Predicted"] = y_pred

            self.shared_data.df_results = results_df  # Store results in shared data
            self.shared_data.task_type = task_type
            self.shared_data.metric_name = metric_name
            self.shared_data.score = score

            result = (
                f"Model: {model_type}\n"
                f"Features: {', '.join(features)}\n"
                f"Target: {target}\n"
                f"{metric_name}: {score:.4f}\n"
                f"{coef_str}"
                f"Actual vs Predicted results stored for visualization.\n"
                f"Results DataFrame:\n{results_df.head()}\n"
            )

        except Exception as e:
            result = f"ERROR: {type(e).__name__}: {e}"

        return result

    def handle_message_fallback(
        self, msg: str | lr.ChatDocument
    ) -> str | lr.ChatDocument | None:
        """
        Handle LLM deviations and guide the agent to use the correct tool or provide the expected output.
        """
        if isinstance(msg, lr.ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            if msg.content.strip() == "done" and self.sent_expression:
                return (
                    "You forgot to PRESENT the answer to the user's query "
                    "based on the results from `data_regression` tool."
                )
            if self.sent_expression:
                self.sent_expression = False
                return None
            else:
                return (
                    "You forgot to use the `data_regression` tool/function "
                    "to find the answer. "
                    "Try again using the `data_regression` tool/function."
                )
        return None
