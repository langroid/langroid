import json
from collections.abc import Generator
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from langroid.agent.special.table_chat_agent import TableChatAgent, TableChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.base import (
    LLMFunctionCall,
    LLMMessage,
    LLMResponse,
    OpenAIToolCall,
    Role,
)
from langroid.language_models.mock_lm import MockLM, MockLMConfig
from langroid.parsing.table_loader import read_tabular_data
from langroid.parsing.utils import closest_string
from langroid.utils.configuration import Settings, set_global
from tests.utils import contains_approx_float

DATA_STRING = """age,gender,income,state,,,,
20,Male,50000,CA,,,
22,Female,55000,TX,,,
25,Male,60000,CA,,,
19,Female,48000,TX,,,
"""


@pytest.fixture
def mock_data_frame_blanks() -> pd.DataFrame:
    return read_tabular_data(StringIO(DATA_STRING))  # type: ignore[arg-type]


@pytest.fixture
def mock_data_file_blanks(tmpdir: Any) -> str:
    file_path = tmpdir.join("mock_data.csv")
    file_path.write(DATA_STRING)
    return str(file_path)


def generate_data(size: int) -> pd.DataFrame:
    # Create a list of states
    states = ["CA", "TX"]

    # Generate random age between 18 and 100
    ages = np.random.randint(18, 50, size)

    # Generate random gender
    genders = np.random.choice(["Male", "Female"], size)

    # Generate random state
    states_col = np.random.choice(states, size)

    # Generate random income between 30000 and 150000
    incomes = np.random.randint(30000, 150001, size)

    # use spaces, mixed cases to make it tricker
    data = {"age ": ages, "GenDer": genders, "State ": states_col, "income": incomes}

    return pd.DataFrame(data)


@pytest.fixture
def mock_dataframe() -> pd.DataFrame:
    data = generate_data(200)  # generate data for 1000 rows
    return data


@pytest.fixture
def mock_data_file(tmp_path: Path) -> Generator[str, None, None]:
    df = generate_data(100)  # generate data for 1000 rows
    file_path = tmp_path / "mock_data.csv"
    df.to_csv(file_path, index=False)
    yield str(file_path)


class _SequenceLLM(MockLM):
    """Return a deterministic sequence of responses for TableChatAgent tests."""

    def __init__(
        self,
        responses: list[tuple[tuple[str, ...], tuple[str, ...], LLMResponse]],
    ) -> None:
        super().__init__(MockLMConfig())
        self.responses = responses
        self.index = 0

    def chat(
        self,
        messages: str | list[LLMMessage],
        *args: Any,
        **kwargs: Any,
    ) -> LLMResponse:
        del args, kwargs
        assert isinstance(messages, list)
        assert self.index < len(self.responses), "unexpected extra model call"
        expected, forbidden, response = self.responses[self.index]
        previous = self.responses[self.index - 1][2] if self.index > 0 else None
        self.index += 1
        self._assert_tool_result_shape(messages[-1], previous)
        last_message = messages[-1].content or ""
        for text in expected:
            assert text in last_message
        for text in forbidden:
            assert text not in last_message
        return response

    @staticmethod
    def _assert_tool_result_shape(
        message: LLMMessage,
        previous: LLMResponse | None,
    ) -> None:
        """Check the tool result is fed back in the shape the API requires.

        Content assertions alone would still pass if the result came back
        with the wrong role or an absent/mismatched `tool_call_id`, which a
        real OpenAI request would reject.
        """
        if previous is None:
            return
        if previous.oai_tool_calls:
            assert message.role == Role.TOOL
            assert message.tool_call_id == previous.oai_tool_calls[0].id
        elif previous.function_call is not None:
            assert message.role == Role.FUNCTION
            assert message.name == previous.function_call.name

    def assert_fully_consumed(self) -> None:
        """Fail if the agent stopped before using every scripted response.

        Without this, a task that terminates early still passes the final
        content assertions, so the later scripted turns (and their
        expected/forbidden checks) would never run.
        """
        assert self.index == len(self.responses), (
            f"only {self.index} of {len(self.responses)} scripted model "
            "responses were consumed"
        )


def _pandas_eval_response(agent: TableChatAgent, expression: str) -> LLMResponse:
    """Build the model response that asks `agent` to eval `expression`.

    The response *shape* is derived from the agent's own config so the test
    always exercises the code path the agent is actually configured for:
    OpenAI tool-calls, legacy function-calls, or a Langroid tool message.
    """
    if agent.config.use_functions_api:
        function_call = LLMFunctionCall(
            name="pandas_eval",
            arguments={"expression": expression},
        )
        if agent.config.use_tools_api:
            return LLMResponse(
                message="",
                oai_tool_calls=[
                    OpenAIToolCall(
                        id="call_pandas_eval",
                        type="function",
                        function=function_call,
                    )
                ],
            )
        return LLMResponse(message="", function_call=function_call)

    return LLMResponse(
        message=json.dumps({"request": "pandas_eval", "expression": expression})
    )


def _average_income_expression_and_answer(
    agent: TableChatAgent,
) -> tuple[str, float]:
    """Build the query expression and expected answer for the average-income test."""
    age_col = closest_string("age", agent.df.columns)
    state_col = closest_string("state", agent.df.columns)
    gender_col = closest_string("gender", agent.df.columns)
    income_col = closest_string("income", agent.df.columns)
    expression = (
        f"df[(df[{age_col!r}] < 40) & "
        f"(df[{state_col!r}] == 'CA') & "
        f"(df[{gender_col!r}] == 'Male')][{income_col!r}].mean()"
    )
    answer = agent.df[
        (agent.df[age_col] < 40)
        & (agent.df[state_col] == "CA")
        & (agent.df[gender_col] == "Male")
    ][income_col].mean()
    return expression, answer


def _test_table_chat_agent(
    fn_api: bool,
    tabular_data: pd.DataFrame | str,
) -> None:
    """
    Test the TableChatAgent with a deterministic model-driven data query.
    """
    agent = TableChatAgent(
        config=TableChatAgentConfig(
            data=tabular_data,
            use_tools=not fn_api,
            use_functions_api=fn_api,
            full_eval=True,  # Allow full evaluation in tests
        )
    )
    expression, answer = _average_income_expression_and_answer(agent)
    llm = _SequenceLLM(
        [
            (
                ("average income",),
                (),
                _pandas_eval_response(agent, expression),
            ),
            (
                (str(answer),),
                ("ERROR",),
                LLMResponse(message=f"DONE The average income is {answer}"),
            ),
        ]
    )
    agent.llm = llm

    task = Task(
        agent,
        name="TableChatAgent",
        interactive=False,
    )
    result = task.run("What is the average income of men under 40 in CA?", turns=6)

    assert result is not None
    assert contains_approx_float(result.content, answer)
    llm.assert_fully_consumed()


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_dataframe(
    test_settings: Settings,
    fn_api: bool,
    mock_dataframe: pd.DataFrame,
) -> None:
    set_global(test_settings)
    _test_table_chat_agent(
        fn_api=fn_api,
        tabular_data=mock_dataframe,
    )


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_file(
    test_settings: Settings,
    fn_api: bool,
    mock_data_file: str,
) -> None:
    set_global(test_settings)
    _test_table_chat_agent(
        fn_api=fn_api,
        tabular_data=mock_data_file,
    )


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_dataframe_blanks(
    test_settings: Settings,
    fn_api: bool,
    mock_data_frame_blanks: pd.DataFrame,
) -> None:
    set_global(test_settings)
    _test_table_chat_agent(
        fn_api=fn_api,
        tabular_data=mock_data_frame_blanks,
    )


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_file_blanks(
    test_settings: Settings,
    fn_api: bool,
    mock_data_file_blanks: str,
) -> None:
    set_global(test_settings)
    _test_table_chat_agent(
        fn_api=fn_api,
        tabular_data=mock_data_file_blanks,
    )


def test_table_chat_agent_assignment_self_correction(test_settings: Settings) -> None:
    """
    Test that TableChatAgent self-corrects when trying to use assignment syntax
    and uses df.assign() instead
    """
    set_global(test_settings)

    # Create a simple dataframe with data that needs cleaning
    df = pd.DataFrame(
        {
            "airline": ["United*", "Delta*", "American*", "Southwest*"],
            "price": [100, 150, 120, 80],
            "destination": ["NYC", "LAX", "CHI", "DEN"],
        }
    )

    agent = TableChatAgent(
        config=TableChatAgentConfig(
            data=df,
            use_tools=True,
            use_functions_api=False,
            full_eval=False,  # Keep security restrictions to test self-correction
        )
    )
    llm = _SequenceLLM(
        [
            (
                ("asterisk",),
                (),
                _pandas_eval_response(
                    agent,
                    "df['airline'] = df['airline'].str.replace('*', '', regex=False)",
                ),
            ),
            (
                ("ERROR", "SyntaxError"),
                (),
                _pandas_eval_response(
                    agent,
                    "df.assign(airline=df['airline'].str.replace('*', ''))",
                ),
            ),
            (
                ("United", "Delta", "American", "Southwest"),
                ("*", "ERROR"),
                LLMResponse(
                    message=(
                        "DONE Removed the asterisks and cleaned data.\n"
                        "airline  price destination\n"
                        "United   100 NYC\n"
                        "Delta    150 LAX\n"
                        "American 120 CHI\n"
                        "Southwest 80 DEN"
                    )
                ),
            ),
        ]
    )
    agent.llm = llm

    task = Task(
        agent,
        name="TableChatAgent",
        interactive=False,
    )

    # Ask to clean the airline column - this should trigger assignment attempt
    result = task.run(
        "Remove the asterisk (*) from all airline names and show me the cleaned data",
        turns=5,
    )

    # Check that the result indicates success
    assert result is not None
    assert "United*" not in result.content
    assert "Delta*" not in result.content
    # The agent successfully cleaned the data (it says so in the message)
    assert "removed" in result.content.lower() and "cleaned" in result.content.lower()
    llm.assert_fully_consumed()


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_url(test_settings: Settings, fn_api: bool) -> None:
    """
    Test the TableChatAgent with a URL of a csv file as data source
    """
    set_global(test_settings)
    URL = "https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv"

    agent = TableChatAgent(
        config=TableChatAgentConfig(
            data=URL,
            use_tools=not fn_api,
            use_functions_api=fn_api,
            full_eval=True,  # Allow full evaluation in tests
        )
    )
    answer = agent.df[agent.df["cotton"] < 500]["poultry"].mean()
    llm = _SequenceLLM(
        [
            (
                ("average poultry",),
                (),
                _pandas_eval_response(
                    agent,
                    "df[df['cotton'] < 500]['poultry'].mean()",
                ),
            ),
            (
                (str(answer),),
                ("ERROR",),
                LLMResponse(message=f"DONE The average poultry export is {answer}"),
            ),
        ]
    )
    agent.llm = llm

    task = Task(
        agent,
        name="TableChatAgent",
        interactive=False,
    )

    # run until LLM says DONE and shows answer,
    # at which point the task loop ends.

    result = task.run(
        """
        What is the average poultry export among states exporting less than 500 units
        of cotton?
        """,
        turns=5,
    )

    assert result is not None
    assert contains_approx_float(result.content, answer)
    llm.assert_fully_consumed()
