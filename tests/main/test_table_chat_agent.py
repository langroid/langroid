from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from langroid.agent.special.table_chat_agent import TableChatAgent, TableChatAgentConfig
from langroid.agent.task import Task
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
def mock_data_frame_blanks():
    return read_tabular_data(StringIO(DATA_STRING))


@pytest.fixture
def mock_data_file_blanks(tmpdir):
    file_path = tmpdir.join("mock_data.csv")
    file_path.write(DATA_STRING)
    return str(file_path)


def generate_data(size: int) -> str:
    # Create a list of states
    states = ["CA", "TX"]

    # Generate random age between 18 and 100
    ages = np.random.randint(18, 101, size)

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
    data = generate_data(100)  # generate data for 1000 rows
    return data


@pytest.fixture
def mock_data_file(tmp_path: Path) -> str:
    df = generate_data(100)  # generate data for 1000 rows
    file_path = tmp_path / "mock_data.csv"
    df.to_csv(file_path, index=False)
    yield str(file_path)


def _test_table_chat_agent(
    fn_api: bool,
    tabular_data: pd.DataFrame | str,
) -> None:
    """
    Test the TableChatAgent with a file as data source
    """
    agent = TableChatAgent(
        config=TableChatAgentConfig(
            data=tabular_data,
            use_tools=not fn_api,
            use_functions_api=fn_api,
        )
    )

    task = Task(
        agent,
        name="TableChatAgent",
        default_human_response="",  # avoid waiting for human response
        only_user_quits_root=False,
        llm_delegate=True,
        single_round=False,
    )

    # run until LLM says DONE and shows answer,
    # at which point the task loop ends.
    result = task.run("What is the average income of men under 40 in CA?", turns=5)
    age_col = closest_string("age", agent.df.columns)
    state_col = closest_string("state", agent.df.columns)
    gender_col = closest_string("gender", agent.df.columns)
    income_col = closest_string("income", agent.df.columns)
    answer = agent.df[
        (agent.df[age_col] < 40)
        & (agent.df[state_col] == "CA")
        & (agent.df[gender_col] == "Male")
    ][income_col].mean()

    assert contains_approx_float(result.content, answer)


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_dataframe(test_settings: Settings, fn_api, mock_dataframe):
    set_global(test_settings)
    _test_table_chat_agent(
        fn_api=fn_api,
        tabular_data=mock_dataframe,
    )


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_file(test_settings: Settings, fn_api, mock_data_file):
    set_global(test_settings)
    _test_table_chat_agent(
        fn_api=fn_api,
        tabular_data=mock_data_file,
    )


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_dataframe_blanks(
    test_settings: Settings, fn_api, mock_data_frame_blanks
):
    set_global(test_settings)
    _test_table_chat_agent(
        fn_api=fn_api,
        tabular_data=mock_data_frame_blanks,
    )


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_file_blanks(
    test_settings: Settings, fn_api, mock_data_file_blanks
):
    set_global(test_settings)
    _test_table_chat_agent(
        fn_api=fn_api,
        tabular_data=mock_data_file_blanks,
    )


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_url(test_settings: Settings, fn_api: bool) -> None:
    """
    Test the TableChatAgent with a dataframe as data source
    """
    set_global(test_settings)
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    agent = TableChatAgent(
        config=TableChatAgentConfig(
            data=URL,
            use_tools=not fn_api,
            use_functions_api=fn_api,
        )
    )

    task = Task(
        agent,
        name="TableChatAgent",
        only_user_quits_root=False,
        llm_delegate=True,
        single_round=False,
        default_human_response="",  # avoid waiting for human response
    )

    # run until LLM says DONE and shows answer,
    # at which point the task loop ends.

    result = task.run(
        """
        What is the average alcohol content of wines with a quality rating above 7?
        """,
        turns=5,
    )

    data = agent.df
    # Filter the dataset for wines with quality above 7
    high_quality_wines = data[data["quality"] > 7]

    # Compute the average alcohol content in this subset
    answer = high_quality_wines["alcohol"].mean()
    assert contains_approx_float(result.content, answer)
