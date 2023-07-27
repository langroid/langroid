from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from langroid.agent.special.table_chat_agent import TableChatAgent, TableChatAgentConfig
from langroid.agent.task import Task
from langroid.utils.configuration import Settings, set_global


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

    data = {"age": ages, "gender": genders, "state": states_col, "income": incomes}

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
    test_settings: Settings,
    fn_api: bool,
    tabular_data: pd.DataFrame | str,
) -> None:
    """
    Test the TableChatAgent with a file as data source
    """
    set_global(test_settings)
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
        llm_delegate=False,
        single_round=False,
    )

    # run for 3 turns:
    # 0: user question
    # 1: LLM response via fun-call/tool
    # 2: agent response, handling the fun-call/tool
    result = task.run("What is the average income of people under 40 in CA?", turns=2)
    answer = agent.df.query("age < 40 and state == 'CA'")["income"].mean()

    assert np.round(float(result.content)) == np.round(answer)


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_dataframe(fn_api, mock_dataframe):
    _test_table_chat_agent(
        test_settings=Settings(),
        fn_api=fn_api,
        tabular_data=mock_dataframe,
    )


@pytest.mark.parametrize("fn_api", [True, False])
def test_table_chat_agent_file(fn_api, mock_data_file):
    _test_table_chat_agent(
        test_settings=Settings(),
        fn_api=fn_api,
        tabular_data=mock_data_file,
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
        default_human_response="",  # avoid waiting for human response
        llm_delegate=False,
        single_round=False,
    )

    # run for 3 turns:
    # 0: user question
    # 1: LLM response via fun-call/tool
    # 2: agent response, handling the fun-call/tool
    result = task.run(
        """
        What is the average alcohol content of wines with a quality rating above 7?
        """,
        turns=2,
    )

    data = agent.df
    # Filter the dataset for wines with quality above 7
    high_quality_wines = data[data["quality"] > 7]

    # Compute the average alcohol content in this subset
    answer = high_quality_wines["alcohol"].mean()

    assert np.round(float(result.content)) == np.round(answer)
