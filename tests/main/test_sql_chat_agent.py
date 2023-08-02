import pytest

from langroid.agent.special.sql_chat_agent import SQLChatAgent, SQLChatAgentConfig
from langroid.agent.task import Task
from langroid.utils.configuration import Settings, set_global


@pytest.fixture
def mock_db() -> str:
    return "sqlite:///tests/main/sql_test.db"


@pytest.fixture
def mock_context() -> dict:
    return {
        "departments": {
            "description": "The 'departments' table holds details about the various "
            + "departments. It relates to the 'employees' table via a foreign key "
            + "in the 'employees' table.",
            "columns": {
                "id": "A unique identifier for a department. This ID is used as a "
                + "foreign key in the 'employees' table.",
                "name": "The name of the department.",
            },
        },
        "employees": {
            "description": "The 'employees' table contains information about the "
            + "employees. It relates to the 'departments' and 'sales' tables via "
            + "foreign keys.",
            "columns": {
                "id": "A unique identifier for an employee. This ID is used as a"
                + " foreign key in the 'sales' table.",
                "name": "The name of the employee.",
                "department_id": "The ID of the department the employee belongs to. "
                + "This is a foreign key referencing the 'id' in the 'departments'"
                + " table.",
            },
        },
        "sales": {
            "description": "The 'sales' table keeps a record of all sales made by "
            + "employees. It relates to the 'employees' table via a foreign key.",
            "columns": {
                "id": "A unique identifier for a sale.",
                "amount": "The amount of the sale.",
                "employee_id": "The ID of the employee who made the sale. This is a "
                + "foreign key referencing the 'id' in the 'employees' table.",
            },
        },
    }


def _test_sql_chat_agent(
    test_settings: Settings,
    fn_api: bool,
    mock_db: str,
    mock_context: dict,
    prompt: str,
    answer: str,
    turns: int = 2,
) -> None:
    """
    Test the SQLChatAgent with a uri as data source
    """
    set_global(test_settings)
    agent = SQLChatAgent(
        config=SQLChatAgentConfig(
            database_uri=mock_db,
            context_descriptions=mock_context,
            use_tools=not fn_api,
            use_functions_api=fn_api,
        )
    )

    task = Task(
        agent,
        name="SQLChatAgent",
        default_human_response="",  # avoid waiting for human response
        llm_delegate=False,
        single_round=False,
    )

    # run for 3 turns:
    # 0: user question
    # 1: LLM response via fun-call/tool
    # 2: agent response, handling the fun-call/tool
    result = task.run(prompt, turns=turns)

    assert answer in result.content


@pytest.mark.parametrize("fn_api", [True, False])
def test_sql_chat_agent_simple_query(fn_api, mock_db, mock_context):
    _test_sql_chat_agent(
        test_settings=Settings(),
        fn_api=fn_api,
        mock_db=mock_db,
        mock_context=mock_context,
        prompt="How many departments are there?",
        answer="2",
    )


@pytest.mark.parametrize("fn_api", [True, False])
def test_sql_chat_agent_complex_query(fn_api, mock_db, mock_context):
    _test_sql_chat_agent(
        test_settings=Settings(),
        fn_api=fn_api,
        mock_db=mock_db,
        mock_context=mock_context,
        prompt="What is the total amount of sales?",
        answer="27604",
    )
