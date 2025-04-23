import pytest

from langroid.agent.task import Task
from langroid.exceptions import LangroidImportError
from langroid.language_models.openai_gpt import OpenAIGPTConfig

try:
    from sqlalchemy import Column, ForeignKey, Integer, String, create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import Session, relationship, sessionmaker
except ImportError as e:
    raise LangroidImportError(extra="sql", error=str(e))

from langroid.agent.special.sql.sql_chat_agent import (
    SQLChatAgent,
    SQLChatAgentConfig,
)
from langroid.utils.configuration import Settings, set_global

Base = declarative_base()


# Define your classes
class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    employees = relationship("Employee", back_populates="department")


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=False)

    department = relationship("Department", back_populates="employees")
    sales = relationship("Sale", back_populates="employee")


class Sale(Base):
    __tablename__ = "sales"

    id = Column(Integer, primary_key=True)
    amount = Column(Integer, nullable=False)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)

    employee = relationship("Employee", back_populates="sales")


@pytest.fixture
def mock_db_session() -> Session:
    # Create an in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    # Insert data
    sales_dept = Department(id=1, name="Sales")
    marketing_dept = Department(id=2, name="Marketing")

    alice = Employee(id=1, name="Alice", department=sales_dept)
    bob = Employee(id=2, name="Bob", department=marketing_dept)

    sale1 = Sale(id=1, amount=100, employee=alice)
    sale2 = Sale(id=2, amount=500, employee=bob)

    session.add(sales_dept)
    session.add(marketing_dept)
    session.add(alice)
    session.add(bob)
    session.add(sale1)
    session.add(sale2)

    session.commit()

    yield session  # this is where the fixture's value comes from

    session.close()


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
    fn_api: bool,
    tools_api: bool,
    json_schema: bool,
    db_session: Session,
    context: dict,
    prompt: str,
    answer: str,
    use_schema_tools: bool = False,
    turns: int = 18,
    addressing_prefix: str = "",
) -> None:
    """
    Test the SQLChatAgent with a uri as data source
    """
    agent_config = SQLChatAgentConfig(
        name="SQLChatAgent",
        database_session=db_session,
        context_descriptions=context,
        use_tools=not fn_api,
        use_functions_api=fn_api,
        use_tools_api=tools_api,
        use_schema_tools=use_schema_tools,
        addressing_prefix=addressing_prefix,
        chat_mode=False,
        use_helper=True,
        llm=OpenAIGPTConfig(supports_json_schema=json_schema),
    )
    agent = SQLChatAgent(agent_config)
    task = Task(agent, interactive=False)

    # run for enough turns to handle LLM deviations
    # 0: user question
    # 1: LLM response via fun-call/tool
    # 2: agent response, handling the fun-call/tool
    # ... so on
    result = task.run(prompt, turns=turns)

    assert answer.lower() in result.content.lower()


@pytest.mark.parametrize("fn_api", [False, True])
@pytest.mark.parametrize("tools_api", [False, True])
@pytest.mark.parametrize("json_schema", [False, True])
@pytest.mark.parametrize(
    "query,answer",
    [
        ("What is the total amount of sales?", "600"),
        ("How many employees are in Sales?", "1"),
        ("How many departments are there?", "2"),
    ],
)
def test_sql_chat_agent_query(
    test_settings: Settings,
    fn_api,
    tools_api,
    json_schema,
    mock_db_session,
    mock_context,
    query,
    answer,
):
    set_global(test_settings)
    # with context descriptions:
    _test_sql_chat_agent(
        fn_api=fn_api,
        tools_api=tools_api,
        db_session=mock_db_session,
        json_schema=json_schema,
        context=mock_context,
        prompt=query,
        answer=answer,
    )

    # without context descriptions:
    _test_sql_chat_agent(
        fn_api=fn_api,
        tools_api=tools_api,
        json_schema=json_schema,
        db_session=mock_db_session,
        context={},
        prompt=query,
        answer=answer,
    )


@pytest.mark.xfail(
    reason="May fail sometimes",
    strict=False,
    run=True,
)
@pytest.mark.parametrize("fn_api", [True, False])
@pytest.mark.parametrize("tools_api", [True, False])
@pytest.mark.parametrize("json_schema", [False, True])
def test_sql_chat_db_update(
    test_settings: Settings,
    fn_api,
    tools_api,
    json_schema,
    mock_db_session,
    mock_context,
):
    set_global(test_settings)
    # with context descriptions:
    _test_sql_chat_agent(
        fn_api=fn_api,
        tools_api=tools_api,
        json_schema=json_schema,
        db_session=mock_db_session,
        context=mock_context,
        prompt="Update Bob's sale amount to 900",
        answer="900",
    )

    _test_sql_chat_agent(
        fn_api=fn_api,
        tools_api=tools_api,
        json_schema=json_schema,
        db_session=mock_db_session,
        context=mock_context,
        prompt="How much did Bob sell?",
        answer="900",
    )

    # without context descriptions:
    _test_sql_chat_agent(
        fn_api=fn_api,
        tools_api=tools_api,
        json_schema=json_schema,
        db_session=mock_db_session,
        context={},
        prompt="Update Bob's sale amount to 9100",
        answer="9100",
    )

    _test_sql_chat_agent(
        fn_api=fn_api,
        tools_api=tools_api,
        json_schema=json_schema,
        db_session=mock_db_session,
        context={},
        prompt="How much did Bob sell?",
        answer="9100",
    )


@pytest.mark.parametrize("tools_api", [True, False])
@pytest.mark.parametrize("fn_api", [True, False])
@pytest.mark.parametrize("json_schema", [False, True])
@pytest.mark.parametrize(
    "query,answer",
    [
        ("How many departments are there?", "2"),
    ],
)
def test_sql_schema_tools(
    test_settings: Settings,
    fn_api,
    tools_api,
    json_schema,
    mock_db_session,
    mock_context,
    query,
    answer,
):
    set_global(test_settings)
    # with schema tools:
    _test_sql_chat_agent(
        fn_api=fn_api,
        tools_api=tools_api,
        json_schema=json_schema,
        db_session=mock_db_session,
        context=mock_context,
        prompt=query,
        answer=answer,
        use_schema_tools=True,
    )
