import pytest
from sqlalchemy import Column, ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

from langroid.agent.special.sql.sql_chat_agent import SQLChatAgent, SQLChatAgentConfig
from langroid.agent.task import Task
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
    engine = create_engine("sqlite:///:memory:", echo=True)
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
    db_session: Session,
    context: dict,
    prompt: str,
    answer: str,
    use_schema_tools: bool = False,
    turns: int = 2,
) -> None:
    """
    Test the SQLChatAgent with a uri as data source
    """
    agent = SQLChatAgent(
        config=SQLChatAgentConfig(
            database_session=db_session,
            context_descriptions=context,
            use_tools=not fn_api,
            use_functions_api=fn_api,
            use_schema_tools=use_schema_tools,
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


@pytest.mark.parametrize(
    "fn_api,query,answer",
    [
        (True, "How many departments are there?", "2"),
        (False, "How many departments are there?", "2"),
        (True, "What is the total amount of sales?", "600"),
        (False, "What is the total amount of sales?", "600"),
        (True, "How many employees are in Sales?", "1"),
        (False, "How many employees are in Sales?", "1"),
    ],
)
def test_sql_chat_agent_query(
    test_settings: Settings,
    fn_api,
    mock_db_session,
    mock_context,
    query,
    answer,
):
    set_global(test_settings)
    # with context descriptions:
    _test_sql_chat_agent(
        fn_api=fn_api,
        db_session=mock_db_session,
        context=mock_context,
        prompt=query,
        answer=answer,
    )

    # without context descriptions:
    _test_sql_chat_agent(
        fn_api=fn_api,
        db_session=mock_db_session,
        context={},
        prompt=query,
        answer=answer,
    )


@pytest.mark.parametrize(
    "fn_api,query,answer",
    [
        (True, "How many departments are there?", "2"),
        (False, "How many departments are there?", "2"),
    ],
)
def test_sql_schema_tools(
    test_settings: Settings,
    fn_api,
    mock_db_session,
    mock_context,
    query,
    answer,
):
    set_global(test_settings)
    # with schema tools:
    _test_sql_chat_agent(
        fn_api=fn_api,
        db_session=mock_db_session,
        context=mock_context,
        prompt=query,
        answer=answer,
        use_schema_tools=True,
        turns=6,
    )
