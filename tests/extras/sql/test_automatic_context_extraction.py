import pytest
from pytest_mysql import factories as mysql_factories
from pytest_postgresql import factories as postgresql_factories
from sqlalchemy import Column, ForeignKey, Integer, String, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

from langroid.agent.special.sql.sql_chat_agent import (
    SQLChatAgent,
    SQLChatAgentConfig,
)
from langroid.utils.configuration import Settings, set_global

Base = declarative_base()


# Define your classes
class Department(Base):
    __tablename__ = "departments"
    __table_args__ = {"comment": "Table for storing department information"}

    id = Column(
        Integer, primary_key=True, comment="Unique identifier for the department"
    )
    name = Column(String(50), nullable=False, comment="Name of the department")

    employees = relationship("Employee", back_populates="department")


class Employee(Base):
    __tablename__ = "employees"
    __table_args__ = {"comment": "Table for storing employee information"}

    id = Column(Integer, primary_key=True, comment="Unique identifier for the employee")
    name = Column(String(50), nullable=False, comment="Name of the employee")
    department_id = Column(
        Integer,
        ForeignKey("departments.id"),
        nullable=False,
        comment="Foreign key to department table",
    )

    department = relationship("Department", back_populates="employees")
    sales = relationship("Sale", back_populates="employee")


class Sale(Base):
    __tablename__ = "sales"
    __table_args__ = {"comment": "Table for storing sales information"}

    id = Column(Integer, primary_key=True, comment="Unique identifier for the sale")
    amount = Column(Integer, nullable=False, comment="Sale amount")
    employee_id = Column(
        Integer,
        ForeignKey("employees.id"),
        nullable=False,
        comment="Foreign key to employee table",
    )

    employee = relationship("Employee", back_populates="sales")


def insert_test_data(session: Session) -> None:
    """Insert test data into the given database session."""
    sales_dept = Department(id=1, name="Sales")
    marketing_dept = Department(id=2, name="Marketing")

    alice = Employee(id=1, name="Alice", department=sales_dept)
    bob = Employee(id=2, name="Bob", department=marketing_dept)

    sale1 = Sale(id=1, amount=100, employee=alice)
    sale2 = Sale(id=2, amount=500, employee=bob)

    session.add_all([sales_dept, marketing_dept, alice, bob, sale1, sale2])
    session.commit()


# Simulate PostgreSQL database
postgresql_proc = postgresql_factories.postgresql_proc(port=None)
postgresql = postgresql_factories.postgresql("postgresql_proc")


@pytest.fixture(scope="function")
def postgresql_engine(postgresql) -> create_engine:
    """Create engine for the PostgreSQL database.

    Args:
        postgresql: The PostgreSQL fixture.

    Returns:
        An engine connected to the PostgreSQL database.
    """
    user = postgresql.info.user
    password = postgresql.info.password or ""
    host = postgresql.info.host
    port = postgresql.info.port
    dbname = postgresql.info.dbname
    url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    return create_engine(url)


@pytest.fixture(scope="function")
def mock_postgresql_session(postgresql_engine: create_engine) -> Session:
    """Create tables in the PostgreSQL database and add entries."""
    Base.metadata.create_all(postgresql_engine)

    # Adding example entries
    Session = sessionmaker(bind=postgresql_engine)
    session = Session()
    insert_test_data(session)

    yield session
    session.close()
    Base.metadata.drop_all(postgresql_engine)


# Simulate MySQL database
mysql_proc = mysql_factories.mysql_proc(
    host="localhost",
    port=3306,
    user="root",
)
mysql = mysql_factories.mysql("mysql_proc", dbname="test")


@pytest.fixture(scope="function")
def mysql_engine(mysql) -> create_engine:
    """Create engine for the MySQL database.
    Args:
        mysql_proc: The MySQL process fixture.
    Returns:
        An engine connected to the MySQL database.
    """
    host = "localhost"
    port = 3306
    user = "root"
    db = "test"
    url = f"mysql+pymysql://{user}@{host}:{port}/{db}"

    engine = create_engine(url)

    with engine.connect() as connection:
        result = connection.execute(text("SHOW DATABASES"))
        db_names = [row[0] for row in result]
        print("Databases:", db_names)

    return create_engine(url)


@pytest.fixture(scope="function")
def mock_mysql_session(mysql_engine: create_engine) -> Session:
    """Create tables in the MySQL database and add entries."""
    Base.metadata.create_all(mysql_engine)

    # Adding example entries
    Session = sessionmaker(bind=mysql_engine)
    session = Session()
    insert_test_data(session)

    yield session
    session.close()
    Base.metadata.drop_all(mysql_engine)


def _test_sql_automatic_context_extraction(
    test_settings: Settings,
    db_session: Session,
) -> None:
    """
    Test the SQLChatAgent with a uri as data source
    """
    set_global(test_settings)
    agent = SQLChatAgent(
        config=SQLChatAgentConfig(
            database_session=db_session,
        )
    )

    expected_context = {
        "departments": {
            "description": "Table for storing department information",
            "columns": {
                "id": "Unique identifier for the department",
                "name": "Name of the department",
            },
        },
        "employees": {
            "description": "Table for storing employee information",
            "columns": {
                "id": "Unique identifier for the employee",
                "name": "Name of the employee",
                "department_id": "Foreign key to department table",
            },
        },
        "sales": {
            "description": "Table for storing sales information",
            "columns": {
                "id": "Unique identifier for the sale",
                "amount": "Sale amount",
                "employee_id": "Foreign key to employee table",
            },
        },
    }
    print(agent.config.context_descriptions)
    assert agent.config.context_descriptions == expected_context


def test_postgresql_automatic_context_extraction(mock_postgresql_session):
    _test_sql_automatic_context_extraction(
        test_settings=Settings(),
        db_session=mock_postgresql_session,
    )


def test_mysql_automatic_context_extraction(mock_mysql_session):
    _test_sql_automatic_context_extraction(
        test_settings=Settings(),
        db_session=mock_mysql_session,
    )
