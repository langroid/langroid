"""Test optional logger functionality in Task."""

from pathlib import Path

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.task import Task, TaskConfig
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity


def test_task_default_loggers_enabled() -> None:
    """Test that loggers are created by default."""
    # Create a simple mock agent
    llm_config = MockLMConfig(response_dict={"user": "Hello from test!"})
    agent_config = ChatAgentConfig(llm=llm_config, name="TestAgent")
    agent = ChatAgent(agent_config)

    # Default behavior - loggers should be created
    task_config = TaskConfig(logs_dir="test_logs")
    task = Task(agent, name="task_with_loggers", config=task_config)

    # Initialize the task to trigger logger creation
    task.init()

    # Check if loggers were created
    assert task.logger is not None
    assert task.tsv_logger is not None

    # Clean up log files
    log_path = Path("test_logs/task_with_loggers.log")
    tsv_path = Path("test_logs/task_with_loggers.tsv")
    if log_path.exists():
        log_path.unlink()
    if tsv_path.exists():
        tsv_path.unlink()

    # Clean up test_logs directory if empty
    test_logs_dir = Path("test_logs")
    if test_logs_dir.exists() and not any(test_logs_dir.iterdir()):
        test_logs_dir.rmdir()


def test_task_loggers_disabled() -> None:
    """Test that loggers are not created when enable_loggers=False."""
    # Create a simple mock agent
    llm_config = MockLMConfig(response_dict={"user": "Hello from test!"})
    agent_config = ChatAgentConfig(llm=llm_config, name="TestAgent")
    agent = ChatAgent(agent_config)

    # With loggers disabled
    task_config = TaskConfig(logs_dir="test_logs", enable_loggers=False)
    task = Task(agent, name="task_without_loggers", config=task_config)

    # Initialize the task - loggers should NOT be created
    task.init()

    # Check if loggers were NOT created
    assert task.logger is None
    assert task.tsv_logger is None


def test_log_message_with_none_loggers() -> None:
    """Test that log_message handles None loggers gracefully."""
    # Create a simple mock agent
    llm_config = MockLMConfig(response_dict={"user": "Hello from test!"})
    agent_config = ChatAgentConfig(llm=llm_config, name="TestAgent")
    agent = ChatAgent(agent_config)

    # With loggers disabled
    task_config = TaskConfig(logs_dir="test_logs", enable_loggers=False)
    task = Task(agent, name="task_without_loggers", config=task_config)

    # Initialize the task
    task.init()

    # Create a test message
    msg = ChatDocument(
        content="Test message", metadata=ChatDocMetaData(sender=Entity.USER)
    )

    # This should not raise any exceptions
    task.log_message(Entity.USER, msg)

    # If we get here without exception, the test passes
    assert True
