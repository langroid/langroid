"""Tests for HTML logger functionality."""

import tempfile
from pathlib import Path

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task, TaskConfig
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity
from langroid.utils.html_logger import HTMLLogger


class TestHTMLLogger:
    """Test HTML logger basic functionality."""

    def test_html_logger_creation(self):
        """Test that HTML logger creates files correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = HTMLLogger(
                filename="test_log", log_dir=temp_dir, model_info="test-model-1.0"
            )

            # Check file was created
            log_path = Path(temp_dir) / "test_log.html"
            assert log_path.exists()

            # Check header content
            content = log_path.read_text()
            assert "<!DOCTYPE html>" in content
            assert "Langroid Task Log" in content
            assert "test_log" in content  # Check task name is in header

            logger.close()

    def test_html_logger_entries(self):
        """Test logging different types of entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = HTMLLogger(filename="test_entries", log_dir=temp_dir)

            # Create a user message
            from pydantic import create_model

            # Simulate what task.py does
            fields_dict1 = {
                "responder": "USER",
                "mark": "",
                "task_name": "root",
                "content": "Hello, how are you?",
                "sender_entity": Entity.USER,
                "sender_name": "",
                "recipient": "",
                "block": None,
                "tool_type": "",
                "tool": "",
            }
            LogFields1 = create_model(
                "LogFields", **{k: (type(v), v) for k, v in fields_dict1.items()}
            )
            log_obj1 = LogFields1(**fields_dict1)
            logger.log(log_obj1)

            # Log an assistant message with tool
            fields_dict2 = {
                "responder": "ASSISTANT",
                "mark": "",
                "task_name": "root",
                "content": '{"request": "search", "query": "weather"}',
                "sender_entity": Entity.LLM,
                "sender_name": "assistant",
                "recipient": "",
                "block": None,
                "tool_type": "TOOL",
                "tool": "search",
            }
            LogFields2 = create_model(
                "LogFields", **{k: (type(v), v) for k, v in fields_dict2.items()}
            )
            log_obj2 = LogFields2(**fields_dict2)
            logger.log(log_obj2)

            logger.close()

            # Check content
            log_path = Path(temp_dir) / "test_entries.html"
            content = log_path.read_text()

            # Check entries are present
            assert "USER" in content
            assert "Hello, how are you?" in content
            assert "ASSISTANT" in content

    def test_task_with_html_logger(self):
        """Test HTML logger integration with Task."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create agent with mock LLM
            config = ChatAgentConfig(
                llm=MockLMConfig(response_dict={"hello": "Hi there!"})
            )
            agent = ChatAgent(config)

            # Create task with HTML logging enabled
            task_config = TaskConfig(
                logs_dir=temp_dir, enable_html_logging=True, enable_loggers=True
            )
            task = Task(agent, name="test_task", config=task_config, interactive=False)

            # Run a simple interaction
            task.run("hello")

            # Close loggers
            task.close_loggers()

            # Check HTML log was created
            html_path = Path(temp_dir) / "test_task.html"
            assert html_path.exists()

            # Check content
            content = html_path.read_text()
            assert "USER" in content
            assert "hello" in content
            assert "ASSISTANT" in content or "LLM" in content
            assert "Hi there!" in content

    def test_html_special_characters(self):
        """Test HTML escaping of special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = HTMLLogger(filename="test_escape", log_dir=temp_dir)

            # Log message with HTML special characters
            from pydantic import create_model

            fields_dict = {
                "responder": "USER",
                "mark": "",
                "task_name": "root",
                "content": 'Test <script>alert("xss")</script> & entities',
                "sender_entity": Entity.USER,
                "sender_name": "",
                "recipient": "",
                "block": None,
                "tool_type": "",
                "tool": "",
            }
            LogFields = create_model(
                "LogFields", **{k: (type(v), v) for k, v in fields_dict.items()}
            )
            log_obj = LogFields(**fields_dict)
            logger.log(log_obj)
            logger.close()

            # Check content is properly escaped
            log_path = Path(temp_dir) / "test_escape.html"
            content = log_path.read_text()

            # Should be escaped
            assert "&lt;script&gt;" in content
            assert "&amp;" in content
            # Should not contain raw script
            assert "<script>alert" not in content
