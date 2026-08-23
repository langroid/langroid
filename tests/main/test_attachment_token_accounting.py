"""Tests for attachment-aware context-preflight token accounting (issue #996).

The accounting itself (attachment payloads counted via their serialized
API form) is covered in `test_prep_llm_message.py`; here we test the
one-time warning emitted when attachments contribute to the token count,
and that accounting without attachments is unchanged.

Note: warnings are captured with a plain `logging.Handler` rather than
pytest's `caplog`, because CI disables the logging plugin
(`PYTEST_ADDOPTS="-p no:logging"`), which makes `caplog` unusable there.
"""

import logging
from contextlib import contextmanager
from typing import Iterator, List

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.parsing.file_attachment import FileAttachment

LOGGER_NAME = "langroid.agent.chat_agent"


@contextmanager
def capture_warnings(logger_name: str) -> Iterator[List[logging.LogRecord]]:
    """Collect WARNING+ records from `logger_name` without pytest's caplog."""
    records: List[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger(logger_name)
    handler = Collector(level=logging.WARNING)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


@pytest.fixture
def agent() -> ChatAgent:
    """ChatAgent with a char-count parser; no live LLM needed."""
    config = ChatAgentConfig(
        system_message="System message",
        llm=OpenAIGPTConfig(
            chat_model="gemini/gemini-2.5-flash",
            chat_context_length=16_000,
        ),
    )
    agent = ChatAgent(config)

    class MockParser:
        def num_tokens(self, text: str) -> int:
            return len(text)

    agent.parser = MockParser()  # type: ignore[assignment]
    return agent


def _user_msg_with_attachment() -> LLMMessage:
    attachment = FileAttachment.from_bytes(
        content=b"pdf-bytes" * 20,
        filename="dummy.pdf",
    )
    return LLMMessage(
        role=Role.USER,
        content="Question about the PDF",
        files=[attachment],
    )


def _attachment_warnings(
    records: List[logging.LogRecord],
) -> List[logging.LogRecord]:
    return [r for r in records if "attachment" in r.getMessage().lower()]


def test_attachment_warning_emitted_once(agent: ChatAgent) -> None:
    """Warning fires once per agent, even across repeated token counts."""
    msg = _user_msg_with_attachment()
    with capture_warnings(LOGGER_NAME) as records:
        agent.chat_num_tokens([msg])
        agent.chat_num_tokens([msg, msg])
        agent.chat_num_tokens([msg])

    warnings = _attachment_warnings(records)
    assert len(warnings) == 1
    assert "estimate" in warnings[0].getMessage().lower()


def test_no_warning_without_attachments(agent: ChatAgent) -> None:
    """No attachments: no warning, and counting is content-only as before."""
    msg = LLMMessage(role=Role.USER, content="Just text")
    with capture_warnings(LOGGER_NAME) as records:
        n_tokens = agent.chat_num_tokens([msg])

    assert n_tokens == len("Just text")
    assert _attachment_warnings(records) == []


def test_no_warning_for_non_user_attachment(agent: ChatAgent) -> None:
    """Attachments on non-user messages are not serialized inline, so they
    neither count nor warn."""
    attachment = FileAttachment.from_bytes(content=b"x" * 100, filename="f.pdf")
    msg = LLMMessage(
        role=Role.ASSISTANT,
        content="reply",
        files=[attachment],
    )
    with capture_warnings(LOGGER_NAME) as records:
        n_tokens = agent.chat_num_tokens([msg])

    assert n_tokens == len("reply")
    assert _attachment_warnings(records) == []
