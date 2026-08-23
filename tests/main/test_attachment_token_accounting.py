"""Tests for attachment-aware context-preflight token accounting (issue #996).

The accounting itself (attachment payloads counted via their serialized
API form) is covered in `test_prep_llm_message.py`; here we test the
one-time warning emitted when attachments contribute to the token count,
and that accounting without attachments is unchanged.
"""

import logging
from typing import List

import pytest
from _pytest.logging import LogCaptureFixture

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.parsing.file_attachment import FileAttachment

LOGGER_NAME = "langroid.agent.chat_agent"


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


def _attachment_warnings(caplog: LogCaptureFixture) -> List[logging.LogRecord]:
    return [r for r in caplog.records if "attachment" in r.getMessage().lower()]


def test_attachment_warning_emitted_once(
    agent: ChatAgent, caplog: LogCaptureFixture
) -> None:
    """Warning fires once per agent, even across repeated token counts."""
    msg = _user_msg_with_attachment()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        agent.chat_num_tokens([msg])
        agent.chat_num_tokens([msg, msg])
        agent.chat_num_tokens([msg])

    warnings = _attachment_warnings(caplog)
    assert len(warnings) == 1
    assert "estimate" in warnings[0].getMessage().lower()


def test_no_warning_without_attachments(
    agent: ChatAgent, caplog: LogCaptureFixture
) -> None:
    """No attachments: no warning, and counting is content-only as before."""
    msg = LLMMessage(role=Role.USER, content="Just text")
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        n_tokens = agent.chat_num_tokens([msg])

    assert n_tokens == len("Just text")
    assert _attachment_warnings(caplog) == []


def test_no_warning_for_non_user_attachment(
    agent: ChatAgent, caplog: LogCaptureFixture
) -> None:
    """Attachments on non-user messages are not serialized inline, so they
    neither count nor warn."""
    attachment = FileAttachment.from_bytes(content=b"x" * 100, filename="f.pdf")
    msg = LLMMessage(
        role=Role.ASSISTANT,
        content="reply",
        files=[attachment],
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        n_tokens = agent.chat_num_tokens([msg])

    assert n_tokens == len("reply")
    assert _attachment_warnings(caplog) == []
