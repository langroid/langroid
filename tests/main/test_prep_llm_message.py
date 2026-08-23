import json
from typing import Any

import pytest

from langroid.agent.chat_agent import (
    ChatAgent,
    ChatAgentConfig,
    _llm_message_has_payload,
)
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.language_models.base import (
    LLMFunctionCall,
    LLMMessage,
    LLMResponse,
    OpenAIToolCall,
    Role,
)
from langroid.language_models.mock_lm import MockLM, MockLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.parsing.file_attachment import FileAttachment

CHAT_CONTEXT_LENGTH = 16_000
MAX_OUTPUT_TOKENS = 1000
MIN_OUTPUT_TOKENS = 50


@pytest.fixture
def agent():
    """Create a ChatAgent with a mock LLM for testing truncation."""
    config = ChatAgentConfig(
        system_message="System message",
        llm=OpenAIGPTConfig(
            # Small context for testing truncation
            chat_context_length=CHAT_CONTEXT_LENGTH,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            min_output_tokens=MIN_OUTPUT_TOKENS,
        ),
    )
    agent = ChatAgent(config)

    # Create a mock parser that counts tokens as characters for simplicity
    class MockParser:
        def num_tokens(self, text: str | LLMMessage):
            if isinstance(text, str):
                return len(text)
            else:
                return len(text.content)

        def truncate_tokens(self, text, tokens, warning=""):
            return text[:tokens] + warning

    agent.parser = MockParser()

    # Create a mock LLM that returns a fixed context length
    class MockLLM:
        def chat_context_length(self) -> int:
            return CHAT_CONTEXT_LENGTH

        def get_stream(self) -> bool:
            return False

        def supports_functions_or_tools(self) -> bool:
            return False

    agent.llm = MockLLM()

    # Initialize message history with a system message
    # agent.message_history = [LLMMessage(role=Role.SYSTEM, content="System message")]
    agent.init_message_history()
    return agent


def test_no_truncation_needed(agent):
    """Test when no truncation is needed."""
    # Add a short user message (well within context limits)
    message = "Short user message"

    # Call the method
    hist, output_len = agent._prep_llm_messages(message)

    # History should include system message and the new user message
    assert len(hist) == 2
    assert hist[0].content == "System message"
    assert hist[1].content == message
    assert output_len == MAX_OUTPUT_TOKENS  # Original max output tokens


def test_reduce_output_length(agent):
    """Test when only output length reduction is needed."""
    # Fill most of the context with long messages
    long_message = "X" * 15_000  # 700 tokens
    agent.message_history.append(LLMMessage(role=Role.USER, content=long_message))

    # New user message
    message = "Another message"

    # Call the method
    hist, output_len = agent._prep_llm_messages(message)

    # Check that output length was reduced but no messages were truncated
    assert len(hist) == 3
    assert hist[1].content == long_message  # Not truncated
    assert output_len < MAX_OUTPUT_TOKENS  # Output length was reduced


def test_truncate_messages(agent):
    """Test when message truncation is needed."""

    # Fill the context with messages that will require truncation
    agent.message_history = [LLMMessage(role=Role.SYSTEM, content="System message")]

    # Add several messages that will need truncation
    for i in range(3):
        agent.message_history.append(
            LLMMessage(role=Role.USER, content=f"User message {i+1} " + "X" * 8_000)
        )
        agent.message_history.append(
            LLMMessage(role=Role.ASSISTANT, content=f"Assistant reply {i+1}")
        )

    orig_msg_len = len(agent.message_history[1].content)
    # Call the method
    hist, output_len = agent._prep_llm_messages("Final message")

    # Check that early messages were truncated
    assert len(hist) == 8  # All messages still present
    assert len(hist[1].content) < orig_msg_len
    # First user message truncated
    assert "Contents truncated" in hist[1].content
    assert output_len >= MIN_OUTPUT_TOKENS  # At least min_output_tokens


@pytest.fixture
def agent_drop_turns():
    """Create a ChatAgent with drop_turns strategy for testing."""
    config = ChatAgentConfig(
        system_message="System message",
        context_overflow_strategy="drop_turns",
        llm=OpenAIGPTConfig(
            # Small context for testing truncation
            chat_context_length=CHAT_CONTEXT_LENGTH,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            min_output_tokens=MIN_OUTPUT_TOKENS,
        ),
    )
    agent = ChatAgent(config)

    # Create a mock parser that counts tokens as characters for simplicity
    class MockParser:
        def num_tokens(self, text: str | LLMMessage):
            if isinstance(text, str):
                return len(text)
            else:
                return len(text.content)

        def truncate_tokens(self, text, tokens, warning=""):
            return text[:tokens] + warning

    agent.parser = MockParser()

    # Create a mock LLM that returns a fixed context length
    class MockLLM:
        def chat_context_length(self):
            return CHAT_CONTEXT_LENGTH

        def supports_functions_or_tools(self):
            return False

    agent.llm = MockLLM()

    agent.init_message_history()
    return agent


def test_drop_turns_strategy(agent_drop_turns):
    """Test when drop_turns strategy is used to handle context overflow."""
    agent = agent_drop_turns

    # Fill the context with messages that will require dropping turns
    agent.message_history = [LLMMessage(role=Role.SYSTEM, content="System message")]

    # Add several complete turns that will need to be dropped
    for i in range(3):
        agent.message_history.append(
            LLMMessage(role=Role.USER, content=f"User message {i+1} " + "X" * 8_000)
        )
        agent.message_history.append(
            LLMMessage(role=Role.ASSISTANT, content=f"Assistant reply {i+1}")
        )

    orig_hist_len = len(agent.message_history)
    # Call the method
    hist, output_len = agent._prep_llm_messages("Final message")

    # Check that turns were dropped (fewer messages than original)
    assert len(hist) < orig_hist_len
    # System message should still be present
    assert hist[0].role == Role.SYSTEM
    assert hist[0].content == "System message"
    # The last user message should be present
    assert hist[-1].role == Role.USER
    assert hist[-1].content == "Final message"
    # Check alternating pattern is preserved
    for i in range(1, len(hist) - 1, 2):
        assert hist[i].role == Role.USER
        assert hist[i + 1].role == Role.ASSISTANT
    assert output_len >= MIN_OUTPUT_TOKENS


def test_chat_num_tokens_counts_attachment_payload(agent):
    """Test attachment payloads are included in chat token accounting."""
    model = "gemini/gemini-2.5-flash"
    agent.config.llm.chat_model = model
    attachment = FileAttachment.from_bytes(
        content=b"pdf-bytes" * 20,
        filename="dummy.pdf",
    )
    message = LLMMessage(
        role=Role.USER,
        content="Question about the PDF",
        files=[attachment],
    )

    expected_attachment_tokens = len(
        json.dumps(
            attachment.to_dict(model),
            separators=(",", ":"),
            sort_keys=True,
        )
    )

    assert agent.chat_num_tokens([message]) == (
        len(message.content) + expected_attachment_tokens
    )


def test_attachment_payload_reduces_output_length():
    """Test preflight shrinks output length when attachments consume context."""
    context_length = 1100
    max_output_tokens = 500
    min_output_tokens = 50
    config = ChatAgentConfig(
        system_message="System message",
        llm=OpenAIGPTConfig(
            chat_model="gemini/gemini-2.5-flash",
            chat_context_length=context_length,
            max_output_tokens=max_output_tokens,
            min_output_tokens=min_output_tokens,
        ),
    )
    agent = ChatAgent(config)

    class MockParser:
        def num_tokens(self, text: str | LLMMessage):
            if isinstance(text, str):
                return len(text)
            return len(text.content)

        def truncate_tokens(self, text, tokens, warning=""):
            return text[:tokens] + warning

    class MockLLM:
        def chat_context_length(self):
            return context_length

        def supports_functions_or_tools(self):
            return False

    agent.parser = MockParser()
    agent.llm = MockLLM()
    agent.init_message_history()

    attachment = FileAttachment.from_bytes(
        content=b"x" * 400,
        filename="dummy.pdf",
    )
    user_input = ChatDocument(
        content="Question about the PDF",
        files=[attachment],
        metadata=ChatDocMetaData(sender=Entity.USER),
    )

    hist, output_len = agent._prep_llm_messages(user_input)

    assert output_len < max_output_tokens
    assert output_len == context_length - agent.chat_num_tokens(hist) - 300
    assert hist[-1].content == user_input.content


def test_drop_turns_preserves_last_turn(agent_drop_turns):
    """Test that drop_turns preserves the system message and last turn."""
    agent = agent_drop_turns

    # Set up history with multiple turns
    agent.message_history = [LLMMessage(role=Role.SYSTEM, content="System message")]

    # Add turns with large content that will force dropping
    for i in range(4):
        agent.message_history.append(
            LLMMessage(role=Role.USER, content=f"User {i+1} " + "Y" * 6_000)
        )
        agent.message_history.append(
            LLMMessage(role=Role.ASSISTANT, content=f"Assistant {i+1}")
        )

    # Call the method with a final message
    hist, output_len = agent._prep_llm_messages("Final user message")

    # System message must be preserved
    assert hist[0].role == Role.SYSTEM
    # Last message must be the final user message
    assert hist[-1].content == "Final user message"
    # No message should contain "Contents truncated" (we drop, not truncate)
    for msg in hist:
        assert "Contents truncated" not in msg.content


def test_drop_turns_accounts_for_buffer():
    """
    Test that drop_turns loop accounts for CHAT_HISTORY_BUFFER.

    This is a regression test for a P1 bug where the loop would exit when:
        tokens <= context - min_output_tokens
    But then output_len = context - tokens - CHAT_HISTORY_BUFFER could go
    negative, causing spurious errors.

    The fix ensures the loop continues until there's room for both
    min_output_tokens AND CHAT_HISTORY_BUFFER.
    """
    # CHAT_HISTORY_BUFFER is 300 in the code
    # We need to create a scenario where history is in the "danger zone":
    # between (context - min_output - buffer) and (context - min_output)
    #
    # With context=16000, min_output=50, buffer=300:
    # - Old buggy threshold: 16000 - 50 = 15950
    # - Fixed threshold: 16000 - 50 - 300 = 15650
    # - Danger zone: 15650 < tokens <= 15950

    config = ChatAgentConfig(
        system_message="S" * 100,  # 100 tokens
        context_overflow_strategy="drop_turns",
        llm=OpenAIGPTConfig(
            chat_context_length=16_000,
            max_output_tokens=1000,
            min_output_tokens=50,
        ),
    )
    agent = ChatAgent(config)

    class MockParser:
        def num_tokens(self, text: str | LLMMessage):
            if isinstance(text, str):
                return len(text)
            return len(text.content)

        def truncate_tokens(self, text, tokens, warning=""):
            return text[:tokens] + warning

    agent.parser = MockParser()

    class MockLLM:
        def chat_context_length(self):
            return 16_000

        def supports_functions_or_tools(self):
            return False

    agent.llm = MockLLM()
    agent.init_message_history()

    # Create history that lands in the danger zone after some turns
    # System msg = 100 tokens
    # We want total around 15800-15900 tokens (in danger zone)
    # Add turns that will require the buffer-aware loop to drop them
    agent.message_history = [LLMMessage(role=Role.SYSTEM, content="S" * 100)]

    # Add turns: each turn is ~5000 tokens (USER 4980 + ASSISTANT 20)
    # 3 turns = ~15000 + system 100 = ~15100
    # Final message ~800 = ~15900 total (in danger zone)
    for i in range(3):
        agent.message_history.append(
            LLMMessage(role=Role.USER, content=f"U{i}" + "X" * 4978)
        )
        agent.message_history.append(
            LLMMessage(role=Role.ASSISTANT, content=f"A{i}" + "Y" * 18)
        )

    # This should NOT raise an error - the fix ensures we drop enough turns
    # to accommodate both min_output_tokens AND CHAT_HISTORY_BUFFER
    hist, output_len = agent._prep_llm_messages("Z" * 800)

    # output_len must be positive and at least min_output_tokens
    assert output_len >= 50, f"output_len={output_len} should be >= 50"
    # History should have been compressed
    assert hist[0].role == Role.SYSTEM
    assert hist[-1].role == Role.USER


# ---------------------------------------------------------------------------
# Regression tests for "missing" (None) vs "empty" ("") message content.
#
# Gemini 3.x rejects (400 INVALID_ARGUMENT) an assistant turn that carries BOTH
# a tool/function call AND non-empty text content. Langroid used to pad empty
# content with a single space (" "), which tripped this on every tool-result
# turn. Content is now carried faithfully: a response with no content becomes
# LLMMessage.content=None (dropped from the wire), distinct from "".
# ---------------------------------------------------------------------------

MODEL = "gpt-4o"


def _tool_call():
    return OpenAIToolCall(
        id="call_1",
        type="function",
        function=LLMFunctionCall(name="check_weather", arguments={"city": "London"}),
    )


def test_api_dict_omits_content_for_tool_call_message():
    """An assistant tool-call turn with no content omits `content` on the wire
    (never emits " ", which Gemini 3.x rejects alongside a tool call)."""
    d = LLMMessage(
        role=Role.ASSISTANT, content=None, tool_calls=[_tool_call()]
    ).api_dict(MODEL)
    assert "content" not in d
    assert "tool_calls" in d


def test_api_dict_omits_content_for_function_call_message():
    """A legacy function-call-only turn also omits `content` on the wire."""
    function_call = LLMFunctionCall(
        name="check_weather",
        arguments={"city": "London"},
    )

    message = LLMMessage(
        role=Role.ASSISTANT,
        content=None,
        function_call=function_call,
    ).api_dict(MODEL)

    assert "content" not in message
    assert message["function_call"]["name"] == "check_weather"


def test_api_dict_pads_empty_message_without_tools():
    """A message with empty content (None or "") and no tool/function call must
    still send something, since some APIs (e.g. Gemini) reject an empty msg."""
    assert LLMMessage(role=Role.USER, content=None).api_dict(MODEL)["content"] == " "
    assert LLMMessage(role=Role.USER, content="").api_dict(MODEL)["content"] == " "


def test_api_dict_keeps_real_content():
    d = LLMMessage(role=Role.USER, content="hello").api_dict(MODEL)
    assert d["content"] == "hello"


def test_missing_content_is_none_end_to_end():
    """A tool-only LLM response (message=None) is carried faithfully as
    content_is_none -> LLMMessage.content=None -> omitted from api_dict."""
    resp = LLMResponse(message=None, oai_tool_calls=[_tool_call()])
    doc = ChatDocument.from_LLMResponse(resp)
    assert doc.content == ""  # ChatDocument.content stays a mandatory str
    assert doc.content_is_none is True

    msg = ChatDocument.to_LLMMessage(doc)[0]
    assert msg.role == Role.ASSISTANT
    assert msg.content is None
    assert msg.tool_calls is not None
    assert "content" not in msg.api_dict(MODEL)


def test_empty_content_stays_empty_not_none():
    """A present-but-empty response (message="") is distinct from missing:
    it must NOT be coerced to None."""
    resp = LLMResponse(message="", oai_tool_calls=[_tool_call()])
    doc = ChatDocument.from_LLMResponse(resp)
    assert doc.content_is_none is False

    msg = ChatDocument.to_LLMMessage(doc)[0]
    assert msg.content == ""


def test_none_content_round_trips_through_chatdocument():
    """LLMMessage.content=None survives an LLMMessage -> ChatDocument ->
    LLMMessage round-trip (history is rebuilt this way)."""
    original = LLMMessage(role=Role.ASSISTANT, content=None, tool_calls=[_tool_call()])
    doc = ChatDocument.from_LLMMessage(original)
    assert doc.content_is_none is True

    rebuilt = ChatDocument.to_LLMMessage(doc)[0]
    assert rebuilt.content is None


def test_prep_retains_fresh_tool_only_message(agent: ChatAgent) -> None:
    """History preparation retains a fresh call-only assistant turn."""
    doc = ChatDocument.from_LLMResponse(
        LLMResponse(message=None, oai_tool_calls=[_tool_call()])
    )

    messages, _ = agent._prep_llm_messages(doc)

    assert messages[-1].content is None
    assert messages[-1].tool_calls == [_tool_call()]


def test_render_tool_only_response_uses_empty_display_content(
    agent: ChatAgent,
) -> None:
    """Rendering safely separates absent text from displayed tool-call data."""
    displayed: dict[str, object] = {}

    def show_llm_response(
        content: str,
        tools_content: str,
        is_tool: bool,
        cached: bool,
        reasoning: str,
    ) -> None:
        displayed.update(
            content=content,
            tools_content=tools_content,
            is_tool=is_tool,
            cached=cached,
            reasoning=reasoning,
        )

    agent.callbacks.show_llm_response = show_llm_response
    response = LLMResponse(message=None, oai_tool_calls=[_tool_call()])

    agent._render_llm_response(response)

    assert displayed["content"] == ""
    assert "check_weather" in str(displayed["tools_content"])


def test_none_content_is_safe_for_token_counting(agent):
    """Token counting treats missing content as zero text tokens."""
    message = LLMMessage(
        role=Role.ASSISTANT,
        content=None,
        tool_calls=[_tool_call()],
    )

    assert agent.chat_num_tokens([message]) == 0


def test_truncate_tool_only_message_preserves_none(agent):
    """Truncation must not add warning text alongside a tool call."""
    agent.message_history.append(
        LLMMessage(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[_tool_call()],
        )
    )

    truncated = agent.truncate_message(-1)

    assert truncated.content is None
    assert "content" not in truncated.api_dict(MODEL)


@pytest.mark.parametrize(
    ("role", "identifying_field"),
    [
        (Role.TOOL, {"tool_call_id": "call_1"}),
        (Role.FUNCTION, {"name": "check_weather"}),
    ],
)
def test_truncate_empty_result_preserves_wire_payload(
    agent: ChatAgent,
    role: Role,
    identifying_field: dict[str, str],
) -> None:
    """Truncation must preserve a present-but-empty result."""
    message = LLMMessage(role=role, content="", **identifying_field)
    agent.message_history.append(message)
    original_payload = message.api_dict(MODEL)

    truncated = agent.truncate_message(-1)

    assert truncated.content == ""
    assert truncated.api_dict(MODEL) == original_payload


@pytest.mark.parametrize(
    "payload_field",
    [
        {"function_call": LLMFunctionCall(name="check_weather", arguments={})},
        {"tool_calls": [_tool_call()]},
    ],
)
@pytest.mark.parametrize(
    ("role", "identifying_field"),
    [
        (Role.TOOL, {"tool_call_id": "call_1"}),
        (Role.FUNCTION, {"name": "check_weather"}),
    ],
)
def test_truncate_blank_identified_result_with_existing_payload(
    agent: ChatAgent,
    role: Role,
    identifying_field: dict[str, str],
    payload_field: dict[str, Any],
) -> None:
    """Blank results already retained by call payloads remain truncatable."""
    agent.message_history.append(
        LLMMessage(
            role=role,
            content="",
            **identifying_field,
            **payload_field,
        )
    )

    truncated = agent.truncate_message(-1)

    assert truncated.content == "\n...[Contents truncated!]"


@pytest.mark.parametrize("role", [Role.TOOL, Role.FUNCTION])
def test_truncate_blank_result_without_identifier(
    agent: ChatAgent,
    role: Role,
) -> None:
    """Malformed blank results remain truncation candidates."""
    agent.message_history.append(LLMMessage(role=role, content=""))

    truncated = agent.truncate_message(-1)

    assert truncated.content == "\n...[Contents truncated!]"


@pytest.mark.parametrize("role", [Role.USER, Role.ASSISTANT])
def test_truncate_whitespace_message_for_non_result_role(
    agent: ChatAgent,
    role: Role,
) -> None:
    """Whitespace-only conversation messages remain truncation candidates."""
    content = " " * 10_000
    agent.message_history.append(LLMMessage(role=role, content=content))

    truncated = agent.truncate_message(-1)

    assert truncated.content != content
    assert truncated.content.endswith("...[Contents truncated!]")


def test_content_is_none_overrides_populated_content_any():
    """A call-only turn stays content=None even when content_any was populated
    (e.g. by _load_output_format parsing the tool args under a strict
    output_format). Otherwise the serialized args would be sent as assistant
    text alongside tool_calls, which is the Gemini 3.x rejection this avoids."""
    doc = ChatDocument(
        content="",
        content_is_none=True,
        # parsed structured output (tool args), NOT message text:
        content_any={"city": "London"},
        oai_tool_calls=[_tool_call()],
        metadata=ChatDocMetaData(sender=Entity.LLM, source=Entity.LLM),
    )
    msg = ChatDocument.to_LLMMessage(doc)[0]
    assert msg.content is None
    assert "content" not in msg.api_dict(MODEL)


def test_content_is_none_overrides_stale_text_after_serialization():
    """The explicit missing-content flag wins after a hostile round-trip."""
    doc = ChatDocument(
        content="stale text",
        content_with_reasoning="<thinking>stale reasoning</thinking>",
        content_is_none=True,
        oai_tool_calls=[_tool_call()],
        metadata=ChatDocMetaData(sender=Entity.LLM, source=Entity.LLM),
    )
    reloaded = ChatDocument.model_validate(doc.model_dump())

    msg = ChatDocument.to_LLMMessage(reloaded)[0]

    assert msg.content is None
    assert msg.tool_calls == [_tool_call()]
    assert "content" not in msg.api_dict(MODEL)


@pytest.mark.parametrize("content", ["", " " * CHAT_CONTEXT_LENGTH])
def test_prep_keeps_legitimately_empty_tool_result(
    agent: ChatAgent,
    content: str,
) -> None:
    """A tool result with genuinely empty content (e.g. a tool that succeeds
    with no output) must stay in history and clear the pending tool call from
    agent.oai_tool_calls, not get silently dropped (issue #1063). Before the
    fix, the history filter kept a message only if it had non-empty content, a
    function_call, or tool_calls -- none of which a TOOL-role result carries,
    so an empty-content result was dropped and its call_id never reached
    `done_tools`, leaving the call marked pending forever."""
    pending_call = _tool_call()
    agent.message_history.append(
        LLMMessage(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[pending_call],
        )
    )
    agent.oai_tool_calls = [pending_call]
    doc = ChatDocument(
        content=content,
        metadata=ChatDocMetaData(
            sender=Entity.AGENT, source=Entity.AGENT, oai_tool_id="call_1"
        ),
    )

    hist, output_len = agent._prep_llm_messages(doc)

    tool_msgs = [m for m in hist if m.role == Role.TOOL]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call_1"
    assert tool_msgs[0].content == ""
    assert agent.oai_tool_calls == []
    assert output_len == MAX_OUTPUT_TOKENS


def test_prep_keeps_legitimately_empty_function_result(
    agent: ChatAgent,
) -> None:
    """An empty legacy function result must stay in message history."""
    function_name = "check_weather"
    function_call = LLMFunctionCall(
        name=function_name,
        arguments={"city": "London"},
    )
    parent = ChatDocument(
        content="",
        content_is_none=True,
        function_call=function_call,
        metadata=ChatDocMetaData(sender=Entity.LLM, source=Entity.LLM),
    )
    agent.message_history.append(
        LLMMessage(
            role=Role.ASSISTANT,
            content=None,
            function_call=function_call,
        )
    )
    result = ChatDocument(
        content="",
        metadata=ChatDocMetaData(
            sender=Entity.AGENT,
            source=Entity.AGENT,
            parent_id=parent.id(),
        ),
    )

    hist, _ = agent._prep_llm_messages(result)

    function_msgs = [m for m in hist if m.role == Role.FUNCTION]
    assert len(function_msgs) == 1
    assert function_msgs[0].name == function_name


def test_prep_empty_result_completes_only_matching_tool_call(
    agent: ChatAgent,
) -> None:
    """An identified empty result completes only its matching pending call."""
    first_call = _tool_call()
    second_call = _tool_call()
    second_call.id = "call_2"
    agent.message_history.append(
        LLMMessage(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[first_call, second_call],
        )
    )
    agent.oai_tool_calls = [first_call, second_call]
    result = ChatDocument(
        content="",
        metadata=ChatDocMetaData(
            sender=Entity.AGENT,
            source=Entity.AGENT,
            oai_tool_id="call_2",
        ),
    )

    hist, _ = agent._prep_llm_messages(result)

    tool_messages = [message for message in hist if message.role == Role.TOOL]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_2"
    assert agent.oai_tool_calls == [first_call]


class RecordingMockLM(MockLM):
    """Mock LM that records the messages supplied to its chat method."""

    def __init__(self, config: MockLMConfig) -> None:
        super().__init__(config)
        self.received_messages: list[LLMMessage] = []

    def chat(
        self,
        messages: str | list[LLMMessage],
        *args: Any,
        **kwargs: Any,
    ) -> LLMResponse:
        """Record messages before returning the configured local response."""
        assert isinstance(messages, list)
        self.received_messages = messages
        return super().chat(messages, *args, **kwargs)


def test_llm_response_sends_empty_tool_result_to_lm() -> None:
    """llm_response sends an empty correlated result and returns a document."""
    llm_config = MockLMConfig(default_response="tool result received")
    agent = ChatAgent(
        ChatAgentConfig(
            system_message="System message",
            llm=llm_config,
        )
    )
    recording_llm = RecordingMockLM(llm_config)
    agent.llm = recording_llm
    agent.init_message_history()
    pending_call = _tool_call()
    agent.message_history.append(
        LLMMessage(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[pending_call],
        )
    )
    agent.oai_tool_calls = [pending_call]
    result = ChatDocument(
        content="",
        metadata=ChatDocMetaData(
            sender=Entity.AGENT,
            source=Entity.AGENT,
            oai_tool_id="call_1",
        ),
    )

    response = agent.llm_response(result)

    assert isinstance(response, ChatDocument)
    tool_messages = [
        message
        for message in recording_llm.received_messages
        if message.role == Role.TOOL
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_1"
    assert tool_messages[0].content == ""


def test_prep_drops_empty_tool_result_without_id(agent: ChatAgent) -> None:
    """An empty tool result without a correlation ID is malformed."""
    pending_call = _tool_call()
    pending_call.id = None
    agent.oai_tool_calls = [pending_call]
    doc = ChatDocument(
        content="",
        metadata=ChatDocMetaData(sender=Entity.AGENT, source=Entity.AGENT),
    )

    hist, output_len = agent._prep_llm_messages(doc)

    assert hist == []
    assert output_len == 0
    assert agent.oai_tool_calls == [pending_call]


def test_prep_drops_empty_function_result_without_name(
    agent: ChatAgent,
) -> None:
    """An empty function result without its function name is malformed."""
    function_call = LLMFunctionCall.model_construct(
        name=None,
        arguments={"city": "London"},
    )
    parent = ChatDocument(
        content="",
        content_is_none=True,
        function_call=function_call,
        metadata=ChatDocMetaData(sender=Entity.LLM, source=Entity.LLM),
    )
    result = ChatDocument(
        content="",
        metadata=ChatDocMetaData(
            sender=Entity.AGENT,
            source=Entity.AGENT,
            parent_id=parent.id(),
        ),
    )

    hist, output_len = agent._prep_llm_messages(result)

    assert hist == []
    assert output_len == 0


@pytest.mark.parametrize("role", [Role.TOOL, Role.FUNCTION])
@pytest.mark.parametrize(
    ("field_state", "identifier"),
    [
        ("none", None),
        ("empty", ""),
        ("space", " "),
        ("whitespace", "\t\n"),
    ],
)
def test_empty_result_requires_identifying_field(
    role: Role,
    field_state: str,
    identifier: str | None,
) -> None:
    """Malformed empty results are not valid history entries."""
    field = "tool_call_id" if role == Role.TOOL else "name"
    message = LLMMessage.model_construct(role=role, content="")
    setattr(message, field, identifier)

    assert not _llm_message_has_payload(message)


@pytest.mark.parametrize("role", [Role.TOOL, Role.FUNCTION])
@pytest.mark.parametrize("identifier", ["", " ", "\t\n"])
def test_prep_drops_empty_result_with_malformed_identifier(
    agent: ChatAgent,
    role: Role,
    identifier: str,
) -> None:
    """Whitespace identifiers cannot complete or remove pending calls."""
    valid_call = _tool_call()
    agent.oai_tool_calls = [valid_call]
    if role == Role.TOOL:
        malformed_call = _tool_call()
        malformed_call.id = identifier
        agent.message_history.append(
            LLMMessage(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[malformed_call],
            )
        )
        agent.oai_tool_calls.append(malformed_call)
        result = ChatDocument(
            content="",
            metadata=ChatDocMetaData(
                sender=Entity.AGENT,
                source=Entity.AGENT,
                oai_tool_id=identifier,
            ),
        )
    else:
        malformed_function = LLMFunctionCall(
            name=identifier,
            arguments={"city": "London"},
        )
        agent.message_history.append(
            LLMMessage(
                role=Role.ASSISTANT,
                content=None,
                function_call=malformed_function,
            )
        )
        parent = ChatDocument(
            content="",
            content_is_none=True,
            function_call=malformed_function,
            metadata=ChatDocMetaData(sender=Entity.LLM, source=Entity.LLM),
        )
        result = ChatDocument(
            content="",
            metadata=ChatDocMetaData(
                sender=Entity.AGENT,
                source=Entity.AGENT,
                parent_id=parent.id(),
            ),
        )

    pending_calls = list(agent.oai_tool_calls)
    hist, output_len = agent._prep_llm_messages(result)

    assert all(message.role != role for message in hist)
    assert output_len == 0
    assert agent.oai_tool_calls == pending_calls
