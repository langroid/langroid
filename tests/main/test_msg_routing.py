from typing import List, Optional, Tuple

import pytest

import langroid as lr
from langroid import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData
from langroid.agent.task import Task, TaskConfig
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity
from langroid.parsing.routing import parse_addressed_message
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import AT, DONE, PASS, PASS_TO, SEND_TO

ADDRESSES = [
    AT + "Alice ",
    AT + "Alice,",
    AT + "Alice:",
    f"{SEND_TO}Alice ",
    f"{SEND_TO}Alice:",
    f"{SEND_TO}Alice,",
]


@pytest.mark.parametrize("address", ADDRESSES)
def test_parse_address(address: str):
    """Test that the address is parsed correctly."""
    msg = f"ok {AT}all, {AT}xyz here is my message to {address} -- {address} Hello"
    (addressee, content) = parse_addressed_message(
        msg,
        addressing=AT if AT in address else SEND_TO,
    )
    assert addressee == "Alice"
    assert content == "Hello"


@pytest.mark.parametrize(
    "msg,prefix,expected_addressee,expected_content",
    [
        # addressee is the final token, with no trailing character
        (f"thanks, now {AT}Alice", AT, "Alice", ""),
        (f"please reply {SEND_TO}Alice", SEND_TO, "Alice", ""),
        # multiple addressees, the last one at end-of-string: must pick the last
        (f"ok {AT}Bob then {AT}Alice", AT, "Alice", ""),
        (f"ask {AT}Bob then {AT}Alice, where?", AT, "Alice", "where?"),
        # whole message is just the address
        (f"{AT}Alice", AT, "Alice", ""),
    ],
)
def test_parse_address_at_end_of_string(
    msg: str, prefix: str, expected_addressee: str, expected_content: str
):
    """An addressee appearing as the final token (no trailing char) must still be
    recognized, and the *last* addressee must win even when it ends the string."""
    (addressee, content) = parse_addressed_message(msg, addressing=prefix)
    assert addressee == expected_addressee
    assert content == expected_content


@pytest.mark.parametrize(
    "msg,prefix,expected_addressee,expected_content",
    [
        # addressee names are maximal runs of word chars, which are
        # Unicode-aware (`\w`), so accented/non-Latin names must be recognized
        (f"{AT}Ålice", AT, "Ålice", ""),
        (f"ok {AT}日本語", AT, "日本語", ""),
        (f"{AT}Müller_9, hello", AT, "Müller_9", "hello"),
        # underscores and digits are word chars: the whole run is the name
        (f"go {AT}agent_42", AT, "agent_42", ""),
        (f"{SEND_TO}agent_007 do it", SEND_TO, "agent_007", "do it"),
        # the *last* occurrence wins, even when the final addressee ends the
        # string and uses Unicode/underscore/digit edge characters
        (f"{AT}Bob then {AT}zoë_9", AT, "zoë_9", ""),
        (f"{SEND_TO}Bob, now {SEND_TO}agent_1: go", SEND_TO, "agent_1", "go"),
    ],
)
def test_parse_address_grammar(
    msg: str, prefix: str, expected_addressee: str, expected_content: str
):
    """Addressee names are maximal Unicode word-character runs (`\\w+`)."""
    (addressee, content) = parse_addressed_message(msg, addressing=prefix)
    assert addressee == expected_addressee
    assert content == expected_content


@pytest.mark.parametrize(
    "msg,prefix",
    [
        # no addressing prefix at all
        ("no address here", AT),
        # prefix not followed by a word char
        (f"{AT} Alice", AT),
        (f"{SEND_TO} Alice", SEND_TO),
        # prefix with no name, at end of string
        (f"hello {AT}", AT),
        (f"ping {SEND_TO}", SEND_TO),
    ],
)
def test_parse_address_no_match(msg: str, prefix: str):
    """When the prefix is absent or not followed by a word-character run,
    the parser must return (None, <original content unchanged>)."""
    (addressee, content) = parse_addressed_message(msg, addressing=prefix)
    assert addressee is None
    assert content == msg


@pytest.mark.parametrize("prefix", [AT, SEND_TO])
@pytest.mark.parametrize("suffix", ["", ".", ","])
def test_bare_address_routes_as_pass_through(prefix: str, suffix: str):
    """A responder ending with a bare address (e.g. "@Alice" / "__SEND__:Alice"
    with no following content) is a pass-through: the pending message must be
    forwarded to the recipient. _process_result_routing must therefore set the
    recipient AND normalize the content to PASS, so that step() (which checks
    `PASS in result.content`) recognizes the pass-through instead of forwarding
    the literal address string."""
    agent = ChatAgent(ChatAgentConfig(name="Bob"))
    task = Task(agent, interactive=False, config=TaskConfig(addressing_prefix=AT))
    task.pending_message = ChatDocument(
        content="the original question",
        metadata=ChatDocMetaData(sender=Entity.USER),
    )
    result = ChatDocument(
        content=f"ok, over to you {prefix}Alice{suffix}",
        metadata=ChatDocMetaData(sender=Entity.LLM),
    )
    out = task._process_result_routing(result, Entity.LLM)
    assert out is not None
    # recipient recorded on the message that will be passed through
    assert task.pending_message.metadata.recipient == "Alice"
    # content normalized so step() recognizes the pass-through
    assert out.content == PASS


class NudgeTool(lr.ToolMessage):
    request: str = "nudge_tool"
    purpose: str = "To nudge an agent."
    who: str


@pytest.mark.parametrize("prefix", [AT, SEND_TO])
def test_bare_address_with_tool_attempt_is_not_routed(prefix: str):
    """A message carrying a tool attempt must NOT be treated as a bare-address
    pass-through: normalizing its content to PASS would make step() forward the
    pending message instead, silently dropping the tool call. Such a message
    must emerge from _process_result_routing with content and recipients
    unchanged, and its tool attempt intact."""
    agent = ChatAgent(ChatAgentConfig(name="Bob"))
    agent.enable_message(NudgeTool)
    task = Task(agent, interactive=False, config=TaskConfig(addressing_prefix=AT))
    task.pending_message = ChatDocument(
        content="the original question",
        metadata=ChatDocMetaData(sender=Entity.USER),
    )
    content = f"{prefix}Alice"
    result = ChatDocument(
        content=content,
        tool_messages=[NudgeTool(who="Alice")],
        metadata=ChatDocMetaData(sender=Entity.LLM),
    )
    assert agent.has_tool_message_attempt(result)
    out = task._process_result_routing(result, Entity.LLM)
    assert out is not None
    # content and recipients unchanged; tool attempt preserved
    assert out.content == content
    assert out.metadata.recipient == ""
    assert task.pending_message.metadata.recipient == ""
    assert out.tool_messages == [NudgeTool(who="Alice")]


ORIGINAL_QUESTION = "the original question"


def _task_with_pending(recognize_string_signals: bool = True) -> Task:
    """Build a non-interactive Task with a pending USER message.

    Args:
        recognize_string_signals: value for
            `TaskConfig.recognize_string_signals`.

    Returns:
        Task: task with `pending_message` set to a USER message.
    """
    agent = ChatAgent(ChatAgentConfig(name="Bob"))
    task = Task(
        agent,
        interactive=False,
        config=TaskConfig(
            addressing_prefix=AT,
            recognize_string_signals=recognize_string_signals,
        ),
    )
    task.pending_message = ChatDocument(
        content=ORIGINAL_QUESTION,
        metadata=ChatDocMetaData(sender=Entity.USER),
    )
    return task


def _llm_doc(content: str) -> ChatDocument:
    """Make a ChatDocument as if it were an LLM response."""
    return ChatDocument(
        content=content,
        metadata=ChatDocMetaData(sender=Entity.LLM),
    )


@pytest.mark.parametrize("content", [PASS, f"intro {PASS} outro"])
def test_plain_pass_content_unchanged(content: str):
    """Content containing plain PASS (no recipient) must be left verbatim:
    the bare-address normalization must not fire, and no recipient may be
    set on the pending message or on the result."""
    task = _task_with_pending()
    out = task._process_result_routing(_llm_doc(content), Entity.LLM)
    assert out is not None
    assert out.content == content
    assert out.metadata.recipient == ""
    assert task.pending_message is not None
    assert task.pending_message.metadata.recipient == ""


def test_pass_to_recipient_content_not_clobbered():
    """A `PASS_TO:<recipient>` result must set the pass-through recipient on
    the pending message, and since its content already contains PASS (so
    step() recognizes the pass-through), the bare-address normalization must
    NOT rewrite it."""
    task = _task_with_pending()
    out = task._process_result_routing(_llm_doc(f"{PASS_TO}Alice"), Entity.LLM)
    assert out is not None
    assert task.pending_message is not None
    assert task.pending_message.metadata.recipient == "Alice"
    # step() must still see this as a pass-through...
    assert PASS in out.content
    # ...and the normalization (which is only for bare addresses) must not
    # have overwritten the PASS_TO form
    assert out.content == f"{PASS_TO}Alice"


def test_bare_address_without_pending_message_unchanged():
    """With no pending message there is nothing to pass through: a
    bare-address result must be returned unchanged (no PASS normalization,
    no recipient set)."""
    agent = ChatAgent(ChatAgentConfig(name="Bob"))
    task = Task(agent, interactive=False, config=TaskConfig(addressing_prefix=AT))
    assert task.pending_message is None
    content = f"over to you {AT}Alice"
    out = task._process_result_routing(_llm_doc(content), Entity.LLM)
    assert out is not None
    assert out.content == content
    assert out.metadata.recipient == ""


@pytest.mark.parametrize("prefix", [AT, SEND_TO])
def test_addressed_content_keeps_send_semantics(prefix: str):
    """An address followed by non-empty content is a SEND, not a
    pass-through: the content (with the address stripped) must be forwarded
    to the recipient on the result itself; it must not be normalized to
    PASS, and the pending message must be left untouched."""
    task = _task_with_pending()
    out = task._process_result_routing(
        _llm_doc(f"{prefix}Alice tell me more"), Entity.LLM
    )
    assert out is not None
    assert out.content == "tell me more"
    assert PASS not in out.content
    assert out.metadata.recipient == "Alice"
    assert task.pending_message is not None
    assert task.pending_message.metadata.recipient == ""


@pytest.mark.parametrize(
    "content",
    [f"over to you {AT}Alice", f"{SEND_TO}Alice", PASS],
)
def test_routing_ignored_when_string_signals_disabled(content: str):
    """With `TaskConfig(recognize_string_signals=False)`, no string-based
    routing may fire: content and recipients must be left unchanged."""
    task = _task_with_pending(recognize_string_signals=False)
    out = task._process_result_routing(_llm_doc(content), Entity.LLM)
    assert out is not None
    assert out.content == content
    assert out.metadata.recipient == ""
    assert task.pending_message is not None
    assert task.pending_message.metadata.recipient == ""


@pytest.mark.parametrize("prefix", [AT, SEND_TO])
@pytest.mark.parametrize("suffix", ["", ".", ","])
def test_bare_address_pass_through_task(prefix: str, suffix: str):
    """Task-level pass-through (sync): an LLM response that is a bare address
    must cause the ORIGINAL pending message -- not the literal address string,
    and not PASS -- to be forwarded to the addressee, who then answers it."""
    alice_received: List[Tuple[str, str]] = []

    class BobAgent(ChatAgent):
        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            if (
                isinstance(message, ChatDocument)
                and message.metadata.sender_name == "Alice"
            ):
                return self.create_llm_response(f"{DONE} {message.content}")
            return self.create_llm_response(f"over to you {prefix}Alice{suffix}")

    class AliceAgent(ChatAgent):
        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            assert isinstance(message, ChatDocument)
            alice_received.append((message.content, message.metadata.recipient))
            return self.create_llm_response(
                f"{DONE} Alice-answer to [{message.content}]"
            )

    bob_task = Task(
        BobAgent(ChatAgentConfig(name="Bob")),
        interactive=False,
        config=TaskConfig(addressing_prefix=AT),
    )
    alice_task = Task(AliceAgent(ChatAgentConfig(name="Alice")), interactive=False)
    bob_task.add_sub_task(alice_task)

    result = bob_task.run(ORIGINAL_QUESTION)

    # the addressee received the original pending message exactly once, with
    # herself as recipient -- not the literal address string, and not PASS
    assert alice_received == [(ORIGINAL_QUESTION, "Alice")]
    # the final result is Alice's answer to the original question
    assert result is not None
    assert result.content == f"Alice-answer to [{ORIGINAL_QUESTION}]"
    assert AT not in result.content
    assert PASS not in result.content


@pytest.mark.parametrize("prefix", [AT, SEND_TO])
@pytest.mark.parametrize("suffix", ["", ".", ","])
@pytest.mark.asyncio
async def test_bare_address_pass_through_task_async(prefix: str, suffix: str):
    """Task-level pass-through (async): same contract as the sync version."""
    alice_received: List[Tuple[str, str]] = []

    class BobAgent(ChatAgent):
        async def llm_response_async(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            if (
                isinstance(message, ChatDocument)
                and message.metadata.sender_name == "Alice"
            ):
                return self.create_llm_response(f"{DONE} {message.content}")
            return self.create_llm_response(f"over to you {prefix}Alice{suffix}")

    class AliceAgent(ChatAgent):
        async def llm_response_async(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            assert isinstance(message, ChatDocument)
            alice_received.append((message.content, message.metadata.recipient))
            return self.create_llm_response(
                f"{DONE} Alice-answer to [{message.content}]"
            )

    bob_task = Task(
        BobAgent(ChatAgentConfig(name="Bob")),
        interactive=False,
        config=TaskConfig(addressing_prefix=AT),
    )
    alice_task = Task(AliceAgent(ChatAgentConfig(name="Alice")), interactive=False)
    bob_task.add_sub_task(alice_task)

    result = await bob_task.run_async(ORIGINAL_QUESTION)

    assert alice_received == [(ORIGINAL_QUESTION, "Alice")]
    assert result is not None
    assert result.content == f"Alice-answer to [{ORIGINAL_QUESTION}]"
    assert AT not in result.content
    assert PASS not in result.content


def _stalling_pass_through_task() -> Task:
    """Task whose LLM always responds with a bare self-address, so every step
    is a pass-through that makes no progress."""
    agent = ChatAgent(
        ChatAgentConfig(
            name="Solo",
            llm=MockLMConfig(default_response=f"{AT}Solo"),
        )
    )
    return Task(
        agent,
        interactive=False,
        max_stalled_steps=3,
        config=TaskConfig(addressing_prefix=AT),
    )


def test_repeated_bare_address_pass_through_stalls():
    """Repeated bare-address pass-throughs that make no progress must
    increment the stalled-step counter and end the task via
    `max_stalled_steps` (sync)."""
    task = _stalling_pass_through_task()
    # cap turns so a stall-accounting regression fails fast instead of hanging
    result = task.run(ORIGINAL_QUESTION, turns=20)
    assert result is None  # stalled task has no result
    assert task.n_stalled_steps == task.max_stalled_steps
    # the pending message was never replaced by the literal address
    assert task.pending_message is not None
    assert task.pending_message.content == ORIGINAL_QUESTION
    assert task.pending_message.metadata.recipient == "Solo"


@pytest.mark.asyncio
async def test_repeated_bare_address_pass_through_stalls_async():
    """Repeated bare-address pass-throughs must be bounded by
    `max_stalled_steps` in the async path too."""
    task = _stalling_pass_through_task()
    result = await task.run_async(ORIGINAL_QUESTION, turns=20)
    assert result is None
    assert task.n_stalled_steps == task.max_stalled_steps
    assert task.pending_message is not None
    assert task.pending_message.content == ORIGINAL_QUESTION
    assert task.pending_message.metadata.recipient == "Solo"


@pytest.mark.parametrize("prefix", [AT, ""])  # enable AT-addressing?
@pytest.mark.parametrize(
    "address",
    ADDRESSES,
)
@pytest.mark.parametrize("x,answer", [(5, 25)])
def test_addressing(
    test_settings: Settings, prefix: str, address: str, x: int, answer: int
):
    """Test that an agent is able to address another agent in a message."""
    set_global(test_settings)

    class BobAgent(ChatAgent):
        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            if (
                isinstance(message, ChatDocument)
                and message.metadata.sender_name == "Alice"
            ):
                return self.create_llm_response(DONE + " " + message.content)

            addr = AT if AT in address else SEND_TO
            # throw in some distracting addresses, to test that
            # only the last one is picked up
            return self.create_llm_response(
                f"Ok {addr}all here {addr}Junk is my question: {address} {x}"
            )

    class AliceAgent(ChatAgent):
        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            # message.content will either be just an an int-string "5"
            # (if prefix != "") or Bob's entire msg otherwise (and hence not an int)
            try:
                y = int(message.content.strip())
            except ValueError:
                return None
            answer = y * y
            return self.create_llm_response(f"{DONE} {answer}")

    bob_config = ChatAgentConfig(name="Bob")

    bob = BobAgent(bob_config)
    bob_task = Task(
        bob,
        interactive=False,
        config=TaskConfig(addressing_prefix=prefix),
    )

    alice_config = ChatAgentConfig(name="Alice")
    alice = AliceAgent(alice_config)
    alice_task = Task(alice, interactive=False)

    bob_task.add_sub_task(alice_task)

    result = bob_task.run()
    if prefix == "" and AT in address:
        assert result is None
    else:
        assert answer == int(result.content.strip())


class MockAgent(ChatAgent):
    def user_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        """
        Mock user_response method for testing
        """
        txt = msg if isinstance(msg, str) else msg.content
        map = dict([("2", "3"), ("3", "5")])
        response = map.get(txt)
        # return the increment of input number
        return self.create_user_response(response)


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("prefix", [AT, SEND_TO])
@pytest.mark.parametrize("addressee", ["user", "User", "USER"])
def test_user_addressing(interactive: bool, prefix: str, addressee: str):
    """Test that when LLM addresses user explicitly, the user
    is allowed to respond, regardless of interactive mode"""

    address = prefix + addressee
    agent = lr.ChatAgent(
        ChatAgentConfig(
            name="Mock",
            llm=MockLMConfig(default_response=f"Ok here we go {address} give a number"),
        )
    )
    task = lr.Task(
        agent,
        interactive=interactive,
        default_human_response=f"{DONE} 1",
        config=TaskConfig(addressing_prefix=AT),
    )
    result = task.run()
    assert "1" in result.content


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("prefix", [AT, SEND_TO])
@pytest.mark.parametrize("addressee", ["user", "User", "USER"])
def test_no_addressing(interactive: bool, prefix: str, addressee: str):
    """Test that when a Task is configured with TaskConfig.addressing_prefix = ''
    (the default), then no routing is recognized. This ensures there is no
    "accidental" addressing due to presence of route-line characters in the message.
    Note the TaskConfig.address_prefix only affects whether "@"-like addressing is
    recognized; it does not affect whether SEND_TO is recognized; SEND_TO-based routing
    is always enabled, as this is a key mechanism by which a response from an entity
    can direct the msg to another entity.
    """

    address = prefix + addressee
    agent = lr.ChatAgent(
        ChatAgentConfig(
            name="Mock",
            llm=MockLMConfig(default_response=f"Ok here we go {address} give a number"),
        )
    )
    task = lr.Task(
        agent,
        interactive=interactive,
        default_human_response=f"{DONE} 1",
    )
    result = task.run()
    if interactive or prefix == SEND_TO:
        assert "1" in result.content  # user gets chance anyway, without addressing
    else:
        assert result is None  # user not explicitly addressed, so they can't respond
