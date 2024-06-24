from typing import Optional

import pytest

import langroid as lr
from langroid import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task, TaskConfig
from langroid.language_models.mock_lm import MockLMConfig
from langroid.parsing.routing import parse_addressed_message
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import AT, DONE, SEND_TO

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
