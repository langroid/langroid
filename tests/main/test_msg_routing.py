from typing import Optional

import pytest

from langroid import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import AT, DONE, SEND_TO


@pytest.mark.parametrize(
    "address",
    [
        AT + "Alice ",
        AT + "Alice,",
        AT + "Alice:",
        f"{SEND_TO}Alice ",
        f"{SEND_TO}Alice:",
        f"{SEND_TO}Alice,",
    ],
)
@pytest.mark.parametrize("x,answer", [(5, 25)])
def test_addressing(test_settings: Settings, address: str, x: int, answer: int):
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
            return self.create_llm_response(f"{address} {x}")

    class AliceAgent(ChatAgent):
        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            y = int(message.content.strip())
            answer = y * y
            return self.create_llm_response(f"{DONE} {answer}")

    bob_config = ChatAgentConfig(name="Bob")

    bob = BobAgent(bob_config)
    # When addressing Alice, set it to non-interactive, else interactive
    bob_task = Task(bob, interactive=False)

    alice_config = ChatAgentConfig(name="Alice")
    alice = AliceAgent(alice_config)
    alice_task = Task(alice, interactive=False)

    bob_task.add_sub_task(alice_task)

    result = bob_task.run()
    assert answer == int(result.content.strip())
