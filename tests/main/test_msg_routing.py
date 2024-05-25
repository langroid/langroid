from typing import Optional

import pytest

from langroid import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import DONE, SEND_TO


@pytest.mark.parametrize(
    "address",
    [
        "@Alice ",
        "@Alice,",
        "@Alice:",
        f"{SEND_TO}Alice ",
        f"{SEND_TO}Alice:",
        f"{SEND_TO}Alice,",
    ],
)
def test_addressing(test_settings: Settings, address: str):
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
                return self.llm_response_template(DONE + " " + message.content)
            return self.llm_response_template(f"{address}what is the square of 10?")

    class AliceAgent(ChatAgent):
        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            return self.llm_response_template(f"{DONE} The square of 10 is 100.")

    bob_config = ChatAgentConfig(name="Bob")

    bob = BobAgent(bob_config)
    bob_task = Task(bob, interactive=False)

    alice_config = ChatAgentConfig(name="Alice")
    alice = AliceAgent(alice_config)
    alice_task = Task(alice, interactive=False)

    bob_task.add_sub_task(alice_task)

    result = bob_task.run()
    assert "100" in result.content
