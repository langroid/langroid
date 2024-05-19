"""
Simple test of a stateful tool: enabling this tool on an agent
allows it to change the agent's state.
"""

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global


class IncrementTool(ToolMessage):
    request: str = "increment"
    purpose: str = "To increment my number by an <amount>."
    amount: int

    def handle(self, agent: ChatAgent) -> str:
        agent.number += self.amount
        return str(agent.number)


class NumberGameAgent(ChatAgent):
    def __init__(self, config: ChatAgentConfig):
        super().__init__(config)
        self.number = 0

    def increment(self, msg: IncrementTool) -> str:
        """
        Increments the agent's number by the amount specified in the message.
        Args:
            msg (IncrementTool): The message containing the amount to increment by.
        Returns:
            str: The agent's number after incrementing.
        """
        return msg.handle(self)


@pytest.mark.parametrize("fn_api", [True, False])
def test_stateful_tool(test_settings: Settings, fn_api: bool):
    set_global(test_settings)
    number_game_agent = NumberGameAgent(
        ChatAgentConfig(
            name="Gamer",
            llm=OpenAIGPTConfig(),
            vecdb=None,
            use_tools=not fn_api,
            use_functions_api=fn_api,
        )
    )

    number_game_agent.enable_message(IncrementTool)
    task = Task(
        number_game_agent,
        interactive=False,
        system_message="""
            I have a number in mind. Your job is to keep incrementing
            it by 5 using the `increment` tool, and I will tell you the result.
            Once you have reached 25 or more, you can say DONE and show me the result.
        """,
    )
    result = task.run()
    assert "25" in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("fn_api", [True, False])
async def test_stateful_tool_async(test_settings: Settings, fn_api: bool):
    set_global(test_settings)
    number_game_agent = NumberGameAgent(
        ChatAgentConfig(
            name="Gamer",
            llm=OpenAIGPTConfig(),
            vecdb=None,
            use_tools=not fn_api,
            use_functions_api=fn_api,
        )
    )

    number_game_agent.enable_message(IncrementTool)
    task = Task(
        number_game_agent,
        interactive=False,
        system_message="""
            I have a number in mind. Your job is to keep incrementing
            it by 5 using the `increment` tool, and I will tell you the result.
            Once you have reached 25 or more, you can say DONE and show me the result.
        """,
    )
    result = await task.run_async()
    assert "25" in result.content
