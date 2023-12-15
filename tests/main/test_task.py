"""
Other tests for Task are in test_chat_agent.py
"""
import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER


@pytest.mark.parametrize("allow_null_result", [True, False])
def test_task_empty_response(test_settings: Settings, allow_null_result: bool):
    set_global(test_settings)
    agent = ChatAgent(ChatAgentConfig(name="Test"))
    task = Task(
        agent,
        interactive=False,
        single_round=True,
        allow_null_result=allow_null_result,
        system_message="""
        User will send you a number. 
        If it is EVEN, repeat the number, else return empty string.
        ONLY return these responses, say NOTHING ELSE
        """,
    )

    response = task.run("4")
    assert response.content == "4"
    response = task.run("3")
    assert response.content == NO_ANSWER
