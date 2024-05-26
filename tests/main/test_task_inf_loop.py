"""
Specific tests of the Task class for infinite loops.
"""

from typing import Optional

import pytest

import langroid as lr
from langroid.agent import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.utils.configuration import Settings, set_global


def test_task_inf_loop(test_settings: Settings):
    """Test that Task.run() can detect infinite loops"""
    set_global(test_settings)

    class DummyTool(ToolMessage):
        request = "dummy"
        purpose = "Dummy tool for testing"
        param: int

        def handle(self) -> str:
            return f"got param {self.param}"

    # set up an agent with a tool so it alternates between
    # agent_response and llm_response forever
    class LoopAgent(ChatAgent):
        iter: int = 0

        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            # for first 10 iters, return tool with a different param
            self.iter += 1
            if self.iter < 10:
                return self.create_llm_response(
                    f"""
                    TOOL: {{"request": "dummy", "param": {self.iter}}}
                    """
                )
            # ... after that return a fixed response
            return self.create_llm_response(
                """
                TOOL: {"request": "dummy", "param": 10}
                """
            )

    loop_agent = LoopAgent(ChatAgentConfig())
    loop_agent.enable_message(DummyTool)
    task = Task(loop_agent, interactive=False)

    # Test with a run that should raise the exception
    with pytest.raises(lr.InfiniteLoopException):
        task.run(turns=500)
