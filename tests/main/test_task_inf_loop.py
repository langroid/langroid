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
from langroid.utils.configuration import settings

settings.stream = False


@pytest.mark.parametrize("loop_start", [0, 10])
@pytest.mark.parametrize("cycle_len", [1000, 1, 3])
def test_task_inf_loop(loop_start: int, cycle_len: int):
    """Test that Task.run() can detect infinite loops"""

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
            if self.iter < loop_start:
                param = self.iter * 1000 + 100
            else:
                param = self.iter % cycle_len
            self.iter += 1
            return self.create_llm_response(
                f"""
                TOOL: {{"request": "dummy", "param": {param}}}
                """
            )

    loop_agent = LoopAgent(ChatAgentConfig())
    loop_agent.enable_message(DummyTool)
    task = Task(loop_agent, interactive=False)

    # Test with a run that should raise the exception
    if cycle_len < 1000:  # i.e. an actual loop within the run
        with pytest.raises(lr.InfiniteLoopException):
            task.run(turns=500)
    else:
        # no loop within this many turns, so we shouldn't raise exception
        result = task.run(turns=100)
        assert result.metadata.status == lr.StatusCode.MAX_TURNS
