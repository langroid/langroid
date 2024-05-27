"""
Specific tests of the Task class for infinite loops.
"""

from typing import Optional

import pytest

import langroid as lr
from langroid.agent import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.utils.configuration import settings

settings.stream = False


@pytest.mark.parametrize("loop_start", [0, 10])
@pytest.mark.parametrize("cycle_len", [1000, 1, 3])
def test_task_inf_loop(loop_start: int, cycle_len: int):
    """Test that Task.run() can detect infinite loops"""

    # set up an agent with a llm_response that produces cyclical output
    class LoopAgent(ChatAgent):
        iter: int = 0

        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            """Mock LLM response"""
            if self.iter < loop_start:
                param = self.iter * 1000 + 100
            else:
                param = self.iter % cycle_len
            self.iter += 1
            response = self.create_llm_response(str(param))
            self._render_llm_response(response)
            return response

    loop_agent = LoopAgent(ChatAgentConfig())
    task = Task(loop_agent, interactive=False)

    # Test with a run that should raise the exception
    if cycle_len < 1000:  # i.e. an actual loop within the run
        with pytest.raises(lr.InfiniteLoopException):
            task.run(turns=500)
    else:
        # no loop within this many turns, so we shouldn't raise exception
        result = task.run(turns=100)
        assert result.metadata.status == lr.StatusCode.MAX_TURNS
