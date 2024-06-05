"""
Specific tests of the Task class for infinite loops.
"""

from typing import Optional

import pytest

import langroid as lr
from langroid.agent import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.utils.configuration import settings

settings.stream = False


@pytest.mark.parametrize("loop_start", [0, 6])
@pytest.mark.parametrize(
    "cycle_len, max_cycle_len",
    [
        (3, 8),  # inf loop
        (5, 3),  # no inf loop
        (1000, 5),  # no inf loop
        (1, 5),  # inf loop
        (3, 0),  # no loop detection
    ],
)
@pytest.mark.parametrize("user_copy", [False, True])
def test_task_inf_loop(
    loop_start: int,
    cycle_len: int,
    max_cycle_len: int,
    user_copy: bool,  # should user response copy the message?
):
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

        def user_response(
            self,
            msg: Optional[str | ChatDocument] = None,
        ) -> Optional[ChatDocument]:
            """Mock user response"""
            if user_copy:
                content = msg if isinstance(msg, str) else msg.content
            else:
                content = "ok"
            return self.create_user_response(content)

    loop_agent = LoopAgent(ChatAgentConfig())
    task_config = lr.TaskConfig(
        inf_loop_cycle_len=max_cycle_len,
    )
    task = lr.Task(
        loop_agent,
        interactive=True,
        config=task_config,
    )

    # Test with a run that should raise the exception
    if cycle_len < max_cycle_len:  # i.e. an actual loop within the run
        with pytest.raises(lr.InfiniteLoopException):
            task.run(turns=80)
    else:
        # no loop within this many turns, so we shouldn't raise exception
        result = task.run(turns=80)
        assert result.metadata.status == lr.StatusCode.MAX_TURNS
