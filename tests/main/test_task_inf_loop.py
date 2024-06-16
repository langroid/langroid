"""
Specific tests of the Task class for infinite loops.
"""

from random import choice
from typing import Optional

import pytest

import langroid as lr
from langroid.agent import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.mock_lm import MockLMConfig
from langroid.utils.configuration import settings
from langroid.utils.constants import NO_ANSWER

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
        assert result.metadata.status == lr.StatusCode.FIXED_TURNS


def test_task_stall():
    """Test that task.run() bails when stalled, i.e. no valid response
    for many steps."""

    agent = ChatAgent(
        ChatAgentConfig(
            name="Random",
            llm=MockLMConfig(
                response_fn=lambda x: choice([str(x) for x in range(30)]),
            ),
        )
    )

    # interactive=False, so in each step,
    # other than LLM, other responders have no response -> stalled
    task = lr.Task(agent, interactive=False)
    result = task.run(turns=100)
    assert result is None

    # set allow_null_result=True, so in each step, when no valid response is found,
    # we create a dummy NO_ANSWER response from the entity "opposite" to the author
    # of the pending message, i.e.
    # - if the author was LLM, then the entity is USER
    # - if the author was not LLM, then the entity is LLM
    # But this should result in an "alternating NA infinite loop", i.e.
    # LLM says x1, then USER says NA, then LLM says x2, then USER says NA, ...
    task = lr.Task(agent, restart=True, interactive=False, allow_null_result=True)
    with pytest.raises(lr.InfiniteLoopException):
        task.run(turns=100)


def test_task_alternating_no_answer():
    """Test that task.run() bails when there's a long enough
    alternation between NO_ANSWER and normal msg."""

    alice = ChatAgent(
        ChatAgentConfig(
            name="Alice",
            llm=MockLMConfig(response_fn=lambda x: choice([str(x) for x in range(50)])),
        )
    )

    alice_task = lr.Task(alice, interactive=True, default_human_response=NO_ANSWER)
    with pytest.raises(lr.InfiniteLoopException):
        alice_task.run(turns=100)

    alice_task = lr.Task(
        alice,
        restart=True,
        interactive=False,
    )
    # Alice keeps sending random msgs, Bob always says NO_ANSWER
    # This simulates an inf loop situation where Alice is asking various questions
    # and the sub-task responds with NO_ANSWER.
    bob = ChatAgent(
        ChatAgentConfig(
            name="Bob",
            llm=MockLMConfig(default_response=NO_ANSWER),
        )
    )

    bob_task = lr.Task(bob, interactive=False, single_round=True)
    alice_task.add_sub_task(bob_task)

    with pytest.raises(lr.InfiniteLoopException):
        alice_task.run(turns=100)
